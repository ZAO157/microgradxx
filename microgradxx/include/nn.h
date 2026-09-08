#pragma once
/**
 * @file nn.h
 * @brief The neural-net layer on top of engine.h's autodiff core: Neuron,
 * Layer (a collection of Neurons), and MLP (a stack of Layers), following
 * the same shape as the original Python micrograd's nn.py.
 */

#include "engine.h"
#include <vector>
#include <random>
#include <iostream>
#include <optional>

#ifndef MICROGRAD_DEFAULT_SEED
#define MICROGRAD_DEFAULT_SEED 0
#endif
#ifndef MICROGRAD_DEFAULT_BUFFER_THRESHOLD
#define MICROGRAD_DEFAULT_BUFFER_THRESHOLD 4
#endif

namespace microgradxx {
    using namespace engine;
    namespace nn {
        // Shared RNG for weight initialization across every Neuron<T>. `inline`
        // (not `static`) is required here since this is a header: `static` at
        // namespace scope gives internal linkage, meaning every .cpp file that
        // includes this header would silently get its own independent
        // generator -- set_seed() in one translation unit would have no effect
        // on any other. `inline` guarantees a single shared instance program-wide.
        inline std::mt19937 generator(MICROGRAD_DEFAULT_SEED);

        /**
         * @brief Seeds the shared weight-initialization RNG used by every
         * Neuron<T>.
         * @param seed the seed value.
         */
        inline void set_seed(unsigned int seed = MICROGRAD_DEFAULT_SEED) {
            generator.seed(seed);
        }

        /**
         * @brief A single neuron: `nin` weights, one bias, optionally
         * followed by ReLU.
         * @tparam T the underlying scalar type.
         */
        template <class T>
        class Neuron {
        private:
            inline static std::uniform_real_distribution<T> uniform{-1.0, 1.0};
            std::vector<Value<T>> w;
            Value<T> b;
            bool nonlin;

        public:
            /**
             * @brief Constructs a neuron with randomly-initialized weights
             * and a zero bias.
             * @param nin number of inputs (weights).
             * @param _nonlin if true, applies ReLU after the weighted sum.
             */
            Neuron(std::size_t nin, bool _nonlin = true) :
                b(static_cast<T>(0)),
                nonlin(_nonlin) {
                w.reserve(nin);
                for (std::size_t i = 0; i < nin; i++) w.emplace_back(uniform(generator));
            }

            /**
             * @brief Forward pass, as a single fused graph node
             * (Value<T>::neuron) instead of nin chained Mul/Add nodes --
             * see NeuronValueImpl in engine.h.
             * @param x this neuron's input, one value per weight.
             * @return the neuron's output.
             */
            Value<T> operator()(const std::vector<Value<T>>& x) {
                return Value<T>::neuron(w, x, b, nonlin);
            }

            /**
             * @brief Same as the one-argument operator(), but `x`'s data()
             * values have already been gathered into `x_data` by the
             * caller (see Layer::operator()).
             * @param x this neuron's input, one value per weight.
             * @param x_data `x`'s data() values, already gathered into one
             * contiguous buffer.
             * @return the neuron's output.
             */
            Value<T> operator()(const std::vector<Value<T>>& x, const T* x_data) {
                return Value<T>::neuron(w, x, b, nonlin, x_data);
            }

            /// Resets every weight's and the bias's gradient to zero.
            void zero_grad() {
                for (auto &v : w) v.zero_grad();
                b.zero_grad();
            }

            /**
             * @brief This neuron's trainable parameters: all weights, then
             * the bias.
             * @return the parameters.
             */
            std::vector<Value<T>> parameters() {
                std::vector<Value<T>> out = w;
                out.push_back(b);
                return out;
            }

            /**
             * @brief Streams a human-readable description of `n`.
             * @param os the output stream.
             * @param n the neuron to print.
             * @return `os`.
             */
            friend std::ostream& operator<<(std::ostream& os, const Neuron<T>& n) {
                return os << (n.nonlin ? "ReLU" : "Linear") << "Neuron(" << n.w.size() << ")";
            }
        };

        /**
         * @brief A layer: `nout` independently-initialized Neurons, each
         * taking the same `nin`-sized input and producing one output -- so
         * calling a Layer maps an nin-vector to an nout-vector.
         * @tparam T the underlying scalar type.
         */
        template <class T>
        class Layer {
        private:
            std::vector<Neuron<T>> neurons;
            inline static std::size_t buffer_threshold = MICROGRAD_DEFAULT_BUFFER_THRESHOLD;

            // Pointer-to-*member*-function, not a plain function pointer:
            // forward_buffered/forward_unbuffer are ordinary (non-static)
            // member functions -- they read this->neurons -- so calling
            // through this needs the `this->*forward` syntax below, and
            // assigning to it needs `&Layer<T>::forward_buffered` (a
            // qualified member-function pointer), not just
            // `&forward_buffered`.
            std::vector<Value<T>>(Layer<T>::*forward)(const std::vector<Value<T>>& x);

            /**
             * @brief Forward pass for layers at or above buffer_threshold:
             * gathers `x` once for the whole layer first. See
             * ENGINE_NOTES.md for why that's worth the extra gather.
             * @param x this layer's input.
             * @return one output value per neuron.
             */
            std::vector<Value<T>> forward_buffered(const std::vector<Value<T>>& x) {
                // Gather x once for the whole layer using the AVX2
                // pointer-gather in ValueImpl<T>::gather() (via
                // Value<T>::gather_data()), instead of a scalar loop over
                // x[i].data(). No #pragma omp parallel for here anymore: a
                // gather is already a handful of SIMD instructions covering
                // 4 elements each, and x is typically small (this codebase's
                // actual layers have nin=3-4) -- spinning up worker threads
                // for that would cost far more in thread-launch overhead
                // than the single-threaded SIMD pass itself takes.
                std::vector<T> x_data(x.size());
                Value<T>::gather_data(x, x_data.data());

                std::vector<std::optional<Value<T>>> out(neurons.size());
            #pragma omp parallel for
                for (std::size_t i = 0; i < neurons.size(); i++) out[i] = neurons[i](x, x_data.data());

                // unwrap back into a plain vector<Value<T>>, single-threaded (no
                // race: the parallel region above already joined, via the
                // implicit barrier at the end of each #pragma omp parallel for)
                std::vector<Value<T>> result;
                result.reserve(out.size());
                for (auto& o : out) result.push_back(std::move(*o));
                return result;
            }

            /**
             * @brief Forward pass for layers below buffer_threshold: skips
             * the shared `x` gather and lets each neuron read `x` directly.
             * @param x this layer's input.
             * @return one output value per neuron.
             */
            std::vector<Value<T>> forward_unbuffer(const std::vector<Value<T>>& x) {
                std::vector<std::optional<Value<T>>> out(neurons.size());
            #pragma omp parallel for
                for (std::size_t i = 0; i < neurons.size(); i++) out[i] = neurons[i](x);

                // unwrap back into a plain vector<Value<T>>, single-threaded (no
                // race: the parallel region above already joined, via the
                // implicit barrier at the end of each #pragma omp parallel for)
                std::vector<Value<T>> result;
                result.reserve(out.size());
                for (auto& o : out) result.push_back(std::move(*o));
                return result;
            }

        public:
            /**
             * @brief Constructs a layer of independently-initialized
             * neurons.
             * @param nin number of inputs per neuron.
             * @param nout number of neurons (outputs).
             * @param nonlin if true, every neuron applies ReLU.
             */
            Layer(std::size_t nin, std::size_t nout, bool nonlin = true) :
                forward(nout >= buffer_threshold ? &Layer<T>::forward_buffered : &Layer<T>::forward_unbuffer) {
                neurons.reserve(nout);
                // each neuron must be constructed independently (not via a vector
                // fill-constructor) so every one gets its own random weight draw --
                // filling with nout copies of one prototype Neuron would give every
                // neuron in the layer identical weights.
                for (std::size_t i = 0; i < nout; i++) neurons.emplace_back(nin, nonlin);
            }

            /**
             * @brief Sets the neuron-count threshold above which
             * Layer::operator() gathers `x` once for the whole layer
             * (forward_buffered) rather than letting each neuron read it
             * directly (forward_unbuffer). See ENGINE_NOTES.md.
             * @param threshold the new threshold.
             */
            static void set_buffer_threshold(std::size_t threshold = MICROGRAD_DEFAULT_BUFFER_THRESHOLD) {
                buffer_threshold = threshold;
            }

            /**
             * @brief Runs every neuron in the layer against the same input
             * `x`, in parallel (OpenMP). See ENGINE_NOTES.md for why this
             * parallelizes safely (unlike the backward pass) and for the
             * std::optional<Value<T>> buffer this relies on internally.
             * @param x this layer's input.
             * @return one output value per neuron.
             */
            std::vector<Value<T>> operator()(const std::vector<Value<T>>& x) {
                return (this->*forward)(x);
            }

            /// Resets every neuron's parameters' gradients to zero.
            void zero_grad() {
                for (auto &n : neurons) n.zero_grad();
            }

            /**
             * @brief This layer's trainable parameters: every neuron's
             * weights and bias, concatenated in neuron order.
             * @return the parameters.
             */
            std::vector<Value<T>> parameters() {
                std::vector<Value<T>> out;
                for (auto& n : neurons) {
                    std::vector<Value<T>> p = n.parameters();
                    out.insert(out.end(), p.begin(), p.end());
                }
                return out;
            }

            /**
             * @brief Streams a human-readable description of `L`.
             * @param os the output stream.
             * @param L the layer to print.
             * @return `os`.
             */
            friend std::ostream& operator<<(std::ostream& os, const Layer<T>& L) {
                os << "Layer of [";
                for (std::size_t i = 0; i < L.neurons.size(); i++) {
                    if (i != 0) os << ", ";
                    os << L.neurons[i];
                }
                return os << "]";
            }
        };

        /**
         * @brief A multi-layer perceptron: `nin` input features feeding
         * through a stack of Layers whose sizes are given by `nout` (e.g.
         * `MLP(3, {4, 4, 1})` is a 3-input network with two hidden layers
         * of 4 and a 1-output layer). Every layer is ReLU-activated except
         * the last, which is linear -- matching the original Python
         * micrograd's convention.
         * @tparam T the underlying scalar type.
         */
        template <class T>
        class MLP {
        private:
            std::vector<Layer<T>> layers;

        public:
            /**
             * @brief Constructs a stack of layers.
             * @param nin number of input features.
             * @param nout sizes of each successive layer, e.g. `{4, 4, 1}`.
             */
            MLP(std::size_t nin, std::vector<std::size_t> nout) {
                layers.reserve(nout.size());
                layers.push_back(Layer<T>(nin, nout[0], nout.size() != 1));
                for (std::size_t i = 1; i < nout.size(); i++) layers.push_back(Layer<T>(nout[i-1], nout[i], i != nout.size() - 1));
            }

            /**
             * @brief Forward pass: feeds `x` through each layer in turn,
             * matching `for layer in self.layers: x = layer(x)` in the
             * Python original.
             * @param x the network's input.
             * @return the final layer's output.
             */
            std::vector<Value<T>> operator()(std::vector<Value<T>> x) {
                for (auto& layer : layers) x = layer(x);
                return x;
            }

            /// Resets every layer's parameters' gradients to zero.
            void zero_grad() {
                for (auto& layer : layers) layer.zero_grad();
            }

            /**
             * @brief This network's trainable parameters: every layer's
             * parameters, concatenated in layer order.
             * @return the parameters.
             */
            std::vector<Value<T>> parameters() {
                std::vector<Value<T>> out;
                for (auto& layer : layers) {
                    std::vector<Value<T>> p = layer.parameters();
                    out.insert(out.end(), p.begin(), p.end());
                }
                return out;
            }

            /**
             * @brief Streams a human-readable description of `m`.
             * @param os the output stream.
             * @param m the network to print.
             * @return `os`.
             */
            friend std::ostream& operator<<(std::ostream& os, const MLP<T>& m) {
                os << "MLP of [";
                for (std::size_t i = 0; i < m.layers.size(); i++) {
                    if (i != 0) os << ", ";
                    os << m.layers[i];
                }
                return os << "]";
            }
        };
    };
};
