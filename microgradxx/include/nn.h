// nn.h -- the neural-net layer on top of engine.h's autodiff core: Neuron,
// Layer (a collection of Neurons), and MLP (a stack of Layers), following
// the same shape as the original Python micrograd's nn.py.

#include "engine.h"
#include <vector>
#include <random>
#include <iostream>

namespace micrograd {
	using namespace engine;
	namespace nn {
		// Shared RNG for weight initialization across every Neuron<T>. `inline`
		// (not `static`) is required here since this is a header: `static` at
		// namespace scope gives internal linkage, meaning every .cpp file that
		// includes this header would silently get its own independent
		// generator -- set_seed() in one translation unit would have no effect
		// on any other. `inline` guarantees a single shared instance program-wide.
		inline std::mt19937 generator(0);
		inline void set_seed(unsigned int seed) {
			generator.seed(seed);
		}

		// A single neuron: nin weights, one bias, optionally followed by ReLU.
		template <class T>
		class Neuron {
		private:
			inline static std::uniform_real_distribution<T> uniform{-1.0, 1.0};
			std::vector<Value<T>> w;
			Value<T> b;
			bool nonlin;

		public:
			Neuron(std::size_t nin, bool _nonlin = true) :
				b(static_cast<T>(0)),
				nonlin(_nonlin) {
				w.reserve(nin);
				for (std::size_t i = 0; i < nin; i++) w.emplace_back(uniform(generator));
			}

			// forward pass is a single fused graph node (Value<T>::neuron), instead
			// of nin chained Mul/Add nodes -- see NeuronValueImpl in engine.h
			Value<T> operator()(const std::vector<Value<T>>& x) {
				return Value<T>::neuron(w, x, b, nonlin);
			}

			void zero_grad() {
				for (auto &v : w) v.zero_grad();
				b.zero_grad();
			}

			// returns this neuron's trainable parameters: all weights, then the bias.
			std::vector<Value<T>> parameters() {
				std::vector<Value<T>> out = w;
				out.push_back(b);
				return out;
			}

			friend std::ostream& operator<<(std::ostream& os, const Neuron<T>& n) {
				return os << (n.nonlin ? "ReLU" : "Linear") << "Neuron(" << n.w.size() << ")";
			}
		};

		// A layer: nout independently-initialized Neurons, each taking the same
		// nin-sized input and producing one output -- so calling a Layer maps an
		// nin-vector to an nout-vector.
		template <class T>
		class Layer {
		private:
			std::vector<Neuron<T>> neurons;

		public:
			Layer(std::size_t nin, std::size_t nout, bool nonlin = true) {
				neurons.reserve(nout);
				// each neuron must be constructed independently (not via a vector
				// fill-constructor) so every one gets its own random weight draw --
				// filling with nout copies of one prototype Neuron would give every
				// neuron in the layer identical weights.
				for (std::size_t i = 0; i < nout; i++) neurons.emplace_back(nin, nonlin);
			}

			std::vector<Value<T>> operator()(const std::vector<Value<T>>& x) {
				std::vector<Value<T>> out;
				out.reserve(neurons.size());
				for (auto &n : neurons) out.push_back(n(x));
				return out;
			}

			void zero_grad() {
				for (auto &n : neurons) n.zero_grad();
			}

			std::vector<Value<T>> parameters() {
				std::vector<Value<T>> out;
				for (auto& n : neurons) {
					std::vector<Value<T>> p = n.parameters();
					out.insert(out.end(), p.begin(), p.end());
				}
				return out;
			}

			friend std::ostream& operator<<(std::ostream& os, const Layer<T>& L) {
				os << "Layer of [";
				for (std::size_t i = 0; i < L.neurons.size(); i++) {
					if (i != 0) os << ", ";
					os << L.neurons[i];
				}
				return os << "]";
			}
		};

		// A multi-layer perceptron: nin input features feeding through a stack
		// of Layers whose sizes are given by `nout` (e.g. MLP(3, {4, 4, 1}) is a
		// 3-input network with two hidden layers of 4 and a 1-output layer).
		// Every layer is ReLU-activated except the last, which is linear --
		// matching the original Python micrograd's convention.
		template <class T>
		class MLP {
		private:
			std::vector<Layer<T>> layers;

		public:
			MLP(std::size_t nin, std::vector<std::size_t> nout) {
				layers.reserve(nout.size());
				layers.push_back(Layer<T>(nin, nout[0], nout.size() != 1));
				for (std::size_t i = 1; i < nout.size(); i++) layers.push_back(Layer<T>(nout[i-1], nout[i], i != nout.size() - 1));
			}

			// forward pass: feed x through each layer in turn, matching
			// `for layer in self.layers: x = layer(x)` in the Python original.
			std::vector<Value<T>> operator()(std::vector<Value<T>> x) {
				for (auto& layer : layers) x = layer(x);
				return x;
			}

			void zero_grad() {
				for (auto& layer : layers) layer.zero_grad();
			}

			std::vector<Value<T>> parameters() {
				std::vector<Value<T>> out;
				for (auto& layer : layers) {
					std::vector<Value<T>> p = layer.parameters();
					out.insert(out.end(), p.begin(), p.end());
				}
				return out;
			}

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