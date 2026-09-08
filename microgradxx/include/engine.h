#pragma once
/**
 * @file engine.h
 * @brief A small scalar autodiff engine (reverse-mode automatic
 * differentiation) -- the C++ core of a from-scratch reimplementation of
 * Andrej Karpathy's micrograd (https://github.com/karpathy/micrograd, MIT
 * License).
 *
 * Value<T> is the only type meant to be used directly; everything else in
 * this file is machinery it depends on. See ENGINE_NOTES.md for the design
 * rationale (ownership model, thread-safety, the AVX2 gather technique)
 * behind what's here.
 */

#include <atomic>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <type_traits>
// AVX2 gather intrinsics are x86-only and only usable when this translation
// unit is actually built with -mavx2 (or an -march= that implies it) --
// guard both the include and every use behind __AVX2__ so this header still
// compiles (falling back to a plain scalar loop) on ARM, on x86 without
// -mavx2, and for any T other than float/double.
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace microgradxx {
    namespace engine {
        // forward declarations: every concrete graph-node type Value<T>'s
        // operators need to `new` before those types are actually defined below
        template <class T> class ValueImpl;
        template <class T> class UnaryValueImpl;
        template <class T> class BinaryValueImpl;
        template <class T> class AddValueImpl;
        template <class T> class MulValueImpl;
        template <class T> class PowValueImpl;
        template <class T> class ReLUValueImpl;
        template <class T> class NegValueImpl;
        template <class T> class SubValueImpl;
        template <class T> class DivValueImpl;
        template <class T> class NeuronValueImpl;

        /**
         * @brief A scalar participating in a computation graph.
         *
         * Arithmetic on Value<T> builds up a graph node by node; backward()
         * later walks that graph to populate every node's grad() via
         * reverse-mode automatic differentiation. Cheap to copy/move/store
         * in containers -- see ENGINE_NOTES.md for the ownership model.
         *
         * @tparam T the underlying scalar type (typically float or double).
         */
        template <class T>
        class Value {
        private:
            ValueImpl<T>* impl;

        public:
            /**
             * @brief Copy constructor. Shares the same underlying node
             * (refcounted).
             * @param v the value to copy.
             */
            Value(const Value<T>& v) :
                impl(v.impl) {
                impl->refcnt++;
            }

            /**
             * @brief Move constructor. Steals ownership; `v` becomes
             * moved-from (only safe to destroy or reassign afterward, same
             * convention the STL uses).
             * @param v the value to move from.
             */
            Value(Value<T>&& v) noexcept :
                impl(v.impl) {
                v.impl = nullptr;
            }

            /**
             * @brief Builds a new leaf node (no children) from a raw value
             * -- how inputs/parameters are created, e.g.
             * `Value<double> x(3.0);`.
             * @param data the node's initial forward value.
             */
            explicit Value(const T data) :
                impl(new ValueImpl<T>(data)) {
                impl->refcnt++;
            }

            /**
             * @brief Wraps an already-allocated node (used internally by
             * the operator overloads below).
             * @param implptr the node to wrap; ownership is shared via
             * refcounting, not transferred.
             */
            explicit Value(ValueImpl<T>* implptr) :
                impl(implptr) {
                impl->refcnt++;
            }

            ~Value() {
                if (impl && --(impl->refcnt) == 0) delete impl;
            }

            /**
             * @brief Returns a new Value<T> wrapping a shallow copy of this
             * node (same children; see each op's own copy() for exactly
             * what's duplicated vs. shared).
             * @return the new, independent Value<T>.
             */
            Value<T> copy() {
                return Value<T>(impl->copy());
            }

            /// Like copy(), but replaces this handle's own target in place.
            void clone() {
                ValueImpl<T>* newimpl = impl->copy();
                if (--(impl->refcnt) == 0) delete impl;
                impl = newimpl;
                impl->refcnt++;
            }

            /**
             * @brief This node's current forward value.
             * @return the forward value.
             */
            inline T data() const {
                return impl->data;
            }

            /**
             * @brief This node's accumulated gradient (populated by
             * backward()).
             * @return the accumulated gradient.
             */
            inline T grad() const {
                return impl->grad;
            }

            /// Resets this node's gradient to zero.
            void zero_grad() {
                impl->grad = 0;
            }

            /**
             * @brief Runs reverse-mode automatic differentiation over the
             * graph rooted at this node, populating grad() on every node
             * reached.
             *
             * Call this on a scalar loss; every parameter's grad() is
             * populated afterward. See ENGINE_NOTES.md for how the
             * traversal order is derived.
             */
            void backward() {
                int head = impl->height;
                std::vector<std::vector<ValueImpl<T>*>> bucket(head + 1);
                impl->grad = 1; // seed: d(output)/d(output) = 1
                impl->queued = true;
                bucket[head].push_back(impl);
                for (; head >= 0; head--) {
                    for (auto node : bucket[head]) {
                        node->grad_propagate(bucket);
                        node->queued = false; // reset so the next backward() call works too
                    }
                }
            }

            /**
             * @brief Copy assignment. Shares `V`'s underlying node.
             * @param V the value to copy.
             * @return *this.
             */
            Value<T>& operator=(const Value<T>& V) {
                V.impl->refcnt++;
                if (impl && --(impl->refcnt) == 0) delete impl;
                impl = V.impl;
                return *this;
            }

            /**
             * @brief Move assignment. Steals ownership from `V`, same as
             * the move constructor; `V` becomes moved-from afterward.
             * @param V the value to move from.
             * @return *this.
             */
            Value<T>& operator=(Value<T>&& V) noexcept {
                if (this != &V) {
                    if (impl && --(impl->refcnt) == 0) delete impl;
                    impl = V.impl;
                    V.impl = nullptr;
                }
                return *this;
            }

            /**
             * @brief Assigns from an already-allocated node.
             * @param implptr the node to wrap; ownership is shared via
             * refcounting, not transferred.
             * @return *this.
             */
            Value<T>& operator=(ValueImpl<T>* const& implptr) {
                implptr->refcnt++;
                if (impl && --(impl->refcnt) == 0) delete impl;
                impl = implptr;
                return *this;
            }

            /**
             * @brief Assigns a new leaf node (no children) from a raw value.
             * @param data the node's initial forward value.
             * @return *this.
             */
            Value<T>& operator=(const T& data) {
                if (impl && --(impl->refcnt) == 0) delete impl;
                impl = new ValueImpl<T>(data);
                impl->refcnt++;
                return *this;
            }

            // --- operators below: each just builds the matching graph-node
            // type and wraps it in a Value<T>. See each *ValueImpl class
            // further down for the forward formula and its derivative. ---

            /**
             * @brief Addition. z = L + R.
             * @param L left operand.
             * @param R right operand.
             * @return the resulting graph node.
             */
            friend Value<T> operator+(const Value<T>& L, const Value<T>& R) {
                return Value<T>(new AddValueImpl<T>(L.impl, R.impl));
            }

            /**
             * @brief Multiplication. z = L * R.
             * @param L left operand.
             * @param R right operand.
             * @return the resulting graph node.
             */
            friend Value<T> operator*(const Value<T>& L, const Value<T>& R) {
                return Value<T>(new MulValueImpl<T>(L.impl, R.impl));
            }

            /**
             * @brief Exponentiation. z = L raised to the power R -- both
             * operands are graph nodes, so gradients flow back through the
             * exponent too, not just the base.
             * @param L the base.
             * @param R the exponent.
             * @return the resulting graph node.
             */
            friend Value<T> operator^(const Value<T>& L, const Value<T>& R) {
                return Value<T>(new PowValueImpl<T>(L.impl, R.impl));
            }

            /**
             * @brief Rectified linear unit. z = max(V, 0).
             * @param V the input node.
             * @return the resulting graph node.
             */
            static Value<T> relu(const Value<T>& V) {
                return Value<T>(new ReLUValueImpl<T>(V.impl));
            }

            /**
             * @brief Rectified linear unit applied to this node. z = max(*this, 0).
             * @return the resulting graph node.
             */
            Value<T> relu() {
                return Value<T>(new ReLUValueImpl<T>(impl));
            }

            /**
             * @brief Negation. z = -R.
             * @param R the operand.
             * @return the resulting graph node.
             */
            friend Value<T> operator-(const Value<T>& R) {
                return Value<T>(new NegValueImpl<T>(R.impl));
            }

            /**
             * @brief Subtraction. z = L - R.
             * @param L left operand.
             * @param R right operand.
             * @return the resulting graph node.
             */
            friend Value<T> operator-(const Value<T>& L, const Value<T>& R) {
                return Value<T>(new SubValueImpl<T>(L.impl, R.impl));
            }

            /**
             * @brief Division. z = L / R.
             * @param L left operand (dividend).
             * @param R right operand (divisor).
             * @return the resulting graph node.
             */
            friend Value<T> operator/(const Value<T>& L, const Value<T>& R) {
                return Value<T>(new DivValueImpl<T>(L.impl, R.impl));
            }

            /**
             * @brief Fused w.x + b (+ optional ReLU) as a single graph
             * node, instead of nin separate Mul/Add nodes chained together
             * -- see NeuronValueImpl.
             * @param w the neuron's weights.
             * @param x the neuron's inputs (must be the same length as `w`).
             * @param b the neuron's bias.
             * @param nonlin if true, applies ReLU to the result.
             * @return the resulting graph node.
             */
            static Value<T> neuron(const std::vector<Value<T>>& w, const std::vector<Value<T>>& x, const Value<T>& b, bool nonlin) {
                std::vector<ValueImpl<T>*> wptrs, xptrs;
                wptrs.reserve(w.size());
                for (auto& wi : w) wptrs.push_back(wi.impl);
                xptrs.reserve(x.size());
                for (auto& xi : x) xptrs.push_back(xi.impl);
                return Value<T>(new NeuronValueImpl<T>(wptrs, xptrs, b.impl, nonlin));
            }

            /**
             * @brief Same as the four-argument neuron(), but the caller has
             * already gathered x's data() values into one contiguous
             * buffer -- worth doing when many neurons share the same x
             * (see nn.h's Layer::operator()).
             * @param w the neuron's weights.
             * @param x the neuron's inputs (must be the same length as `w`).
             * @param b the neuron's bias.
             * @param nonlin if true, applies ReLU to the result.
             * @param x_data `x`'s data() values, already gathered into one
             * contiguous buffer of at least `x.size()` elements.
             * @return the resulting graph node.
             */
            static Value<T> neuron(const std::vector<Value<T>>& w, const std::vector<Value<T>>& x, const Value<T>& b, bool nonlin, const T* x_data) {
                std::vector<ValueImpl<T>*> wptrs, xptrs;
                wptrs.reserve(w.size());
                for (auto& wi : w) wptrs.push_back(wi.impl);
                xptrs.reserve(x.size());
                for (auto& xi : x) xptrs.push_back(xi.impl);
                return Value<T>(new NeuronValueImpl<T>(wptrs, xptrs, b.impl, nonlin, x_data));
            }

            /**
             * @brief Streams `v`'s data and grad in human-readable form.
             * @param os the output stream.
             * @param v the value to print.
             * @return `os`.
             */
            friend std::ostream& operator<<(std::ostream& os, const Value<T>& v) {
                return os << "Value(data=" << v.data() << ", grad=" << v.grad() << ")";
            }

            /**
             * @brief Gathers v[i].data() for every element of `v` into the
             * caller-owned, contiguous buffer `out`, using AVX2 where
             * available. See ENGINE_NOTES.md for the gather technique and
             * its benchmarks.
             * @param v the values to gather from.
             * @param out destination buffer; must already be sized to at
             * least `v.size()` elements.
             */
            static void gather_data(const std::vector<Value<T>>& v, T* out) {
                static_assert(sizeof(Value<T>) == sizeof(ValueImpl<T>*),
                    "gather_data's no-buffer fast path assumes Value<T> holds "
                    "nothing but the impl pointer -- if Value<T> grows another "
                    "data member, this needs to fall back to the old "
                    "extract-into-a-vector-of-ValueImpl<T>*-then-gather approach "
                    "instead of silently computing wrong addresses.");
                std::size_t n = v.size();
                std::size_t i = 0;
            #if defined(__AVX2__)
                if (n > 0) {
                    const std::int64_t impl_off = reinterpret_cast<const char*>(&v[0].impl) - reinterpret_cast<const char*>(&v[0]);
                    const std::int64_t data_off = reinterpret_cast<const char*>(&v[0].impl->data) - reinterpret_cast<const char*>(v[0].impl);
                    const char* base = reinterpret_cast<const char*>(v.data()) + impl_off;
                    const __m256i vdata_off = _mm256_set1_epi64x(data_off);
                    if constexpr (std::is_same_v<T, double>) {
                        for (; i + 4 <= n; i += 4) {
                            __m256i ptrs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base + i * sizeof(Value<T>)));
                            __m256i addr = _mm256_add_epi64(ptrs, vdata_off);
                            __m256d vals = _mm256_i64gather_pd(nullptr, addr, 1);
                            _mm256_storeu_pd(out + i, vals);
                        }
                    }
                    else if constexpr (std::is_same_v<T, float>) {
                        for (; i + 4 <= n; i += 4) {
                            __m256i ptrs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base + i * sizeof(Value<T>)));
                            __m256i addr = _mm256_add_epi64(ptrs, vdata_off);
                            __m128 vals = _mm256_i64gather_ps(nullptr, addr, 1);
                            _mm_storeu_ps(out + i, vals);
                        }
                    }
                }
            #endif
                for (; i < n; i++) out[i] = v[i].impl->data;
            }
        };

        /**
         * @brief Base class for every computation-graph node.
         *
         * Holds `data` (the forward value), `grad` (accumulated by
         * backward()), `height` (used to topologically order the backward
         * pass), `queued` (this node's dedup flag for the current
         * backward() call), and `refcnt` (see ENGINE_NOTES.md). A plain
         * ValueImpl<T> (not one of its subclasses) is a leaf: no children,
         * so grad_propagate() is a no-op.
         *
         * @tparam T the underlying scalar type.
         */
        template <class T>
        class ValueImpl {
            friend class Value<T>;
            friend class BinaryValueImpl<T>;
            friend class UnaryValueImpl<T>;
            friend class AddValueImpl<T>;
            friend class MulValueImpl<T>;
            friend class PowValueImpl<T>;
            friend class ReLUValueImpl<T>;
            friend class NegValueImpl<T>;
            friend class SubValueImpl<T>;
            friend class DivValueImpl<T>;
            friend class NeuronValueImpl<T>;

        protected:
            std::atomic<unsigned int> refcnt{0};
            T data;
            T grad;
            const unsigned int height;
            bool queued = false;

            /**
             * @brief Constructs a leaf node.
             * @param data_ the node's initial forward value.
             * @param height_ the node's height (0 for a leaf).
             */
            ValueImpl(T data_, unsigned int height_ = 0) :
                data(data_),
                grad(0),
                height(height_) {}

            /**
             * @brief "Pointer copy constructor": duplicates another node's
             * data/grad/height (not its refcnt, which always starts at 0
             * for a freshly allocated node) -- used by every subclass's own
             * copy-from-pointer constructor, which copy() relies on.
             * @param V the node to copy from.
             */
            ValueImpl(ValueImpl<T>* V) :
                data(V->data),
                grad(V->grad),
                height(V->height) {}

            /**
             * @brief Returns a new heap-allocated shallow copy of this
             * node. Every subclass overrides this to `new` its own
             * concrete type.
             * @return the newly allocated node.
             */
            virtual ValueImpl<T>* copy() {
                return new ValueImpl<T>(this);
            }

            /**
             * @brief Propagates this node's already-accumulated `grad`
             * back to its children (if any), and enqueues each
             * not-yet-visited child so backward()'s main loop visits it.
             * @param bucket the height-bucketed traversal queue.
             */
            virtual void grad_propagate(std::vector<std::vector<ValueImpl<T>*>>& bucket) {
                (void)bucket;
            }

            /**
             * @brief Gathers nodes[i]->data for every element of `nodes`
             * into the caller-owned contiguous buffer `out`, using AVX2
             * where available. The low-level primitive behind
             * Value<T>::gather_data(); see ENGINE_NOTES.md.
             * @param nodes the nodes to gather from.
             * @param out destination buffer; must already be sized to at
             * least `nodes.size()` elements.
             */
            static void gather(const std::vector<ValueImpl<T>*>& nodes, T* out) {
                std::size_t n = nodes.size();
                std::size_t i = 0;
            #if defined(__AVX2__)
                if (!nodes.empty()) {
                    const std::int64_t off = reinterpret_cast<const char*>(&nodes[0]->data) - reinterpret_cast<const char*>(nodes[0]);
                    const __m256i voff = _mm256_set1_epi64x(off);
                    if constexpr (std::is_same_v<T, double>) {
                        for (; i + 4 <= n; i += 4) {
                            __m256i ptrs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(nodes.data() + i));
                            __m256i addr = _mm256_add_epi64(ptrs, voff);
                            __m256d vals = _mm256_i64gather_pd(nullptr, addr, 1);
                            _mm256_storeu_pd(out + i, vals);
                        }
                    }
                    else if constexpr (std::is_same_v<T, float>) {
                        for (; i + 4 <= n; i += 4) {
                            __m256i ptrs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(nodes.data() + i));
                            __m256i addr = _mm256_add_epi64(ptrs, voff);
                            __m128 vals = _mm256_i64gather_ps(nullptr, addr, 1); // 4 floats: pointer-width indices, not 8-wide
                            _mm_storeu_ps(out + i, vals);
                        }
                    }
                }
            #endif
                // scalar remainder (n % 4) -- and the entire loop, for any T
                // other than float/double, or when built without -mavx2.
                for (; i < n; i++) out[i] = nodes[i]->data;
            }

        public:
            /// Virtual so `delete` through a base ValueImpl<T>* runs the
            /// most-derived destructor.
            virtual ~ValueImpl() {}
        };

        /**
         * @brief Shared base for every two-child (binary) operation.
         *
         * Owns childL/childR, handles their refcounting, and provides
         * enqueue() so each concrete op's grad_propagate() only has to
         * state its own derivative.
         *
         * @tparam T the underlying scalar type.
         */
        template <class T>
        class BinaryValueImpl : public ValueImpl<T> {
        protected:
            ValueImpl<T>* childL;
            ValueImpl<T>* childR;

            /**
             * @brief Constructs a binary op node.
             * @param data_ the node's forward value.
             * @param L the left child.
             * @param R the right child.
             */
            BinaryValueImpl(T data_, ValueImpl<T>* L, ValueImpl<T>* R) :
                ValueImpl<T>(data_, std::max(L->height, R->height) + 1),
                childL(L),
                childR(R) {
                (childL->refcnt)++;
                (childR->refcnt)++;
            }

            /**
             * @brief "Pointer copy constructor" -- see ValueImpl<T>'s own.
             * @param V the node to copy from.
             */
            BinaryValueImpl(BinaryValueImpl<T>* V) :
                ValueImpl<T>(V),
                childL(V->childL),
                childR(V->childR) {
                (childL->refcnt)++;
                (childR->refcnt)++;
            }

            virtual ValueImpl<T>* copy() override = 0;

            virtual void grad_propagate(std::vector<std::vector<ValueImpl<T>*>>& bucket) override = 0;

            /**
             * @brief Marks and enqueues each child exactly once.
             * @param bucket the height-bucketed traversal queue.
             */
            void enqueue(std::vector<std::vector<ValueImpl<T>*>>& bucket) {
                if (!childL->queued) {
                    childL->queued = true; bucket[childL->height].push_back(childL);
                }
                if (!childR->queued) {
                    childR->queued = true; bucket[childR->height].push_back(childR);
                }
            }

        public:
            ~BinaryValueImpl() {
                if (--(childL->refcnt) == 0) delete childL;
                if (--(childR->refcnt) == 0) delete childR;
            }
        };

        /**
         * @brief Same idea as BinaryValueImpl, but for single-child (unary)
         * operations.
         * @tparam T the underlying scalar type.
         */
        template <class T>
        class UnaryValueImpl : public ValueImpl<T> {
        protected:
            ValueImpl<T>* childL;

            /**
             * @brief Constructs a unary op node.
             * @param data_ the node's forward value.
             * @param L the child.
             */
            UnaryValueImpl(T data_, ValueImpl<T>* L) :
                ValueImpl<T>(data_, L->height + 1),
                childL(L) {
                (childL->refcnt)++;
            }

            /**
             * @brief "Pointer copy constructor" -- see ValueImpl<T>'s own.
             * @param V the node to copy from.
             */
            UnaryValueImpl(UnaryValueImpl<T>* V) :
                ValueImpl<T>(V),
                childL(V->childL) {
                (childL->refcnt)++;
            }

            virtual ValueImpl<T>* copy() override = 0;

            virtual void grad_propagate(std::vector<std::vector<ValueImpl<T>*>>& bucket) override = 0;

            /**
             * @brief Marks and enqueues the child exactly once.
             * @param bucket the height-bucketed traversal queue.
             */
            void enqueue(std::vector<std::vector<ValueImpl<T>*>>& bucket) {
                if (!childL->queued) {
                    childL->queued = true; bucket[childL->height].push_back(childL);
                }
            }

        public:
            ~UnaryValueImpl() {
                if (--(childL->refcnt) == 0) delete childL;
            }
        };

        /**
         * @brief Addition node. z = L + R.  dz/dL = 1, dz/dR = 1.
         * @tparam T the underlying scalar type.
         */
        template <class T>
        class AddValueImpl : public BinaryValueImpl<T> {
        public:
            /**
             * @brief Constructs an addition node.
             * @param L the left operand.
             * @param R the right operand.
             */
            AddValueImpl(ValueImpl<T>* L, ValueImpl<T>* R) :
                BinaryValueImpl<T>(L->data + R->data, L, R) {}

            /// @param V the node to copy from.
            AddValueImpl(AddValueImpl<T>* V) :
                BinaryValueImpl<T>(V) {}

        protected:
            virtual ValueImpl<T>* copy() override {
                return new AddValueImpl<T>(this);
            }

            virtual void grad_propagate(std::vector<std::vector<ValueImpl<T>*>>& bucket) override {
                this->childL->grad += this->grad;
                this->childR->grad += this->grad;
                this->enqueue(bucket);
            }
        };

        /**
         * @brief Multiplication node. z = L * R.  dz/dL = R, dz/dR = L.
         * @tparam T the underlying scalar type.
         */
        template <class T>
        class MulValueImpl : public BinaryValueImpl<T> {
        public:
            /**
             * @brief Constructs a multiplication node.
             * @param L the left operand.
             * @param R the right operand.
             */
            MulValueImpl(ValueImpl<T>* L, ValueImpl<T>* R) :
                BinaryValueImpl<T>(L->data * R->data, L, R) {}

            /// @param V the node to copy from.
            MulValueImpl(MulValueImpl<T>* V) :
                BinaryValueImpl<T>(V) {}

        protected:
            virtual ValueImpl<T>* copy() override {
                return new MulValueImpl<T>(this);
            }

            virtual void grad_propagate(std::vector<std::vector<ValueImpl<T>*>>& bucket) override {
                this->childL->grad += this->childR->data * this->grad;
                this->childR->grad += this->childL->data * this->grad;
                this->enqueue(bucket);
            }
        };

        /**
         * @brief Exponentiation node. z = base ^ exponent, both
         * differentiable graph nodes (not a fixed scalar exponent).
         * @tparam T the underlying scalar type.
         */
        template <class T>
        class PowValueImpl : public BinaryValueImpl<T> {
        public:
            /**
             * @brief Constructs a power node.
             * @param L the base.
             * @param R the exponent.
             */
            PowValueImpl(ValueImpl<T>* L, ValueImpl<T>* R) :
                BinaryValueImpl<T>(std::pow(L->data, R->data), L, R) {}

            /// @param V the node to copy from.
            PowValueImpl(PowValueImpl<T>* V) :
                BinaryValueImpl<T>(V) {}

        protected:
            virtual ValueImpl<T>* copy() override {
                return new PowValueImpl<T>(this);
            }

            virtual void grad_propagate(std::vector<std::vector<ValueImpl<T>*>>& bucket) override {
                // z = childL ^ childR
                // dz/dchildL = childR * childL^(childR-1)
                // dz/dchildR = z * ln(childL)      -- only defined for childL > 0
                this->childL->grad += this->childR->data * std::pow(this->childL->data, this->childR->data - 1) * this->grad;
                this->childR->grad += this->data * std::log(this->childL->data) * this->grad;
                this->enqueue(bucket);
            }
        };

        /**
         * @brief ReLU node. z = max(L, 0).  dz/dL = 1 if the
         * (post-activation) output is positive, else 0 -- checking
         * `data > 0` is equivalent to checking the pre-ReLU input's sign
         * here, since relu(x) > 0 iff x > 0.
         * @tparam T the underlying scalar type.
         */
        template <class T>
        class ReLUValueImpl : public UnaryValueImpl<T> {
        public:
            /**
             * @brief Constructs a ReLU node.
             * @param L the operand.
             */
            ReLUValueImpl(ValueImpl<T>* L) :
                UnaryValueImpl<T>(L->data > 0 ? L->data : 0, L) {}

            /// @param V the node to copy from.
            ReLUValueImpl(ReLUValueImpl<T>* V) :
                UnaryValueImpl<T>(V) {}

        protected:
            virtual ValueImpl<T>* copy() override {
                return new ReLUValueImpl<T>(this);
            }

            virtual void grad_propagate(std::vector<std::vector<ValueImpl<T>*>>& bucket) override {
                this->childL->grad += this->data > 0 ? this->grad : 0;
                this->enqueue(bucket);
            }
        };

        /**
         * @brief Negation node. z = -L.  dz/dL = -1.
         * @tparam T the underlying scalar type.
         */
        template <class T>
        class NegValueImpl : public UnaryValueImpl<T> {
        public:
            /**
             * @brief Constructs a negation node.
             * @param L the operand.
             */
            NegValueImpl(ValueImpl<T>* L) :
                UnaryValueImpl<T>(-L->data, L) {}

            /// @param V the node to copy from.
            NegValueImpl(NegValueImpl<T>* V) :
                UnaryValueImpl<T>(V) {}

        protected:
            virtual ValueImpl<T>* copy() override {
                return new NegValueImpl<T>(this);
            }

            virtual void grad_propagate(std::vector<std::vector<ValueImpl<T>*>>& bucket) override {
                this->childL->grad -= this->grad;
                this->enqueue(bucket);
            }
        };

        /**
         * @brief Subtraction node. z = L - R.  dz/dL = 1, dz/dR = -1.
         * @tparam T the underlying scalar type.
         */
        template <class T>
        class SubValueImpl : public BinaryValueImpl<T> {
        public:
            /**
             * @brief Constructs a subtraction node.
             * @param L the left operand.
             * @param R the right operand.
             */
            SubValueImpl(ValueImpl<T>* L, ValueImpl<T>* R) :
                BinaryValueImpl<T>(L->data - R->data, L, R) {}

            /// @param V the node to copy from.
            SubValueImpl(SubValueImpl<T>* V) :
                BinaryValueImpl<T>(V) {}

        protected:
            virtual ValueImpl<T>* copy() override {
                return new SubValueImpl<T>(this);
            }

            virtual void grad_propagate(std::vector<std::vector<ValueImpl<T>*>>& bucket) override {
                this->childL->grad += this->grad;
                this->childR->grad -= this->grad;
                this->enqueue(bucket);
            }
        };

        /**
         * @brief Division node. z = L / R.  dz/dL = 1/R, dz/dR = -L/R^2.
         * @tparam T the underlying scalar type.
         */
        template <class T>
        class DivValueImpl : public BinaryValueImpl<T> {
        public:
            /**
             * @brief Constructs a division node.
             * @param L the left operand (dividend).
             * @param R the right operand (divisor).
             */
            DivValueImpl(ValueImpl<T>* L, ValueImpl<T>* R) :
                BinaryValueImpl<T>(L->data / R->data, L, R) {}

            /// @param V the node to copy from.
            DivValueImpl(DivValueImpl<T>* V) :
                BinaryValueImpl<T>(V) {}

        protected:
            virtual ValueImpl<T>* copy() override {
                return new DivValueImpl<T>(this);
            }

            virtual void grad_propagate(std::vector<std::vector<ValueImpl<T>*>>& bucket) override {
                // z = childL / childR
                // dz/dchildL = 1 / childR      dz/dchildR = -childL / childR^2
                this->childL->grad += this->grad / this->childR->data;
                this->childR->grad -= (this->childL->data / (this->childR->data * this->childR->data)) * this->grad;
                this->enqueue(bucket);
            }
        };

        /**
         * @brief Fuses an entire neuron's forward computation -- w . x + b,
         * with an optional ReLU -- into a single graph node, instead of
         * 2*nin separate Mul/Add nodes (plus one ReLU node) chained
         * together.
         *
         * Doesn't derive from Binary/UnaryValueImpl since it has a
         * variable number of children (w and x), so its refcounting and
         * grad_propagate() are handled directly here.
         *
         * @tparam T the underlying scalar type.
         */
        template <class T>
        class NeuronValueImpl : public ValueImpl<T> {
        protected:
            std::vector<ValueImpl<T>*> w;
            std::vector<ValueImpl<T>*> x;
            ValueImpl<T>* b;
            bool nonlin;

            /**
             * @brief Forward formula when w and x are both scattered on
             * the heap (used for layers below nn.h's buffer_threshold).
             * See ENGINE_NOTES.md for the AVX2 gather technique and its
             * measured cost at different fan-ins.
             * @param w the neuron's weights.
             * @param x the neuron's inputs.
             * @param b the neuron's bias.
             * @param nonlin if true, applies ReLU to the result.
             * @return the neuron's forward value.
             */
            static T computeData(const std::vector<ValueImpl<T>*>& w, const std::vector<ValueImpl<T>*>& x, ValueImpl<T>* b, bool nonlin) {
                std::size_t n = w.size();
                std::size_t i = 0;
                T sum = 0;
            #if defined(__AVX2__)
                if (n > 0) {
                    const std::int64_t woff = reinterpret_cast<const char*>(&w[0]->data) - reinterpret_cast<const char*>(w[0]);
                    const std::int64_t xoff = reinterpret_cast<const char*>(&x[0]->data) - reinterpret_cast<const char*>(x[0]);
                    const __m256i vwoff = _mm256_set1_epi64x(woff);
                    const __m256i vxoff = _mm256_set1_epi64x(xoff);
                    if constexpr (std::is_same_v<T, double>) {
                        __m256d acc = _mm256_setzero_pd();
                        for (; i + 4 <= n; i += 4) {
                            __m256i wptrs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w.data() + i));
                            __m256i waddr = _mm256_add_epi64(wptrs, vwoff);
                            __m256d wval = _mm256_i64gather_pd(nullptr, waddr, 1);
                            __m256i xptrs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.data() + i));
                            __m256i xaddr = _mm256_add_epi64(xptrs, vxoff);
                            __m256d xval = _mm256_i64gather_pd(nullptr, xaddr, 1);
                            acc = _mm256_fmadd_pd(wval, xval, acc);
                        }
                        alignas(32) double tmp[4];
                        _mm256_store_pd(tmp, acc);
                        sum = static_cast<T>((tmp[0] + tmp[1]) + (tmp[2] + tmp[3]));
                    }
                    else if constexpr (std::is_same_v<T, float>) {
                        __m128 acc = _mm_setzero_ps();
                        for (; i + 4 <= n; i += 4) {
                            __m256i wptrs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w.data() + i));
                            __m256i waddr = _mm256_add_epi64(wptrs, vwoff);
                            __m128 wval = _mm256_i64gather_ps(nullptr, waddr, 1);
                            __m256i xptrs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.data() + i));
                            __m256i xaddr = _mm256_add_epi64(xptrs, vxoff);
                            __m128 xval = _mm256_i64gather_ps(nullptr, xaddr, 1);
                            acc = _mm_fmadd_ps(wval, xval, acc);
                        }
                        alignas(16) float tmp[4];
                        _mm_store_ps(tmp, acc);
                        sum = static_cast<T>((tmp[0] + tmp[1]) + (tmp[2] + tmp[3]));
                    }
                }
            #endif
                for (; i < n; i++) sum += w[i]->data * x[i]->data;
                sum += b->data;
                return nonlin ? (sum > 0 ? sum : 0) : sum;
            }

            /**
             * @brief Same forward formula, but x's data() values are
             * already gathered into one contiguous buffer by the caller --
             * see ENGINE_NOTES.md for why that's worth doing when a whole
             * layer's neurons share the same x.
             * @param w the neuron's weights.
             * @param x the neuron's inputs (unused directly; kept for a
             * consistent signature with the four-argument overload).
             * @param b the neuron's bias.
             * @param nonlin if true, applies ReLU to the result.
             * @param x_data `x`'s data() values, already gathered into one
             * contiguous buffer of at least `w.size()` elements.
             * @return the neuron's forward value.
             */
            static T computeData(const std::vector<ValueImpl<T>*>& w, const std::vector<ValueImpl<T>*>& x, ValueImpl<T>* b, bool nonlin, const T* x_data) {
                (void)x;
                std::size_t n = w.size();
                std::size_t i = 0;
                T sum = 0;
            #if defined(__AVX2__)
                if (n > 0) {
                    const std::int64_t off = reinterpret_cast<const char*>(&w[0]->data) - reinterpret_cast<const char*>(w[0]);
                    const __m256i voff = _mm256_set1_epi64x(off);
                    if constexpr (std::is_same_v<T, double>) {
                        __m256d acc = _mm256_setzero_pd();
                        for (; i + 4 <= n; i += 4) {
                            __m256i wptrs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w.data() + i));
                            __m256i waddr = _mm256_add_epi64(wptrs, voff);
                            __m256d wval = _mm256_i64gather_pd(nullptr, waddr, 1);
                            __m256d xval = _mm256_loadu_pd(x_data + i); // contiguous -- plain load, no gather
                            acc = _mm256_fmadd_pd(wval, xval, acc);
                        }
                        alignas(32) double tmp[4];
                        _mm256_store_pd(tmp, acc);
                        sum = static_cast<T>((tmp[0] + tmp[1]) + (tmp[2] + tmp[3]));
                    }
                    else if constexpr (std::is_same_v<T, float>) {
                        __m128 acc = _mm_setzero_ps();
                        for (; i + 4 <= n; i += 4) {
                            __m256i wptrs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w.data() + i));
                            __m256i waddr = _mm256_add_epi64(wptrs, voff);
                            __m128 wval = _mm256_i64gather_ps(nullptr, waddr, 1);
                            __m128 xval = _mm_loadu_ps(x_data + i); // contiguous -- plain load, no gather
                            acc = _mm_fmadd_ps(wval, xval, acc);
                        }
                        alignas(16) float tmp[4];
                        _mm_store_ps(tmp, acc);
                        sum = static_cast<T>((tmp[0] + tmp[1]) + (tmp[2] + tmp[3]));
                    }
                }
            #endif
                for (; i < n; i++) sum += w[i]->data * x_data[i];
                sum += b->data;
                return nonlin ? (sum > 0 ? sum : 0) : sum;
            }

            /**
             * @brief Computes this neuron's height (1 + the tallest child).
             * @param w the neuron's weights.
             * @param x the neuron's inputs.
             * @param b the neuron's bias.
             * @return the neuron's height.
             */
            static unsigned int computeHeight(const std::vector<ValueImpl<T>*>& w, const std::vector<ValueImpl<T>*>& x, ValueImpl<T>* b) {
                unsigned int h = b->height;
                for (std::size_t i = 0; i < w.size(); i++) h = std::max(h, std::max(w[i]->height, x[i]->height));
                return h + 1;
            }

        public:
            /**
             * @brief Constructs a neuron node, gathering `x` scattered on
             * the heap.
             * @param w_ the neuron's weights.
             * @param x_ the neuron's inputs (must be the same length as `w_`).
             * @param b_ the neuron's bias.
             * @param nonlin_ if true, applies ReLU to the result.
             */
            NeuronValueImpl(const std::vector<ValueImpl<T>*>& w_, const std::vector<ValueImpl<T>*>& x_, ValueImpl<T>* b_, bool nonlin_) :
                ValueImpl<T>(computeData(w_, x_, b_, nonlin_), computeHeight(w_, x_, b_)),
                w(w_), x(x_), b(b_), nonlin(nonlin_) {
                for (auto* wi : w) wi->refcnt++;
                for (auto* xi : x) xi->refcnt++;
                b->refcnt++;
            }

            /**
             * @brief Constructs a neuron node, with `x`'s data() values
             * already gathered into `x_data` by the caller.
             * @param w_ the neuron's weights.
             * @param x_ the neuron's inputs (must be the same length as `w_`).
             * @param b_ the neuron's bias.
             * @param nonlin_ if true, applies ReLU to the result.
             * @param x_data `x_`'s data() values, already gathered into one
             * contiguous buffer.
             */
            NeuronValueImpl(const std::vector<ValueImpl<T>*>& w_, const std::vector<ValueImpl<T>*>& x_, ValueImpl<T>* b_, bool nonlin_, const T* x_data) :
                ValueImpl<T>(computeData(w_, x_, b_, nonlin_, x_data), computeHeight(w_, x_, b_)),
                w(w_), x(x_), b(b_), nonlin(nonlin_) {
                for (auto* wi : w) wi->refcnt++;
                for (auto* xi : x) xi->refcnt++;
                b->refcnt++;
            }

            /**
             * @brief "Pointer copy constructor" -- see ValueImpl<T>'s own.
             * @param V the node to copy from.
             */
            NeuronValueImpl(NeuronValueImpl<T>* V) :
                ValueImpl<T>(V),
                w(V->w), x(V->x), b(V->b), nonlin(V->nonlin) {
                for (auto* wi : w) wi->refcnt++;
                for (auto* xi : x) xi->refcnt++;
                b->refcnt++;
            }

            ~NeuronValueImpl() {
                for (auto* wi : w) if (--(wi->refcnt) == 0) delete wi;
                for (auto* xi : x) if (--(xi->refcnt) == 0) delete xi;
                if (--(b->refcnt) == 0) delete b;
            }

        protected:
            virtual ValueImpl<T>* copy() override {
                return new NeuronValueImpl<T>(this);
            }

            virtual void grad_propagate(std::vector<std::vector<ValueImpl<T>*>>& bucket) override {
                // sum = b + Σ w[i]*x[i]; data = nonlin ? relu(sum) : sum
                // d(data)/d(sum) = (nonlin && sum <= 0) ? 0 : 1  -- same test as ReLUValueImpl
                T dsum = (nonlin && !(this->data > 0)) ? 0 : this->grad;
                for (std::size_t i = 0; i < w.size(); i++) {
                    w[i]->grad += x[i]->data * dsum;
                    x[i]->grad += w[i]->data * dsum;
                    if (!w[i]->queued) {
                        w[i]->queued = true; bucket[w[i]->height].push_back(w[i]);
                    }
                    if (!x[i]->queued) {
                        x[i]->queued = true; bucket[x[i]->height].push_back(x[i]);
                    }
                }
                b->grad += dsum;
                if (!b->queued) {
                    b->queued = true; bucket[b->height].push_back(b);
                }
            }
        };
    };
};
