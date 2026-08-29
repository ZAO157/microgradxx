// engine.h -- a small scalar autodiff engine (reverse-mode automatic
// differentiation), the C++ core of a from-scratch reimplementation of
// Andrej Karpathy's micrograd (https://github.com/karpathy/micrograd, MIT
// License). The public-facing type is Value<T>; everything else in this
// file is internal machinery it depends on.
//
// ---------------------------------------------------------------------
// Ownership model, in one paragraph:
//
// Value<T> is a thin, reference-counted HANDLE (like a small hand-rolled
// shared_ptr) -- it is cheap to copy, cheap to move, and safe to store in
// containers, reassign, let go out of scope, etc. The actual computation
// graph node it points to is a separate heap-allocated ValueImpl<T> (or one
// of its subclasses below), which carries its own `refcnt`. A node is only
// ever deleted once its refcnt drops to zero, i.e. once nothing --
// neither a Value<T> handle nor another node that uses it as a child --
// still needs it. This decoupling is deliberate and load-bearing: it lets
// a graph node outlive whatever local variable or temporary container
// originally produced it, which is exactly what's needed for backward()
// to walk the *entire* graph after the forward pass's own local storage
// (e.g. one layer's activations) has already gone out of scope.
//
// Every Value<T> is guaranteed to hold a non-null impl, EXCEPT a
// moved-from Value<T> (impl == nullptr) -- the same convention the
// standard library uses for moved-from objects: such a Value may only be
// destroyed or assigned to afterward, never read via data()/grad()/etc.
// There is deliberately no default constructor: a Value<T> you can hold
// is always either freshly built from a T/ValueImpl<T>*, or a copy/move of
// one that was. (A container that needs to default-construct a slot before
// it has a real Value<T> to put there -- e.g. writing into a pre-sized
// vector from multiple threads -- should hold std::optional<Value<T>>
// instead; see Layer::operator() in nn.h.)
//
// ---------------------------------------------------------------------
// Thread-safety note:
//
// `refcnt` is a std::atomic<unsigned int>, not a plain unsigned int. This
// engine allows two or more Value<T>/ValueImpl<T> node graphs to share a
// child (the common case being several Neuron<T>s in one Layer all taking
// the same input vector x): building those nodes concurrently -- as
// nn.h's parallel Layer::operator() does -- means multiple threads
// increment the *same* child's refcnt at the same time. A plain
// unsigned int refcnt++ is a read-modify-write with no atomicity
// guarantee at all; two threads can both read the same pre-increment
// value and both write back count+1, silently losing one increment. That
// under-count can (and, empirically, does) let a node's refcnt reach zero
// while something still points to it, triggering a premature delete and
// a use-after-free the next time that "something" is touched. Every
// refcnt read-modify-write in this file (++, --, and the compound
// decrement-and-test used before every delete) goes through
// std::atomic's own ++/--/compound operators, which are individually
// atomic, so this race is closed without changing any of the calling
// code's syntax. See the class-level encapsulation note below for why
// this member (and the rest of a node's internals) isn't reachable from
// outside Value<T> in the first place.
// ---------------------------------------------------------------------

#include <atomic>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

namespace micrograd {
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

		// The user-facing scalar type. Wraps a pointer to a ValueImpl<T> graph
		// node and manages its reference count; arithmetic on Value<T> builds up
		// a computation graph that backward() can later walk to populate every
		// node's .grad() via reverse-mode autodiff.
		template <class T>
		class Value {
		private:
			ValueImpl<T>* impl;

		public:
			Value(const Value<T>& v) :
				impl(v.impl) {
				impl->refcnt++;
			}

			// steals ownership -- no refcnt touched at all, that's the whole point.
			// leaves v in a "moved-from" state: v.impl is null, so v may only be
			// destroyed or assigned to afterward, same convention the STL uses.
			Value(Value<T>&& v) noexcept :
				impl(v.impl) {
				v.impl = nullptr;
			}

			// builds a new leaf node (no children) directly from a raw value --
			// e.g. Value<double> x(3.0); this is how inputs/parameters are created.
			// explicit: nothing should silently convert a T into a graph node.
			explicit Value(const T data) :
				impl(new ValueImpl<T>(data)) {
				impl->refcnt++;
			}

			// wraps an already-allocated node (freshly `new`'d by one of the
			// operator overloads below) and takes a reference to it.
			// explicit: nothing should silently convert a raw ValueImpl<T>* into
			// a Value<T> handle.
			explicit Value(ValueImpl<T>* implptr) :
				impl(implptr) {
				impl->refcnt++;
			}

			~Value() {
				if (impl && --(impl->refcnt) == 0) delete impl;
			}

			// returns a new Value<T> wrapping a shallow copy of this node (same
			// children, same data/grad) -- see ValueImpl<T>::copy()/each subclass's
			// override for exactly what gets duplicated vs. shared.
			Value<T> copy() {
				return Value<T>(impl->copy());
			}

			// like copy(), but replaces *this* handle's target in place instead of
			// returning a new Value<T>.
			void clone() {
				ValueImpl<T>* newimpl = impl->copy();
				if (--(impl->refcnt) == 0) delete impl;
				impl = newimpl;
				impl->refcnt++;
			}

			inline T data() const {
				return impl->data;
			}

			inline T grad() const {
				return impl->grad;
			}

			void zero_grad() {
				impl->grad = 0;
			}

			// Reverse-mode autodiff over the graph rooted at this node.
			//
			// Every node's `height` is defined at construction time as
			// 1 + max(height of its children) -- so a node's height is always
			// strictly greater than any of its children's. That means bucketing
			// nodes by height and processing the buckets from tallest to
			// shortest is a valid topological order: by the time a node's bucket
			// is reached, every node that could still add to its .grad() (i.e.
			// every parent, which necessarily sits at a taller height) has
			// already run. `queued` prevents a node reachable through more than
			// one path (e.g. a value reused twice) from being enqueued -- and
			// therefore having its own grad_propagate() run -- more than once per
			// backward() call, which would otherwise double-count its
			// contribution to its children's gradients.
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

			Value<T>& operator=(const Value<T>& V) {
				V.impl->refcnt++;
				if (impl && --(impl->refcnt) == 0) delete impl;
				impl = V.impl;
				return *this;
			}

			// steals ownership from V, same as the move constructor; V becomes
			// moved-from (impl == nullptr) afterward.
			Value<T>& operator=(Value<T>&& V) noexcept {
				if (this != &V) {
					if (impl && --(impl->refcnt) == 0) delete impl;
					impl = V.impl;
					V.impl = nullptr;
				}
				return *this;
			}

			Value<T>& operator=(ValueImpl<T>* const& implptr) {
				implptr->refcnt++;
				if (impl && --(impl->refcnt) == 0) delete impl;
				impl = implptr;
				return *this;
			}

			Value<T>& operator=(const T& data) {
				if (impl && --(impl->refcnt) == 0) delete impl;
				impl = new ValueImpl<T>(data);
				impl->refcnt++;
				return *this;
			}

			// --- operators below: each just builds the matching graph-node type
			// and wraps it in a Value<T>. See each *ValueImpl class further down
			// for the forward formula and its backward (derivative) rule. ---

			friend Value<T> operator+(const Value<T>& L, const Value<T>& R) {
				return Value<T>(new AddValueImpl<T>(L.impl, R.impl));
			}

			friend Value<T> operator*(const Value<T>& L, const Value<T>& R) {
				return Value<T>(new MulValueImpl<T>(L.impl, R.impl));
			}

			// L raised to the power R -- both operands are graph nodes, so
			// gradients flow back through the exponent too, not just the base.
			friend Value<T> operator^(const Value<T>& L, const Value<T>& R) {
				return Value<T>(new PowValueImpl<T>(L.impl, R.impl));
			}

			static Value<T> relu(const Value<T>& V) {
				return Value<T>(new ReLUValueImpl<T>(V.impl));
			}

			Value<T> relu() {
				return Value<T>(new ReLUValueImpl<T>(impl));
			}

			friend Value<T> operator-(const Value<T>& R) {
				return Value<T>(new NegValueImpl<T>(R.impl));
			}

			friend Value<T> operator-(const Value<T>& L, const Value<T>& R) {
				return Value<T>(new SubValueImpl<T>(L.impl, R.impl));
			}

			friend Value<T> operator/(const Value<T>& L, const Value<T>& R) {
				return Value<T>(new DivValueImpl<T>(L.impl, R.impl));
			}

			// fused w.x + b (+ optional ReLU) as a single graph node, instead of
			// nin separate Mul/Add nodes chained together -- see NeuronValueImpl
			// for why this matters (far fewer heap allocations per neuron).
			static Value<T> neuron(const std::vector<Value<T>>& w, const std::vector<Value<T>>& x, const Value<T>& b, bool nonlin) {
				std::vector<ValueImpl<T>*> wptrs, xptrs;
				wptrs.reserve(w.size());
				for (auto& wi : w) wptrs.push_back(wi.impl);
				xptrs.reserve(x.size());
				for (auto& xi : x) xptrs.push_back(xi.impl);
				return Value<T>(new NeuronValueImpl<T>(wptrs, xptrs, b.impl, nonlin));
			}

			// same as above, but the caller has already gathered x's data() values
			// into one contiguous buffer (x_data) -- see NeuronValueImpl::computeData
			// below for why that's worth doing when many neurons share the same x,
			// as nn.h's parallel Layer::operator() does.
			static Value<T> neuron(const std::vector<Value<T>>& w, const std::vector<Value<T>>& x, const Value<T>& b, bool nonlin, const T* x_data) {
				std::vector<ValueImpl<T>*> wptrs, xptrs;
				wptrs.reserve(w.size());
				for (auto& wi : w) wptrs.push_back(wi.impl);
				xptrs.reserve(x.size());
				for (auto& xi : x) xptrs.push_back(xi.impl);
				return Value<T>(new NeuronValueImpl<T>(wptrs, xptrs, b.impl, nonlin, x_data));
			}

			friend std::ostream& operator<<(std::ostream& os, const Value<T>& v) {
				return os << "Value(data=" << v.data() << ", grad=" << v.grad() << ")";
			}
		};

		// Base class for every computation-graph node. Holds the forward value
		// (`data`), the accumulated gradient (`grad`, populated by backward()),
		// `height` (used to topologically order the backward pass -- see
		// Value<T>::backward() above), `queued` (this node's dedup flag for the
		// current backward() pass), and `refcnt` (see the thread-safety note at
		// the top of the file).
		//
		// Encapsulation: everything here is `protected`, and Value<T> is the
		// only class ever granted `friend` access to it. A node's internals
		// (its refcnt in particular) are only ever safe to touch through the
		// bookkeeping Value<T> already does in its constructors/destructor/
		// assignment operators -- poking `data`/`grad`/`refcnt` directly from
		// outside that bookkeeping is exactly how you'd corrupt the graph or
		// leak/double-free a node. The concrete op subclasses' *constructors*
		// stay public further down, since Value<T>'s own operator+/operator*/
		// etc. (friend *functions*, not friend of every ValueImpl subclass)
		// need to `new` them directly -- but by the time you can call one of
		// those constructors you can only have gotten its ValueImpl<T>*
		// arguments from an existing Value<T>::impl in the first place, so
		// nothing outside this file can ever produce a "bare" node that isn't
		// already owned by some Value<T>.
		//
		// A plain ValueImpl<T> (as opposed to one of its subclasses) is a leaf:
		// no children, so grad_propagate() is a no-op -- gradients simply accumulate
		// on it and go nowhere further.
		template <class T>
		class ValueImpl {
			// The complete, closed list of code allowed to touch a node's
			// internals: the public handle (Value<T>) and every node type in
			// this file. The node types all need this -- not just to access
			// *their own* inherited data/grad/height, which ordinary protected
			// inheritance already allows, but to reach into a *sibling* node's
			// data/grad/height/refcnt through a plain ValueImpl<T>* (e.g.
			// AddValueImpl's constructor reading L->data/R->data, or any
			// grad_propagate() writing childL->grad) -- C++ does not treat
			// "protected, inherited from the same base" as accessible through
			// an arbitrary same-base pointer to a *different* subclass, only
			// through a pointer of your own (or friended) type, so plain
			// `protected` alone isn't enough to let AddValueImpl read a field
			// on whatever concrete node type L actually points to. Nothing
			// outside this file appears on this list.
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

			ValueImpl(T data_, unsigned int height_ = 0) :
				data(data_),
				grad(0),
				height(height_) {}

			// "pointer copy constructor": duplicates another node's data/grad/
			// height (but NOT its refcnt, which always starts at 0 for a freshly
			// allocated node) -- used by every subclass's own copy-from-pointer
			// constructor, which is in turn what copy() below relies on.
			ValueImpl(ValueImpl<T>* V) :
				data(V->data),
				grad(V->grad),
				height(V->height) {}

			// returns a new heap-allocated node that is a shallow copy of this
			// one (same children, refcounted accordingly). Every subclass
			// overrides this to `new` its own concrete type -- there's no way to
			// avoid one such override per leaf type, since only a type's own
			// constructor can produce an object of that exact type ("virtual
			// constructor" idiom).
			virtual ValueImpl<T>* copy() {
				return new ValueImpl<T>(this);
			}

			// Propagates this node's already-accumulated `grad` back to its
			// children (if any), and enqueues each not-yet-visited child into
			// `bucket` so Value<T>::backward()'s main loop will visit it. A leaf
			// node has nothing to propagate to, hence the empty default body.
			virtual void grad_propagate(std::vector<std::vector<ValueImpl<T>*>>&) {}

		public:
			// virtual so that `delete` through a base ValueImpl<T>* (as happens
			// throughout this file, e.g. in ~BinaryValueImpl) correctly runs the
			// most-derived destructor. Public: deleting a node is a normal part
			// of every subclass's own destructor tearing down its children.
			virtual ~ValueImpl() {}
		};

		// Shared base for every two-child (binary) operation: owns childL/childR,
		// handles their refcounting, and provides `enqueue()` so each concrete
		// op's grad_propagate() only has to state its own derivative, not repeat the
		// "mark child visited and push it into the right height bucket"
		// bookkeeping. Abstract -- copy()/grad_propagate() are left pure virtual since
		// only a concrete op knows its own forward formula and derivative.
		template <class T>
		class BinaryValueImpl : public ValueImpl<T> {
		protected:
			ValueImpl<T>* childL;
			ValueImpl<T>* childR;

			BinaryValueImpl(T data_, ValueImpl<T>* L, ValueImpl<T>* R) :
				ValueImpl<T>(data_, std::max(L->height, R->height) + 1),
				childL(L),
				childR(R) {
				(childL->refcnt)++;
				(childR->refcnt)++;
			}

			BinaryValueImpl(BinaryValueImpl<T>* V) :
				ValueImpl<T>(V),
				childL(V->childL),
				childR(V->childR) {
				(childL->refcnt)++;
				(childR->refcnt)++;
			}

			virtual ValueImpl<T>* copy() override = 0;

			virtual void grad_propagate(std::vector<std::vector<ValueImpl<T>*>>& bucket) override = 0;

			// shared by every binary op's grad_propagate(): mark+enqueue a child exactly once
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

		// Same idea as BinaryValueImpl, but for single-child (unary) operations.
		template <class T>
		class UnaryValueImpl : public ValueImpl<T> {
		protected:
			ValueImpl<T>* childL;

			UnaryValueImpl(T data_, ValueImpl<T>* L) :
				ValueImpl<T>(data_, L->height + 1),
				childL(L) {
				(childL->refcnt)++;
			}

			UnaryValueImpl(UnaryValueImpl<T>* V) :
				ValueImpl<T>(V),
				childL(V->childL) {
				(childL->refcnt)++;
			}

			virtual ValueImpl<T>* copy() override = 0;

			virtual void grad_propagate(std::vector<std::vector<ValueImpl<T>*>>& bucket) override = 0;

			// shared by every unary op's grad_propagate(): mark+enqueue the child exactly once
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

		// z = L + R.  dz/dL = 1, dz/dR = 1.
		template <class T>
		class AddValueImpl : public BinaryValueImpl<T> {
		public:
			// public: this is the sanctioned way to build one of these nodes, and
			// it's what Value<T>::operator+ (a friend *function* of Value<T>, not
			// of AddValueImpl) calls directly.
			AddValueImpl(ValueImpl<T>* L, ValueImpl<T>* R) :
				BinaryValueImpl<T>(L->data + R->data, L, R) {}

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

		// z = L * R.  dz/dL = R, dz/dR = L.
		template <class T>
		class MulValueImpl : public BinaryValueImpl<T> {
		public:
			MulValueImpl(ValueImpl<T>* L, ValueImpl<T>* R) :
				BinaryValueImpl<T>(L->data * R->data, L, R) {}

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

		// Pow is a binary op: base ^ exponent, both differentiable graph nodes
		// (as opposed to a fixed scalar exponent).
		template <class T>
		class PowValueImpl : public BinaryValueImpl<T> {
		public:
			PowValueImpl(ValueImpl<T>* L, ValueImpl<T>* R) :
				BinaryValueImpl<T>(std::pow(L->data, R->data), L, R) {}

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

		// z = max(L, 0).  dz/dL = 1 if the (post-activation) output is positive,
		// else 0 -- checking `data > 0` is equivalent to checking the pre-ReLU
		// input's sign here, since relu(x) > 0 iff x > 0.
		template <class T>
		class ReLUValueImpl : public UnaryValueImpl<T> {
		public:
			ReLUValueImpl(ValueImpl<T>* L) :
				UnaryValueImpl<T>(L->data > 0 ? L->data : 0, L) {}

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

		// z = -L.  dz/dL = -1.
		template <class T>
		class NegValueImpl : public UnaryValueImpl<T> {
		public:
			NegValueImpl(ValueImpl<T>* L) :
				UnaryValueImpl<T>(-L->data, L) {}

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

		// z = L - R.  dz/dL = 1, dz/dR = -1.
		template <class T>
		class SubValueImpl : public BinaryValueImpl<T> {
		public:
			SubValueImpl(ValueImpl<T>* L, ValueImpl<T>* R) :
				BinaryValueImpl<T>(L->data - R->data, L, R) {}

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

		// z = L / R.  dz/dL = 1/R, dz/dR = -L/R^2.
		template <class T>
		class DivValueImpl : public BinaryValueImpl<T> {
		public:
			DivValueImpl(ValueImpl<T>* L, ValueImpl<T>* R) :
				BinaryValueImpl<T>(L->data / R->data, L, R) {}

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

		// Fuses an entire neuron's forward computation -- w . x + b, with an
		// optional ReLU -- into a single graph node, instead of 2*nin separate
		// Mul/Add nodes (plus one ReLU node) chained together. Doesn't derive
		// from BinaryValueImpl/UnaryValueImpl since it has a variable number of
		// children (the weight vector and input vector), not a fixed one or
		// two, so its refcounting and grad_propagate() are handled directly here.
		template <class T>
		class NeuronValueImpl : public ValueImpl<T> {
		protected:
			std::vector<ValueImpl<T>*> w;
			std::vector<ValueImpl<T>*> x;
			ValueImpl<T>* b;
			bool nonlin;

			static T computeData(const std::vector<ValueImpl<T>*>& w, const std::vector<ValueImpl<T>*>& x, ValueImpl<T>* b, bool nonlin) {
				T sum = b->data;
				for (std::size_t i = 0; i < w.size(); i++) sum += w[i]->data * x[i]->data;
				return nonlin ? (sum > 0 ? sum : 0) : sum;
			}

			// Same forward formula as above, but takes x's data() values already
			// gathered into one contiguous buffer (x_data) by the caller, instead
			// of reading x[i]->data through nin separate pointer dereferences.
			// This matters when many neurons share the same x (a whole Layer's
			// worth, as nn.h's parallel Layer::operator() does): gathering once
			// per *layer* instead of once per *neuron* turns nin*nout scattered
			// reads into nin scattered reads (the gather nn.h already did) plus
			// nin*nout contiguous ones (this dot product).
			//
			// w[i]->data is still read through a pointer per element -- each
			// neuron's own weights are unique to that neuron, so there's nothing
			// to hoist there the way there was for the shared x -- but gathering
			// them into a contiguous buffer first still helps, because it turns
			// one pointer-chasing loop plus one arithmetic loop into one
			// pointer-chasing loop (line below, inherently scalar -- there's no
			// way to vectorize a gather through scattered heap addresses without
			// changing the node's storage layout) followed by one loop over two
			// contiguous, aligned buffers that a compiler can actually
			// auto-vectorize. The buffer itself must be thread_local: this
			// function runs concurrently, once per neuron, from nn.h's
			// `#pragma omp parallel for`, so a plain function-local `static`
			// buffer shared across threads would just be a second data race of
			// exactly the same shape as the refcnt one described at the top of
			// this file. thread_local gives each worker thread its own buffer,
			// reused (not reallocated) across every neuron it processes, in this
			// call and in every later one -- so the allocation cost is paid once
			// per thread, not once per neuron per forward pass.
			static T computeData(const std::vector<ValueImpl<T>*>& w, const std::vector<ValueImpl<T>*>&, ValueImpl<T>* b, bool nonlin, const T* x_data) {
				thread_local std::vector<T> wbuf;
				wbuf.resize(w.size());
				for (std::size_t i = 0; i < w.size(); i++) wbuf[i] = w[i]->data;
				T sum = b->data;
				// reduction(+:sum) tells the compiler it may reorder/vectorize
				// this accumulation (floating-point addition isn't associative,
				// so without permission to reorder it, auto-vectorization of a
				// running sum is off the table) -- needs -fopenmp or
				// -fopenmp-simd to take effect; it's a harmless no-op pragma
				// otherwise.
			#pragma omp simd reduction(+:sum)
				for (std::size_t i = 0; i < w.size(); i++) sum += wbuf[i] * x_data[i];
				return nonlin ? (sum > 0 ? sum : 0) : sum;
			}

			static unsigned int computeHeight(const std::vector<ValueImpl<T>*>& w, const std::vector<ValueImpl<T>*>& x, ValueImpl<T>* b) {
				unsigned int h = b->height;
				for (std::size_t i = 0; i < w.size(); i++) h = std::max(h, std::max(w[i]->height, x[i]->height));
				return h + 1;
			}

		public:
			NeuronValueImpl(const std::vector<ValueImpl<T>*>& w_, const std::vector<ValueImpl<T>*>& x_, ValueImpl<T>* b_, bool nonlin_) :
				ValueImpl<T>(computeData(w_, x_, b_, nonlin_), computeHeight(w_, x_, b_)),
				w(w_), x(x_), b(b_), nonlin(nonlin_) {
				for (auto* wi : w) wi->refcnt++;
				for (auto* xi : x) xi->refcnt++;
				b->refcnt++;
			}

			NeuronValueImpl(const std::vector<ValueImpl<T>*>& w_, const std::vector<ValueImpl<T>*>& x_, ValueImpl<T>* b_, bool nonlin_, const T* x_data) :
				ValueImpl<T>(computeData(w_, x_, b_, nonlin_, x_data), computeHeight(w_, x_, b_)),
				w(w_), x(x_), b(b_), nonlin(nonlin_) {
				for (auto* wi : w) wi->refcnt++;
				for (auto* xi : x) xi->refcnt++;
				b->refcnt++;
			}

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
