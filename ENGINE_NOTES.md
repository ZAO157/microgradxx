# Engine notes

Design rationale for `engine.h`/`nn.h` that doesn't belong in the header
comments themselves -- the headers document *how to use* the public API;
this file documents *why it's built this way*.

## Ownership model

`Value<T>` is a thin, reference-counted handle (like a small hand-rolled
`shared_ptr`) -- cheap to copy, cheap to move, safe to store in containers,
reassign, let go out of scope, etc. The actual computation-graph node it
points to is a separate heap-allocated `ValueImpl<T>` (or a subclass),
carrying its own `refcnt`. A node is only deleted once its refcnt drops to
zero -- once nothing (neither a `Value<T>` handle nor another node using it
as a child) still needs it. This decoupling is deliberate: it lets a graph
node outlive whatever local variable or temporary container originally
produced it, which is exactly what's needed for `backward()` to walk the
*entire* graph after the forward pass's own local storage (e.g. one layer's
activations) has already gone out of scope.

Every `Value<T>` is guaranteed to hold a non-null impl, except a moved-from
`Value<T>` (`impl == nullptr`) -- the same convention the standard library
uses for moved-from objects. There is deliberately no default constructor:
a `Value<T>` you can hold is always either freshly built from a `T`/
`ValueImpl<T>*`, or a copy/move of one that was. A container that needs to
default-construct a slot before it has a real `Value<T>` to put there (e.g.
writing into a pre-sized vector from multiple threads) should hold
`std::optional<Value<T>>` instead -- see `Layer::operator()` in `nn.h`.

## Thread safety: refcnt

`refcnt` is `std::atomic<unsigned int>`, not a plain `unsigned int`. This
engine allows two or more graphs to share a child (the common case being
several `Neuron<T>`s in one `Layer` all taking the same input vector `x`):
building those nodes concurrently -- as `nn.h`'s parallel `Layer::operator()`
does -- means multiple threads increment the *same* child's refcnt at the
same time. A plain `unsigned int refcnt++` is a read-modify-write with no
atomicity guarantee; two threads can both read the same pre-increment value
and both write back `count+1`, silently losing one increment. That
under-count can (and, empirically, did, before this was fixed) let a node's
refcnt reach zero while something still points to it, triggering a
premature delete and a use-after-free the next time that "something" is
touched.

Encapsulation: every node's internals are `protected`, and `Value<T>` is
the only class ever granted `friend` access to it from outside the node
type hierarchy itself. A node's internals (its refcnt in particular) are
only ever safe to touch through the bookkeeping `Value<T>` already does in
its constructors/destructor/assignment operators.

## The backward pass, and why it's sequential-only right now

Every node's `height` is defined at construction as `1 + max(height of its
children)`, so a node's height is always strictly greater than any of its
children's. Bucketing nodes by height and processing buckets from tallest
to shortest is therefore a valid topological order: by the time a bucket
is reached, every node that could still add to its `grad` (i.e. every
parent, which necessarily sits at a taller height) has already run.
`queued` prevents a node reachable through more than one path from being
enqueued -- and therefore having its own `grad_propagate()` run -- more
than once per `backward()` call, which would otherwise double-count its
contribution to its children's gradients. `backward()` walks this order
single-threaded, so no synchronization is needed anywhere: there is only
ever one thread touching `queued`, `grad`, and the bucket vectors.

`engine.h` had, for a while, an opt-in OpenMP-parallel backward driver
(processing one whole height level at a time across multiple threads, with
an atomic dedup flag and a critical section guarding the shared bucket).
It measured 3-6x *slower* than sequential `backward()` on every shape
tried, including a genuinely large one on a higher-core-count machine: this
scalar (one `ValueImpl<T>` per elementary operation or fused neuron, not
per-tensor) engine does too little floating-point work per node for a
thread team's spin-up cost to be worth it. Rather than carry that
complexity in the main engine for a feature that wasn't paying for itself,
it's been set aside -- the last working version lives in `archive/`,
self-contained and still buildable, in case a more elegant approach is
worth building later. `engine.h` itself now has no parallel-backward
machinery of any kind, and `Value<T>::backward()` is always this
single-threaded driver.

## Why the forward pass's parallelism is a different story

`Layer::operator()` (in `nn.h`) parallelizes every neuron in a layer via
`#pragma omp parallel for`, and this genuinely helps: each neuron's forward
computation is a real, fused dot-product-plus-bias (see below), enough
work per thread to earn back the thread-team overhead once a layer is wide
enough (`buffer_threshold`, default 4 neurons). Unlike the backward pass,
there is *no data race to protect against* here at all: every neuron only
ever writes to the one graph node its own `operator()` call returns
(`out[i]`, a distinct element of a pre-sized
`std::vector<std::optional<Value<T>>>` -- writing by index to different
elements of an already-sized vector is safe from multiple threads; the
`std::optional` wrapper exists only because `Value<T>` has no default
constructor to pre-size a plain `std::vector<Value<T>>` with). Reading
shared input nodes (`x`) concurrently is likewise safe -- forward reads
never mutate `data`.

## The AVX2 gather techniques

Both `Value<T>::gather_data()` and `ValueImpl<T>::gather()` (and
`NeuronValueImpl::computeData`'s two overloads) use AVX2's 64-bit-index
gather instructions to dereference 4 node pointers directly into a SIMD
register in one instruction, instead of a scalar pointer-chasing loop.
The 64-bit-index gather is required, not `i32gather`: a 32-bit index would
truncate a real 64-bit heap address and silently read garbage on most
platforms -- so both `float` and `double` gather 4 elements per
instruction, not 8; the lane count is fixed by needing a full pointer's
worth of bits per index, not by the output element size.

`Value<T>::gather_data()` specifically avoids ever materializing a
temporary `std::vector<ValueImpl<T>*>` first: since `v` (a
`std::vector<Value<T>>`) is itself already contiguous, and (per the
`static_assert` in that function) `Value<T>` holds nothing but its `impl`
pointer, `v[i].impl` for 4 consecutive `i` is itself sitting contiguously
in `v`'s own storage -- reading it is a single SIMD load straight out of
`v`, not a gather. Only the second step (following those pointers to each
node's `data`) is a genuine gather. Benchmarked against the
extract-to-a-temporary-buffer-first approach: 1.3-7x faster across
nin=3..256, with the biggest win at this codebase's actual fan-ins
(nin=3,4), where the eliminated buffer write+read was relatively the most
expensive part.

`NeuronValueImpl::computeData`'s `x_data`-buffer overload exists because
gathering `x` once per *layer* (into one contiguous buffer, by
`Layer::forward_buffered`) instead of once per *neuron* turns what would
be nin*nout scattered reads of a shared input into nin scattered reads
(the one gather) plus nin*nout contiguous ones. `w`, unlike `x`, is never
shared across neurons, so there's nothing to hoist there -- it's still
gathered directly into a SIMD register per neuron, fused with the FMA
rather than materialized into an intermediate buffer first (benchmarked
1.5-2.7x faster than gathering into a `thread_local` scratch buffer first,
at this codebase's actual fan-ins).

`Layer`'s `buffer_threshold` (default 4) exists because building that
shared `x_data` buffer costs something too, and isn't worth it for very
narrow layers -- below the threshold, `forward_unbuffer` skips it and
lets each neuron gather its own scattered `x` reads directly.

## Offset computation, not `offsetof()`

Every gather routine computes the byte offset of `data` within
`ValueImpl<T>` via pointer arithmetic on one real, live node
(`reinterpret_cast<const char*>(&node->data) - reinterpret_cast<const
char*>(node)`) rather than via `offsetof()`. `ValueImpl<T>` has virtual
functions and so isn't a standard-layout type, which makes `offsetof()`
conditionally-supported (technically UB by the strict standard) even
though every mainstream compiler happens to implement it anyway. Pointer
arithmetic on an object that actually exists sidesteps that: `data` sits
at the same byte offset in every `ValueImpl<T>` regardless of its dynamic
subclass (a base subobject's layout doesn't depend on what derives from
it), so the offset measured from any one live node applies to all of
them.

## Where the parallel-backward exploration went

`archive/` holds the last working iteration of an OpenMP-parallel backward
driver -- a self-contained snapshot (its own copies of `engine.h`, `nn.h`,
a `backward_parallel.h` add-on, and its correctness test), independent of
the live `engine.h` one level up. See the comment at the top of
`archive/backward_parallel.h` for the numbers that led to setting it aside
and how to build it if you want to pick the idea back up.
