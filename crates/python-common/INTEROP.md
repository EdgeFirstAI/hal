# Cross-package interoperability

The four `edgefirst-*` Python packages (`edgefirst-tensor`, `edgefirst-codec`,
`edgefirst-image`, `edgefirst-decoder`) each ship as an independent PyO3
extension module — see [`ARCHITECTURE.md`](ARCHITECTURE.md). This document
is for anyone who hits a `TypeError` mentioning this file, and for anyone
implementing their own producer or consumer of an `edgefirst.*` object.

## Why

Every `edgefirst.*` extension module statically links its own copy of the
Rust binding code, so every module caches its **own** PyO3 type objects —
even for a class with an identical name and identical fields. Concretely:

```python
from edgefirst.tensor import Tensor as TTensor
from edgefirst.codec import Tensor as CTensor

isinstance(TTensor(...), CTensor)  # always False
```

`TTensor` and `CTensor` are both named `Tensor`, both wrap the same Rust
type, and both report `__module__ == "edgefirst.tensor"` — but they are
different Python classes, because each `.so` registered its own copy at
import time. This is not a bug in this project; it is
[PyO3 issue #1444](https://github.com/PyO3/pyo3/issues/1444), a known
limitation of how PyO3 (and CPython extension modules generally) manage
per-module type identity. There is no supported way to share a `#[pyclass]`
type object across two independently-linked `.so` files.

So passing a tensor created by `edgefirst.codec` into an `edgefirst.image`
function cannot be solved by `isinstance`/downcasting — the object has to
identify itself structurally instead. Every `edgefirst.*` package that needs
to accept an object from a sibling package does it through a **capsule
protocol**: a dunder method that returns a
[`PyCapsule`](https://docs.python.org/3/c-api/capsule.html) wrapping a
plain-old-data description of the object, following the shape popularized by
the [Arrow PyCapsule Interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html)
for exactly this cross-extension problem. numpy, pyarrow and DLPack all use
the same duck-typed pattern for the same reason.

## The protocols

| Object | Producer method | Capsule name | Payload |
|---|---|---|---|
| Tensor | `__edgefirst_tensor__(access=None)` | `edgefirst_tensor_v1` | `#[repr(C)] TensorCapsulePayload` — an `TensorDesc` (shape, dtype, backing-store kind, capacity) plus an optional host pin |
| ProtoData | `__edgefirst_protodata__()` | *(none — composed of two tensor capsules)* | `(mask_coefficients_capsule, protos_capsule, layout_str)`, each capsule an `edgefirst_tensor_v1` |
| Decoder | `__edgefirst_decoder__()` | `edgefirst_decoder_v1` | `#[repr(C)]` payload: raw pointer + `size_of`/`align_of` layout guard |

A consumer never constructs one of these by hand; it calls the producer's
method and reads the capsule back through the matching `interop::*Arg`
extractor (`crates/python-common/src/interop.rs`). This document describes
the Python-visible half of the contract — the shape a third-party producer
or consumer needs to match.

## `typing.Protocol` definitions

These are published as real, importable classes (`edgefirst.tensor.EdgeFirstTensorExportable`,
`edgefirst.decoder.EdgeFirstDecoderExportable`, `edgefirst.decoder.EdgeFirstProtoDataExportable`)
so that annotating a cross-package parameter doesn't require redeclaring the
protocol. Copy them instead if you would rather not add an `edgefirst.*`
import dependency — the protocol is duck-typed by design and does not care
where the `Protocol` class itself came from:

```python
from typing import Optional, Protocol, Tuple


class EdgeFirstTensorExportable(Protocol):
    """Anything that can hand a tensor across an edgefirst.* package
    boundary. `access` is `None` (no pin — shape/format/native handle
    only), or `"read"` / `"write"` / `"readwrite"` to pin host memory
    and fill in the address."""

    def __edgefirst_tensor__(self, access: Optional[str] = None) -> object: ...


class EdgeFirstDecoderExportable(Protocol):
    """Anything that can hand a Decoder across an edgefirst.* package
    boundary."""

    def __edgefirst_decoder__(self) -> object: ...


class EdgeFirstProtoDataExportable(Protocol):
    """Anything that can hand mask-prototype data across an edgefirst.*
    package boundary."""

    def __edgefirst_protodata__(self) -> Tuple[object, object, str]: ...
```

The capsule itself has no useful static type (`object`/`PyCapsule` — Python's
`typing` module has no capsule type), so these protocols only get a caller
past the *first* mile: whether an object is exportable at all. What is
inside the capsule is the ABI described below, not something a type checker
verifies.

## Consumer guidance: duck type, never `isinstance`

```python
# CORRECT — works regardless of which edgefirst.* package produced obj
if hasattr(obj, "__edgefirst_tensor__"):
    ...

# WRONG — always False for an object from a sibling package, even though
# it is a perfectly valid tensor. See "Why" above.
if isinstance(obj, edgefirst.image.Tensor):
    ...
```

Every `edgefirst.*` entry point that accepts a foreign object follows the
same rule internally: try a same-module downcast first (the fast, zero-copy
path when producer and consumer are the same package), and if that fails,
call the protocol method rather than rejecting the object. Write your own
producers and consumers the same way.

## Lifetime and ownership

The tensor descriptor **borrows**. It is only valid while the capsule that
carries it is alive:

- `access=None` requests no pin. The descriptor still carries shape, dtype,
  backing-store kind and the native handle (dma-buf fd, IOSurface id, PBO
  id, CUDA device pointer) — everything a zero-copy GPU/DMA consumer needs —
  but `ptr` is null and no host address is guaranteed.
- `"read"` / `"write"` / `"readwrite"` pins host memory for that access and
  fills in `ptr`. The pin is owned by the capsule: dropping the capsule
  releases it. A consumer that needs the address to outlive the capsule
  (rather than just the call it was extracted for) must not just hold onto
  `ptr` — it must dup the underlying fd (`Tensor.dmabuf_clone()`) or retain
  the surface itself.

A consumer **may call the producer method more than once per operation** —
for example, `TensorArg::extract` retries with `access="read"` when an
`access=None` call comes back host-backed with no address. Producers
(including third-party ones) must therefore implement `__edgefirst_tensor__`
/ `__edgefirst_decoder__` / `__edgefirst_protodata__` as **side-effect
free**: repeated calls, with the same or different arguments, must be safe
and must not accumulate state.

The `Decoder` capsule borrows even more narrowly: it is valid only for the
duration of the call it is passed into and must never be stored past that
call.

## Decode write-back (`decode_into` / `decode_file_into`)

`edgefirst.codec.decode_into()` and `decode_file_into()` accept a foreign
destination the same way every other cross-package entry point does. The
pixel write is always correct and always happens — that part does not
depend on anything below.

Decoding also determines the image's format, dimensions and colorimetry,
and a same-module decode (`Tensor.decode_image`) updates the destination
tensor itself to reflect them. The cross-package functions try to leave a
foreign destination in that same state, but only on a **best-effort**
basis: they call the destination's `configure_image(width, height, format)`
method and set its `colorimetry` attribute, and if either is missing, or
`colorimetry` turns out to be read-only, that step is silently skipped — a
warning is logged, nothing is raised, and the decode is still reported as
successful. `EdgeFirstTensorExportable` above requires only
`__edgefirst_tensor__`; a producer that implements just that minimum is
fully conforming, and its destinations decode correctly, but they will not
pick up the format/dimension/colorimetry write-back unless they also
implement `configure_image()` and a settable `colorimetry`.

**The returned `ImageInfo` always describes the decode accurately,
regardless of what the destination does or does not implement.** A caller
that needs the decoded format or dimensions reliably — rather than only
when the destination happens to be a `Tensor` from this crate — should read
them from `ImageInfo`, not from the destination tensor.

## Versioning

**The rule, stated once for all three capsules: a layout change to a
capsule's payload gets a new capsule name, for every capsule in this
document, not only the tensor one.** A consumer's own `size_of`/`version`
check inside the payload cannot substitute for this, because it runs
*after* the payload has already been read at this build's (possibly larger
or differently-shaped) type — by the time a mismatch could be noticed, the
out-of-bounds or misaligned read has already happened. The capsule name
check happens first, via `PyCapsule::pointer_checked`, and gates the unsafe
read itself: old producers and consumers keep working against each other,
and a new consumer talking to an old producer degrades to "not exportable"
— rejected before any byte of the mismatched payload is read — instead of
misreading memory. `ProtoData` inherits this transitively: it carries no
capsule name of its own, but composes two `edgefirst_tensor_v1` capsules, so
a tensor-capsule name bump covers it automatically.

This is the same conclusion DLPack reached, the hard way. DLPack shipped
`dltensor` with no version anywhere, found there was no way to add a field
without silently breaking every existing consumer, and in v0.7/1.0
deliberately broke ABI: a new `DLManagedTensorVersioned` struct carrying a
version field *and* a new capsule name, `dltensor_versioned`. The new field
alone would not have been enough — you cannot read the version field of a
struct whose layout you do not yet know — which is exactly why the name,
not the field, is this protocol's gate. Arrow's PyCapsule interface takes
the other route and never versions its names (`arrow_schema`,
`arrow_array`): its C Data Interface structs are frozen on release, "should
not change in any way – including adding new members", and anything
incompatible would become a separate specification. Both designs agree on
the underlying point — the identity of the wire format has to be knowable
before the payload is touched.

**The rule binds at first release, not during development.** Until
`edgefirst_tensor_v1` ships in a tagged release there is no producer or
consumer in the wild to protect, so the descriptor can still grow freely
under the same name -- and it has, twice, to reserve `flags` and `sync`
before the layout is fixed. Renaming during development instead buys
nothing and actively misleads: a published `_v3` would tell every future
maintainer that two earlier wire formats exist and might need compatibility
consideration, when none ever did. Once a release goes out, every layout
change takes the name with it.

**Tensor** (`edgefirst_tensor_v1`, `TensorCapsulePayload` in
`crates/python-common/src/interop.rs`): this is the protocol's initial
published version; no earlier capsule name was ever released. The
descriptor's own `version` field is `ABI_VERSION` (currently `1`,
checked by `TensorDyn::import_descriptor` in
`crates/tensor/src/tensor_dyn.rs`), and it is the second line of defense,
not the first: it covers a hypothetical future change to what a same-sized
`TensorDesc`'s fields *mean*, which a name bump does not imply by itself.
Any change to the layout — a new field, a reordering, a size change — goes
to `edgefirst_tensor_v2` in the same commit that makes it, not as a
follow-up.

**Decoder** (`edgefirst_decoder_v1`, `DecoderCapsulePayload` in the same
file): the same rule, and likewise an initial version. A layout change to
`DecoderCapsulePayload`, or to what its `size_of`/`align_of` guard covers,
goes to `_v2`.

## The identity caveat

The protocol makes objects **usable** across packages. It does not make them
**identical**. `isinstance` stays `False` by design — that is the same
trade Arrow, numpy and DLPack all made for the same reason (see "Why"
above), not a gap this protocol is meant to close.

**Value types are the exception.** `PixelFormat`, `TensorMemory`, the colour
axis enums (`ColorSpace`/`ColorTransfer`/`ColorEncoding`/`ColorRange`), and
`Region` are still independently registered per package — the identity
problem is the same — but they carry hand-written `__eq__`/`__hash__` that
compares **by value** (discriminant or field values) rather than by native
identity or a bare int. So while `isinstance` is still `False` across
packages for these too, `==`, `!=`, and use as dict keys or set members all
work correctly across packages: an `edgefirst.codec.PixelFormat.Rgb`
compares equal to, and hashes identically to, an
`edgefirst.image.PixelFormat.Rgb`.

## Accepted risk: the `Decoder` layout guard

`ProtoData` is sound by construction: `__edgefirst_protodata__` composes the
already-proven `__edgefirst_tensor__` protocol (it returns two tensor
capsules plus a layout string) rather than describing `ProtoData`'s own
memory layout. There is no raw pointer and no version coupling between
packages to get wrong.

`Decoder` cannot do that. It is a live Rust object carrying internal
post-processing state, not a value that decomposes into tensors and enums,
so `__edgefirst_decoder__` has no choice but to hand across a **raw
pointer**. The consumer's `unsafe { &*ptr }` is only sound if the producer's
and consumer's copies of `edgefirst-decoder` agree bit-for-bit on
`Decoder`'s memory layout. The guard
(`interop::DecoderArg::extract` in `crates/python-common/src/interop.rs`)
checks `size_of::<Decoder>()` and `align_of::<Decoder>()` equality — the
payload carries only the raw pointer plus those two values, nothing else.
An earlier revision also carried a version string as diagnostic-only text,
dropped because a version string says nothing about which Cargo features
were compiled in on each side (`crates/python-image/pyproject.toml` pins
`edgefirst-decoder` with a `~=` compatible-release specifier, which admits
patch releases that a strict version-string check would wrongly reject) and
because it was itself a fat pointer of unspecified internal layout riding
inside a `#[repr(C)]`
struct — see `crates/python-common/src/interop.rs`'s `DecoderCapsulePayload`
doc comment for the full story.

State this plainly: **this is accepted residual risk, not a solved
problem.** Two `Decoder` layouts of equal size and alignment but with
permuted field order would still pass the guard undetected. Matching
`size_of`/`align_of` narrows the failure mode from "any layout drift
whatsoever" down to "a layout drift that also happens to change size or
alignment" — it does not eliminate it. If you are pinning dependency
versions across `edgefirst-image` and `edgefirst-decoder` in your own
project, keep them matched; the guard is a safety net, not a substitute for
that.
