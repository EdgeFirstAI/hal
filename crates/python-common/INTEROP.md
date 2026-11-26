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
import time. This is not a bug in this project; it is a
[known limitation](https://github.com/PyO3/pyo3/issues/1444)
of how PyO3 (and CPython extension modules generally) manage
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
| Tensor | `__edgefirst_tensor__(access=None)` | `edgefirst_tensor_v2` | `#[repr(C)] TensorCapsulePayload` — a `TensorDesc` (shape, dtype, backing-store kind, capacity), a `QuantDesc` (scale/zero-point arrays, borrowed), the `plane_offset`, plus an optional host pin |
| ProtoData | `__edgefirst_protodata__()` | *(none — composed of two tensor capsules)* | `(mask_coefficients_capsule, protos_capsule, layout_str)`, each capsule an `edgefirst_tensor_v2` |
| Decoder | `__edgefirst_decoder__()` | `edgefirst_decoder_v1` | `#[repr(C)]` payload: raw pointer + `size_of`/`align_of` layout guard |

### Quantization

The tensor capsule carries quantization metadata alongside the descriptor,
in a `#[repr(C)] QuantDesc`: a length, a channel axis (`+1`, so all-zeroes
means "none"), and pointers to the scale and zero-point arrays. The arrays
are **borrowed**, not inlined — per-channel quantization is variable-length
— and the payload's `quant_keepalive` (an `Arc<Quantization>` cloned out of
the producing tensor) owns them for the capsule's life, on exactly the terms
`TensorDesc::ptr` borrows the producer's host address. A consumer copies the
values out during import and never retains the pointers.

It rides in the capsule rather than in `TensorDesc` because `TensorDesc` is
also the C ABI's descriptor, and the C libraries pass real `ef_tensor`
handles into one shared `libedgefirst_tensor.so` — nothing there rebuilds a
tensor from a descriptor, so nothing there loses the metadata. The Python
packages are the only consumers that reconstruct, so they are the only ones
that need it on the wire.

### Plane offset

The payload also carries `plane_offset`, the byte offset within the backing buffer where this tensor's data starts — what `Tensor::view()` records for a sub-region. It is in the capsule for the same reason quantization is: without it a consumer converts the parent buffer's origin instead of the requested sub-region.

The consumer applies it only to a **handle-based** import. Under the host
kinds (`kind::MEM`, `kind::SHM`) the descriptor's `ptr` is the producer's pinned address for the
view itself, so the offset is already in it and re-applying it would advance
past the sub-region a second time. `DMABUF` (dup's the fd), `IOSURFACE`
(looks the surface up by id), `D3D11_TEXTURE` (opens the NT handle) and
`PBO` (by buffer id) all re-derive the base from a handle naming the whole
parent buffer, so for those it must be put back. See
`interop::apply_plane_offset`.

`Tensor::set_plane_offset` syncs the storage-internal offset that `map()`
adds for every backing but one: `Mem`, `Pbo`, `Shm` and `Dma` on Linux,
macOS/iOS and Windows. Its match is exhaustive — there is no `_ => {}`, so
a backing left out of it is a compile error rather than a silent
fall-through — and the lone exception is Android's `AHardwareBufferTensor`,
whose arm is an explicit no-op and which no import path reaches today. The
clear sites in `set_format` and `reshape` are exhaustive the same way.

On macOS/iOS the descriptor import restores an IOSurface view's pitch (`restore_imported_row_stride` includes `IOSURFACE`; a surface's `bytesPerRow` is shared, unlike a D3D11 staging pitch), and the GL engine's Apple leaf refuses to zero-copy attach a source carrying a plane offset, as the Windows leaf does, because the ANGLE IOSurface binding has no offset attribute.

On Windows the D3D11 import accepts a packed window, opens the whole texture and narrows it, keeping the texture's pitch as the row stride; the storage offset is then written back by `set_plane_offset` and cleared by `set_format` and `reshape`. Two details follow from the texture being the unit of import. The ANGLE image binds the whole texture from its origin and cannot express an offset, so the engine's ANGLE leaves (D3D11 and IOSurface) refuse to attach a *source* carrying a plane offset and the engine uploads it through `map()` instead. A *destination* rebuilt from a descriptor has the offset but not the `view_origin` the engine lowers a fresh view's tile to a viewport from, so the engine asks the platform whether a zero-copy destination can be placed (`GlPlatform::dst_import_places`) and lowers one that cannot to the mapped texture path, whose readback writes through `map()` at the offset. The offset itself is applied as the producer measured it: a texture tensor measures it in the row pitch of its staging texture, and a consumer opens the same texture on the same adapter, so its own staging has the same pitch. The descriptor's stride is not used to translate it: a single-row `view()` records a tight stride for its map span while its offset is still in the pitch, so dividing by that stride would name a different row.

For the other backings: `Mem`/`Shm` report `kind::MEM`/`kind::SHM` and take the pinned-pointer path; Android reports `kind::DMABUF` whose import arm is Linux-only, so it fails instead of reconstructing; and a `view()` of a PBO-backed image stays PBO-backed, so it reaches the PBO arm and carries its own offset. See `interop::apply_plane_offset`'s doc comment for the per-backing accounting.

`interop::reconstruct` — the same-module path, where no capsule is involved
— carries the offset across the same way, because it reconstructs through
the identical `TensorDesc`.

A consumer never constructs one of these by hand; it calls the producer's
method and reads the capsule back through the matching `interop::*Arg`
extractor (`crates/python-common/src/interop.rs`). This document describes
the Python-visible half of the contract — the shape a third-party producer
or consumer needs to match.

### Multi-plane tensors are refused

The payload describes one plane. A tensor whose chroma plane is a separate tensor (`is_multiplane()`: a two-buffer NV12 such as an `import_image(..., chroma_fd=...)`, chroma at an offset in the same buffer, a decoded video frame, a `VideoEncoder.create_source()` image) cannot be described by it, so `__edgefirst_tensor__` raises `NotImplementedError` ("a tensor whose chroma plane is a separate tensor cannot be passed to another edgefirst package") instead of exporting the luma plane alone. The proto data capsules of `edgefirst.decoder` refuse the same way. Such a tensor works only with the package that made it, where calls use it directly.

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
# Correct — works regardless of which edgefirst.* package produced obj
if hasattr(obj, "__edgefirst_tensor__"):
    ...

# Wrong — always False for an object from a sibling package, even though
# it is a perfectly valid tensor. See "Why" above.
if isinstance(obj, edgefirst.image.Tensor):
    ...
```

Every `edgefirst.*` entry point that accepts a foreign object follows the
same rule internally: try a same-module downcast first (the fast, zero-copy
path when producer and consumer are the same package), and if that fails,
call the protocol method rather than rejecting the object. Write your own
producers and consumers the same way.

## Backing-store kinds

The descriptor's `kind` says what backs the tensor, and it is what decides how
to read `handle`, `ptr` and `sync`. A consumer that does not recognize a kind
must refuse the tensor rather than guess.

The kind codes are the `TensorMemory` codes (`TensorMemory::code()`, the C
header's `ef_storage_kind`), re-exported as `protocol::kind`, so one integer
names one backing on every surface. A producer reports the concrete backing:
a `TensorMemory::DmaBuf` tensor is `DMABUF` on Linux and Android, `IOSURFACE`
on macOS/iOS and `D3D11_TEXTURE` on Windows.

`sync` is meaningful only when the `SYNC_PRESENT` flag is set, and its
flavour is keyed by kind as well.

| `kind` | Backing | `handle` | `ptr` | `sync` flavour |
|---|---|---|---|---|
| `MEM` (0) | heap memory | `-1` | host address when pinned, else null | none defined; `SYNC_PRESENT` must be clear |
| `SHM` (1) | POSIX shared memory | `-1` | host address when pinned, else null | none defined; `SYNC_PRESENT` must be clear |
| `DMABUF` (2) | Linux dma-buf | the fd | host address when pinned, else null | a `sync_file` fd the consumer closes |
| `IOSURFACE` (3) | Apple IOSurface | the surface id | host address when pinned, else null | none defined; `SYNC_PRESENT` must be clear |
| `PBO` (4) | OpenGL pixel buffer object | the buffer id | `*const PboOpsVtable`, or null | a `GLsync`, valid only in the producer's share group |
| `CUDA` (5) | CUDA device memory | `-1` | device pointer, not host-addressable | a `cudaEvent_t` |
| `D3D11_TEXTURE` (6) | Windows `ID3D11Texture2D` | the texture's NT shared handle | the device fence's NT shared handle, or null | a fence value, the last recorded GPU write |

`D3D11_TEXTURE` is the only kind a producer sets `sync` for, and it carries a **value on a timeline** rather than a handle: the fence it names is the one whose NT handle `ptr` carries, so a producer that sets `SYNC_PRESENT` must fill in `ptr` as well. A descriptor that sets the flag with a null `ptr` is refused — it advertises a completion nobody can wait on. Both handle values are valid in the producing process, other modules of the same process included, and the import duplicates whatever it keeps, exactly as the `DMABUF` kind does with its fd.

### CPU access flags

A producer reports its tensor's declared `CpuAccess` in three `flags` bits: `CPU_ACCESS_DECLARED` (bit 1) says the report is present, `CPU_READ` (bit 2) and `CPU_WRITE` (bit 3) carry the directions. `import_descriptor` narrows the imported alias to that declaration with `restrict_cpu_access`, so a CPU map the producer's tensor would refuse is refused on the alias too, and `Tensor.declared_cpu_access` on the alias reports it. A consumer that must pin a host descriptor for writing reads the bits first and asks for `access="write"` rather than `"readwrite"` when the producer declared no reads; a pin outside the producer's declaration raises `PermissionError`. A descriptor without `CPU_ACCESS_DECLARED` imports with the default `ReadWrite` declaration.

The narrowing is the consumer's to apply. A consumer that does not read these bits imports the alias as `ReadWrite` and so cannot enforce the producer's declaration: a CPU map through that alias succeeds where the producer's own tensor would refuse it. A producer that relies on its declaration being honoured (a `None` buffer nothing may touch on the CPU, a `Write` buffer nothing may read) needs consumers that read the bits.

An `access="read"` capsule over a texture tensor pins host memory, but no consumer can reach the pixels through that address, because the pixels live in the texture. The pin is a keepalive only. A consumer that wants the bytes imports the descriptor and maps the imported tensor.

The `PBO` kind's `ptr` carries a `PboOpsVtable` address, and that vtable embeds an `ef_client_state` — an opaque context plus a `retain`/`release` pair — alongside the map and unmap function pointers. A consumer's reconstruction takes its own reference on that channel through `retain`, so it does not depend on the producer's keepalive still being held: a reconstructed PBO tensor keeps the buffer's map and unmap reachable even after the producing tensor and its capsule are gone. `retain` and `release` govern the channel only. The producer's own destructor stays the sole caller of `glDeleteBuffers`, and a reconstructed tensor's delete operation is a no-op.

### View flag

`VIEW` (bit 4) marks a producer that addresses part of a larger buffer: a `view()`, or a `batch()` element of a packed image, at any offset including zero. The descriptor cannot carry the view's parent geometry, so `import_descriptor` marks the alias as a view instead (`is_view`), and a CPU write into it maps `ReadWrite` exactly as the producer's view would: its rows share cache lines with bytes outside it, so a write-only map is refused for a `Write`-declared view on both sides. The bit is clear for a whole tensor, which is also what a producer that does not set the bit reports.

### Carrying a texture between processes

The descriptor protocol is an in-process contract: its handle values mean
nothing in another process. The blob (`edgefirst_tensor::blob`,
`ef_tensor_export` / `ef_tensor_import`) is what crosses that boundary, and on
Windows it does so without any file descriptors — the `fds` array is empty and
`fds_out` may be null.

A reference-mode blob carries the producer's `pid` at byte 28 of the fixed
header, and each D3D11 plane carries 24 bytes of `handle_bytes`: three
little-endian `u64`s, the texture NT handle value, the fence NT handle value
and the fence value of the last recorded GPU write. The importer opens the
producer with `OpenProcess(PROCESS_DUP_HANDLE, ...)` and calls
`DuplicateHandle` for each handle into its own process. Two things follow from
that, and each fails distinctly rather than silently producing a handle-less
tensor: a producer that has already exited gives `NotFound` ("exporting process
`<pid>` is gone"), and a producer this process is not allowed to open for
`PROCESS_DUP_HANDLE` — on Windows, a different user, or one whose token does
not grant it — gives a permission error naming the pid.

Inline mode copies the pixels instead and has neither requirement.

## Lifetime and ownership

The tensor descriptor **borrows**. It is only valid while the capsule that
carries it is alive:

- `access=None` requests no pin. The descriptor still carries shape, dtype,
  backing-store kind and the native handle (dma-buf fd, IOSurface id, PBO
  id, CUDA device pointer, D3D11 texture NT handle) — everything a zero-copy
  GPU/DMA consumer needs — but no host address is guaranteed, and `ptr` is
  null except where the table above gives that field another meaning.
- `"read"` / `"write"` / `"readwrite"` pins host memory for that access and
  fills in `ptr`. The pin is owned by the capsule: dropping the capsule
  releases it. A consumer that needs the address to outlive the capsule
  (rather than just the call it was extracted for) must not just hold onto
  `ptr` — it must dup the underlying fd (`Tensor.dmabuf_clone()`), duplicate
  the texture handle (`Tensor.d3d11_shared_handle()`), or retain the surface
  itself.

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
capsule name of its own, but composes two `edgefirst_tensor_v2` capsules, so
a tensor-capsule name bump covers it automatically.

DLPack and Arrow gate the same way. DLPack's `DLManagedTensorVersioned` struct carries a version field *and* travels under its own capsule name, `dltensor_versioned`: the version field of a struct cannot be read before its layout is known, which is why the name, not the field, is this protocol's gate. Arrow's PyCapsule interface never versions its names (`arrow_schema`, `arrow_array`): its C Data Interface structs are frozen on release, "should not change in any way – including adding new members", and anything incompatible becomes a separate specification. Both designs agree on the underlying point — the identity of the wire format has to be knowable before the payload is touched.

**A capsule name is frozen once it ships in a tagged release.** From then on every layout change takes the name with it, including a change to the payload's Rust-layout tail and a field slotted into existing padding.

**Tensor** (`edgefirst_tensor_v2`, `TensorCapsulePayload` in `crates/python-common/src/interop.rs`). The payload carries the descriptor, a `QuantDesc` and a `plane_offset`. A producer of another capsule name and a `_v2` consumer meet at the name check and the object is reported as not exportable, rather than its payload being misread at `_v2`'s layout. The next payload change, however small, is `_v3`.

`PboOpsVtable` is not part of the payload, but it crosses between separately-compiled copies of `edgefirst-tensor` inside one process behind the same capsule name. A change to its layout therefore takes the capsule name with it, on the same terms as a `TensorCapsulePayload` change: a differently-sized vtable behind an unchanged name is read at the wrong offsets with nothing to catch it. The importer holds its own channel reference, so `TensorCapsulePayload::pbo_keepalive` is redundant; removing it is a payload change and takes the capsule name with it (`edgefirst_tensor_v3`).

The descriptor's own `version` field is `ABI_VERSION` (`2`, checked by `TensorDyn::import_descriptor` in `crates/tensor/src/tensor_dyn.rs`), and it is the second line of defense, not the first: it covers a change to what a same-sized `TensorDesc`'s fields *mean*, which a name bump does not imply by itself. Any change to the layout — a new field, a reordering, a size change — goes to `edgefirst_tensor_v3` in the same commit that makes it. Version `2` numbers `kind` with the `TensorMemory` codes, and a consumer refuses a descriptor of another version instead of reading its kinds with that numbering. `TensorCapsulePayload`'s `#[repr(C)]` prefix (`desc` + `quant` + `plane_offset`) is pinned by a `const` assertion in `interop.rs`; `crates/tensor/tests/protocol.rs` pins `TensorDesc` alone, which is only the first field of that prefix.

**Decoder** (`edgefirst_decoder_v1`, `DecoderCapsulePayload` in the same file): the same rule. A layout change to `DecoderCapsulePayload`, or to what its `size_of`/`align_of` guard covers, goes to `_v2`.

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

**This is accepted residual risk, not a solved problem.** Two `Decoder` layouts of equal size and alignment but with permuted field order pass the guard undetected. Matching `size_of`/`align_of` narrows the failure mode from "any layout drift whatsoever" down to "a layout drift that also happens to change size or alignment" — it does not eliminate it. If you are pinning dependency versions across `edgefirst-image` and `edgefirst-decoder` in your own project, keep them matched; the guard is a safety net, not a substitute for that.
