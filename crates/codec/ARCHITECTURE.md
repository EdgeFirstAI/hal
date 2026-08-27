# EdgeFirst Codec Architecture

## Overview

The `edgefirst-codec` crate provides image decoding into pre-allocated
tensor buffers. It is designed for real-time vision pipelines where the
anti-pattern of allocating new output buffers on every frame must be
avoided.

The core principle: **allocate once at init, decode in the hot loop**.

A second principle drives the data path: by default the decoder emits each
image in its **native pixel format** and does nothing else — no colour
conversion, no resize, no rotation. JPEG decodes to the NV format matching its
own chroma sampling — `Nv12` (4:2:0), `Nv16` (4:2:2), `Nv24` (4:4:4) — or
`Grey` (greyscale); PNG decodes to `Rgb` / `Rgba` / `Grey`. Everything beyond
raw decode — colour-space conversion, EXIF orientation, resize, crop — belongs
to [`ImageProcessor::convert()`](../image), which runs on the GPU where
available. This keeps the decode path branch-free and lets a single
`convert()` fold all the geometry/colour work into one pass.

One measured exception: `ImageDecoder::set_output_format` lets a caller opt
into a **pure CPU fused decode output** — `Rgb` (4:4:4 or native 4:2:0
sources) or `Nv12` (any colour source) — where the conversion happens inside
the software MCU
write stage (not a GPU hybrid). Callers still run `ImageProcessor::convert()`
afterward for model-input preprocessing. See
[Fused Decode Output](#fused-decode-output-opt-in-pure-cpu).

## Crate Position in the Workspace

```
edgefirst-tensor ← edgefirst-codec ← edgefirst-image (re-export)
                                    ← edgefirst-hal (re-export)
                                    ← crates/python (bindings)
```

`edgefirst-codec` depends only on `edgefirst-tensor` plus `zune-png`
(for PNG decoding) and `kamadak-exif` (for reading EXIF orientation). JPEG
decoding uses a custom from-scratch decoder with no external dependencies.
The two default-on hardware features add a little on Linux — `v4l2` pulls
`nix` and `libc` (Linux-target only), `nvjpeg` pulls `libloading` — and both
compile to nothing elsewhere. Neither adds a link-time GPU dependency:
`nvjpeg` resolves `libnvjpeg.so.12` through `dlopen` at runtime. The crate has
no dependency on `edgefirst-image` or any GPU libraries, keeping the
dependency graph clean.

## Module Map

| Module       | Purpose                                         |
|--------------|-------------------------------------------------|
| `lib.rs`     | Crate root, public re-exports                   |
| `error.rs`   | `CodecError` enum (capacity / dtype / unsupported / IO / V4L2) |
| `pixel.rs`   | `ImagePixel` trait (u8, u16, i8, i16, f32)      |
| `options.rs` | `ImageInfo` struct (decoded metadata + reported EXIF orientation) |
| `exif.rs`    | EXIF orientation parsing → `(rotation_degrees, flip_horizontal)`; never applied |
| `decoder.rs` | `ImageDecoder` struct, magic-byte format detection, decode dispatch |
| `traits.rs`  | `ImageLoad` extension trait for Tensor/TensorDyn|
| `jpeg/`      | Custom baseline JPEG decoder + V4L2 / nvJPEG backends (see below) |
| `png.rs`     | Native-format PNG decode with 8-bit and native 16-bit paths |

### JPEG Module Map (`jpeg/`)

| Module           | Purpose                                              |
|------------------|------------------------------------------------------|
| `mod.rs`         | `JpegDecoderState`, `decode_jpeg_into<T>()`, `native_format()` + `resolve_output_format()`, nvJPEG→V4L2→CPU dispatch seam, EXIF reporting |
| `types.rs`       | `Component`, `SamplingFactor`, `ImageHeader`, `QuantTable`, `ZIGZAG` |
| `markers.rs`     | SOF/SOS/DQT/DHT/DRI/APP marker parsing               |
| `bitstream.rs`   | 64-bit bit buffer with FF/00 byte-stuffing, bulk refill |
| `cpu.rs`         | Runtime `NeonTier` / `IntelTier` probes: Huffman lookahead width, entropy prefetch, and SIMD kernel selection per micro-architecture / ISA |
| `huffman.rs`     | Tier-sized lookahead Huffman LUT + combined code+magnitude fast-AC LUT, `decode_block()` emitting quantised `i16` coefficients |
| `color.rs`       | Fused YCbCr→RGB row conversion (Q14 `SQRDMULH` / matching SSE4.1 `qrdmulh`, NEON + Intel + bit-exact scalar) |
| `idct.rs`        | IDCT dispatcher (scalar/NEON/x86 via `NeonTier`/`IntelTier`); kernels dequantise internally |
| `idct/fast.rs`   | Opt-in AAN `ifast`-class IDCT (`DctMethod::Fast`, off by default): AAN scale factors folded into the dequant table, four Q15 `SQDMULH` constants, 16-bit lanes end to end; scalar reference + NEON kernel, bit-exact to each other. COCO-measured accuracy vs the default kernel: cosine ≥ 0.99985, PSNR ≥ 42 dB, max pixel delta 24 |
| `idct/scalar.rs` | Two-pass Loeffler 8×8 IDCT with DC-only fast path; the reference every SIMD kernel is asserted against, so it wraps and saturates where the vector kernels are forced to |
| `idct/neon.rs`   | NEON 8×8 IDCT on 16-bit coefficients: 8-lane dequant, lane-indexed `vmull` Loeffler butterfly, sparse bottom-row/right-column shortcuts, register-resident workspace |
| `idct/sse2.rs`   | x86 8×8 IDCT: 16-bit-lane `islow` Loeffler, `pmullw` dequant + `pmaddwd` butterflies (shared by all Intel tiers) |
| `idct/avx2.rs`   | VEX-encoded build of the `sse2.rs` kernel |
| `mcu.rs`         | MCU decode loops — dedicated 4:4:4 1×1 direct loop (`decode_image_444_direct`: tables resolved to plain locals, no sampling loops; the generic loop's frame spilled past the register file and cost 4–8% on every core) + generic sampling loop — `McuScratch`, native `Grey`/`Nv12`/`Nv16`/`Nv24` row writes, fused `Rgb`/`Nv12` writes (NEON/SSE chroma downsample for `Nv12`; `write_rgb_rows_420` box-upsamples native 4:2:0 chroma into `Rgb`), chroma downsample for the non-matching subsamplings (`avg_block`) |
| `v4l2/`          | Optional Linux hardware JPEG backend (see below)     |
| `nvjpeg/`        | Optional nvJPEG GPU backend (Linux + CUDA, see below) |

Both hardware backends expose crate-level runtime probes —
`edgefirst_codec::v4l2_available()` (opens and drops the first working
decoder device; honours `EDGEFIRST_DISABLE_V4L2`) and
`edgefirst_codec::nvjpeg_available()` (honours the `EDGEFIRST_ENABLE_NVJPEG`
opt-in) — for consumers that must fail fast rather than silently fall back
to the CPU decoder (e.g. the hardware-decode benchmark arms). Routing note:
hardware decoders are only attempted when no fused output is requested
(`allow_hw = output_fmt == native_fmt`); a fused `Rgb`/`Nv12` preference
selects the CPU decoder even when a hardware backend is available. nvJPEG,
whose fixed output is interleaved RGB, engages under a `native` request and
reconfigures the destination to `Rgb` itself.

### nvJPEG Backend Module Map (`jpeg/nvjpeg/`, Linux + `nvjpeg` feature)

| Module        | Purpose                                                |
|---------------|--------------------------------------------------------|
| `mod.rs`      | `NvJpegProbe` lifecycle, persistent `NvJpegContext` (handle/state/stream), `try_decode()` tri-state, decode-into-PBO-device-pointer (RGB), circuit breaker |
| `loader.rs`   | `dlopen` of `libnvjpeg.so.12` (explicit CUDA paths; requires `nvjpeg*` symbols to reject the libjpeg-turbo decoy), `OnceLock` table, `EDGEFIRST_ENABLE_NVJPEG` opt-in (**default off**) |
| `ffi.rs`      | Hand-rolled `#[repr(C)]` `nvjpegImage_t`, enums/consts, `extern "C"` fn-pointer typedefs (verified vs the on-target `nvjpeg.h` 12.3.3) |

### V4L2 Backend Module Map (`jpeg/v4l2/`, Linux + `v4l2` feature)

| Module        | Purpose                                                |
|---------------|--------------------------------------------------------|
| `mod.rs`      | `V4l2Probe` lifecycle, persistent streaming session, `try_decode()` orchestration, DMABUF capture targets (zero-copy + scratch), JPEG metadata stripping, NEON YUV24→NV24 deinterleave |
| `device.rs`   | Capability-based probe: env overrides, enumerate `/dev/video*`, `QUERYCAP` + `ENUM_FMT` (require JPEG on OUTPUT) |
| `ioctl.rs`    | All raw `#[repr(C)]` UAPI structs, FourCC + buffer-type/memory constants, `nix` ioctl macro defs |
| `buffers.rs`  | RAII `Mmap` wrapper for the persistent OUTPUT (coded) buffer |
| `format.rs`   | `classify()` the driver-chosen CAPTURE FourCC → `CapKind` (`Nv12` / `Grey` / 4:4:4-packed) |

## Key Design Decisions

### Native-Format Output

The decoder writes the image's native pixel format and configures the
destination tensor to match (`Tensor::configure_image(w, h, format)`), within
the tensor's existing allocation:

- **JPEG**, 3-component colour → the NV format matching the source's chroma
  sampling, so nothing is resampled: 4:2:0 → `Nv12`, 4:2:2 → `Nv16`, 4:4:4 →
  `Nv24` (Y plane + interleaved Cb/Cr in each case). Exotic subsamplings
  (4:1:1, 4:1:0, or Cb and Cr sampled differently from each other) are the only
  cases that resample, block-averaging down to `Nv12`.
- **JPEG**, 1-component → `Grey`.
- **PNG** → `Rgb` / `Rgba` / `Grey` per the source colorspace.

JPEG output is `u8` only — the NV formats and `Grey` are byte layouts — so a
non-`u8` destination is rejected with `CodecError::UnsupportedDtype`. The PNG
path supports the full set of tensor element types (see below).

No colour conversion, resize, or rotation happens here (unless the caller
opts into a fused output, below). Callers that need `Rgb`/`Rgba`/`Bgra`, a
resize, or EXIF orientation applied run `ImageProcessor::convert()` on the
native decode.

### Fused Decode Output (opt-in, pure CPU)

`ImageDecoder::set_output_format(Some(fmt))` requests a conversion **inside
the MCU write stage** of the **software** JPEG decoder, where the IDCT output
is still hot in cache — one CPU pass instead of decode-to-NV followed by a
full-image colour step. This is **not** a GPU hybrid path and is unrelated to
nvJPEG: whenever the resolved output differs from native, the V4L2 and nvJPEG
backends are bypassed so the CPU path can honour the preference.

`resolve_output_format()` honours the preference only where it is a pure win
and silently falls back to native otherwise, so callers can set it
unconditionally:

- **`Rgb`** — 4:4:4 colour JPEGs (every component at full resolution, so no
  chroma resampling is needed) and native 4:2:0 colour JPEGs (`mcu::is_native_420`:
  luma 2×2, both chroma 1×1 — the dominant real-world subsampling). For 4:4:4,
  standard 1×1 sampling converts each MCU through the `color.rs` YCbCr→RGB
  kernel as it is produced (no planar scratch); other 4:4:4 MCU sizes still
  convert a full MCU-row band via `write_rgb_rows`. The kernel is Q14 fixed
  point on the `SQRDMULH` rounding-doubling high-half multiply (baseline
  ASIMD, so the same kernel runs A53 → Apple Silicon), `vst3` interleaved
  stores, and a scalar path that reproduces the arithmetic bit-for-bit. For
  native 4:2:0, `write_rgb_rows_420` does a 2×2 **nearest-neighbour (box)**
  chroma upsample — not libjpeg's fancy/triangle filter — into two reused
  `McuScratch` rows, then calls the same `ycbcr_to_rgb_row` kernel (one
  upsample pass serves the 2 luma rows a chroma sample covers, since
  nearest-neighbour repeats it vertically too). Box upsampling is a
  deliberate speed tradeoff, measured at 44–50 dB PSNR / max pixel delta
  20–24 against PIL's fancy-upsampled reference on the COCO-family test
  fixtures (`examples/dump_rgb420.rs`) — comparable to the existing
  accurate-vs-`fast`-DCT accuracy cost (see BENCHMARKS.md). Other
  subsamplings (4:2:2, 4:1:1, ...) fall back to native NV output.
- **`Nv12`** — any colour JPEG. 4:2:0 passes through; 4:4:4 is 2×2
  box-averaged and 4:2:2 vertically averaged with NEON row kernels at the
  write stage.

Callers still run [`ImageProcessor::convert()`](../image) on the fused result
for model-input preprocessing (letterbox / resize / EXIF orientation / further
format work). With fused RGB that convert step is typically a pure resize —
the colour conversion already happened in the decode write stage. Measured
end-to-end (decode + 640×640 letterbox, COCO): ~1 ms/frame faster than NV24 +
convert on Cortex-A53, ~0.3–0.4 ms on A55/A76/A78AE. The fused RGB output
clears the destination's colorimetry; fused NV12 keeps JFIF BT.601 full-range
like any native NV output.

### EXIF Orientation Is Reported, Not Applied

`markers.rs` (JPEG APP1) and `zune-png` (`eXIf` chunk) surface the EXIF
orientation tag, which `exif.rs` maps to `(rotation_degrees, flip_horizontal)`
and returns in `ImageInfo`. The decoder **never** rotates or flips: it writes
the source's native, unrotated pixels and dimensions. The caller applies the
reported transform downstream (typically via the same `convert()` call that
does colour conversion and resize). This avoids a redundant in-place rotation
pass in the decode hot loop.

### Hardware Decode Tried Before CPU (Linux)

On Linux, `decode_jpeg_into` dispatches through a three-tier priority —
**nvJPEG → V4L2 → CPU**:

1. With the `nvjpeg` feature **and** `EDGEFIRST_ENABLE_NVJPEG` set (opt-in,
   off by default — see [nvJPEG GPU Backend](#nvjpeg-gpu-backend)), if a CUDA
   device + `libnvjpeg` are present **and** the destination is a CUDA-backed
   tensor (a PBO on Jetson), the nvJPEG GPU backend decodes interleaved RGB
   straight into the tensor's device pointer.
2. Otherwise, with the `v4l2` feature, the V4L2 hardware backend (native
   NV12/Grey/NV24) — see [V4L2 Hardware Backend](#v4l2-hardware-backend).
3. Otherwise the from-scratch CPU decoder.

Each tier returns `Ok(None)` (not applicable) or a `Fallback` error to cascade
to the next. On non-Linux targets, or with both features disabled, the seams
compile to nothing and the CPU decoder always runs.

### Standalone `ImageDecoder` Struct

The decoder is a standalone struct rather than being embedded in
`ImageProcessor` or stored in thread-local state. This gives callers explicit
ownership and composability — one decoder per pipeline stage, no hidden global
state. Its `JpegDecoderState` holds the reusable MCU scratch and (on Linux) the
lazily-probed nvJPEG and V4L2 backends — including nvJPEG's persistent
handle/state/stream — all amortised across frames.

```rust
let mut decoder = ImageDecoder::new();
// Scratch buffers and the hardware session amortize across calls
loop {
    let info = tensor.load_image(&mut decoder, &bytes)?;
}
```

### `ImageLoad` Extension Trait

The primary user-facing API is the `ImageLoad` trait, implemented for both
`Tensor<T>` (where `T: ImagePixel`) and `TensorDyn`. This keeps the tensor
types in `edgefirst-tensor` unaware of codec internals.

### `&[u8]` as the Hot Path

The decode pipeline takes `&[u8]` as input — the most common case (memory-
mapped files, network buffers, camera frames). `Read`-based wrappers buffer
into `ImageDecoder.input_buffer` before delegating to the `&[u8]` path.

### Strided Output

Decoders write row-by-row using the tensor's `effective_row_stride()`. This
supports tensors with GPU pitch alignment padding (e.g., 64-byte alignment
for Mali DMA-BUF import). The stride gap bytes are untouched.

```
Tensor buffer layout (1280×720 Grey, 64-byte aligned stride = 1280):
┌──────────────────────────────┬────┐
│ row 0: 1280×1 = 1280 bytes  │ 0  │  ← no padding (1280 % 64 == 0)
├──────────────────────────────┼────┤
│ row 1: 1280×1 = 1280 bytes  │ 0  │
├──────────────────────────────┼────┤
│ ...                          │    │
└──────────────────────────────┴────┘
```

For misaligned widths (e.g., 641 pixels Grey = 641 bytes, padded to 704):
```
┌────────────────────────┬──────────┐
│ row 0: 641 bytes       │ 63 pad   │  ← stride = 704
├────────────────────────┼──────────┤
│ row 1: 641 bytes       │ 63 pad   │
└────────────────────────┴──────────┘
```

### Works Best with `ImageProcessor::create_image()`

While `ImageLoad` works with any `Tensor<T>` or `TensorDyn`, optimal
performance requires tensors allocated by `ImageProcessor::create_image()`:

- **DMA-BUF backing**: zero-copy path to GPU for `convert()`, and the V4L2
  zero-copy decode path
- **PBO backing**: when GL is the active transfer path
- **GPU pitch alignment**: row stride padded for Mali DMA-BUF import
- **CPU access declaration**: decode targets are CPU-written — allocate
  them with `CpuAccess::Write` (the decode loop maps via `map_write()`,
  which selects a write-oriented mapping / dma-buf sync direction);
  `ReadWrite` also works but declares reads the decoder never performs

Free-standing `Tensor::new()` or `Tensor::image()` works but:
- Cannot produce PBO tensors (requires GL context)
- May not have GPU-aligned pitch (works, but `convert()` may use CPU path)
- Is never eligible for V4L2 zero-copy (which requires a DMA-backed tensor)

The decoded tensor is the **source** stage of the batched-preprocessing
pipeline: `convert()` then renders it (resize / letterbox / format) into a
tile of the batched `[N, …]` model input (packed `NHWC` or planar `NCHW`). A
decode target reused across frames (a source pool) re-imports correctly because
`configure_image()` invalidates its cached EGLImage entry — a stable
`BufferIdentity` with changed geometry must not return the previous frame's GPU
import. See
[project `ARCHITECTURE.md` § Batched Preprocessing](https://github.com/EdgeFirstAI/hal/blob/main/ARCHITECTURE.md#batched-preprocessing).

### Tensor Dimensions After Decode

When a smaller image (e.g., 640×480) is decoded into a larger tensor
(e.g., 1920×1080), the decoder reconfigures the tensor's logical shape to the
decoded dimensions within the same physical allocation. `ImageInfo` reports the
decoded dimensions. Callers use `Crop` with `ImageProcessor::convert()` to
process the decoded region and apply any reported EXIF orientation:

```rust
let info = tensor.load_image(&mut decoder, &bytes)?;
let rot = Rotation::from_degrees_clockwise(info.rotation_degrees as usize);
let flip = if info.flip_horizontal { Flip::Horizontal } else { Flip::None };
// The decode reconfigured the tensor's logical shape to the decoded
// dimensions, so a whole-source crop covers exactly the decoded region.
processor.convert(&tensor, &mut dst, rot, flip, Crop::new())?;
```

## Decode Pipeline

### JPEG Decode Flow

The custom baseline JPEG decoder processes images through these stages:

1. **Marker parsing** (`markers.rs`): parse SOF0, DQT, DHT, DRI, SOS, APP1
   segments. Quantisation tables are built from DQT; Huffman LUTs are taken
   from `JpegDecoderState`'s `HuffmanCache` when the DHT payload matches the
   previous frame (moved, not cloned), otherwise rebuilt. EXIF orientation is
   read and reported only.
2. **Native format + capacity** (`mod.rs`): derive the native format
   (`native_format()`: 1 component → `Grey`; 3 components → `Nv24`/`Nv16`/`Nv12`
   from the chroma-to-luma sampling ratio) and reconfigure the destination
   tensor via `configure_image()`, which errors with `InsufficientCapacity` if
   the image exceeds the tensor's allocation.
3. **Hardware attempt** (Linux + `v4l2`): try `V4l2Probe::try_decode()`. On
   success it returns the `ImageInfo`; on `Fallback` (no device / unsupported
   capture format / transient hardware failure) the CPU path below runs.
4. **MCU decode loop** (`mcu.rs`): for each MCU row:
   a. **Huffman decode** (`huffman.rs`): tier-sized lookahead LUT decodes DC/AC
      coefficients into a quantised `i16[64]` block (libjpeg-turbo `JCOEF`
      model — the entropy loop is multiply-free).
   b. **IDCT** (`idct/`): two-pass Loeffler 8×8 IDCT with a DC-only fast path
      dequantises on load and converts frequency coefficients → spatial
      component samples.
   c. **Write** (`mcu.rs`): `write_grey_rows` copies the luma plane for
      `Grey`; `write_nv12_rows` writes the Y plane and the interleaved Cb/Cr
      plane, downsampling chroma to 4:2:0 via `avg_block` (passthrough for an
      already-4:2:0 source, block-average for 4:2:2 / 4:4:4 / 4:4:0). Standard
      4:4:4 (1×1 sampling) writes NV24 UV per MCU (`write_nv24_uv_block`) so
      chroma never lands in the planar scratch. With fused `Rgb` on that same
      1×1 path, `ycbcr_to_rgb_blocks` converts paired 8×8 MCUs through
      `color.rs` (16-pixel `vst3q_u8` NEON, one dispatch per pair) instead of
      a second full-row pass. Other fused RGB on a 4:4:4-equivalent source
      still uses `write_rgb_rows`; fused RGB on a native 4:2:0 source uses
      `write_rgb_rows_420` (box chroma upsample, see [Fused Decode
      Output](#fused-decode-output-opt-in-pure-cpu)). All write at the
      tensor's `effective_row_stride()`.
5. **Return** `ImageInfo` with decoded dimensions, native format, row stride,
   and the reported EXIF orientation.

**Untrusted input.** Everything above runs on attacker-controlled bytes, so
three places take deliberate care:

- **DHT counts** (`huffman.rs`) are checked against Kraft's inequality before
  the lookahead table is filled. The fill indexes by code value shifted up to
  the table width, so an oversubscribed level — three 1-bit codes, say — walks
  off the end of the allocation; the segment is rejected instead.
- **A truncated scan** (`bitstream.rs`, `mcu.rs`) cannot be caught block by
  block: the buffer zero-pads past end-of-data, and zero is both a valid DC
  code and a valid EOB, so a scan that simply stops decodes as well-formed
  empty blocks. `BitStream::overran()` records that a consumer was handed bits
  the file never held, and the scan loop turns that into `InvalidData`.
- **Coefficient × quantiser** can overflow the 16-bit lanes the IDCT kernels
  dequantise into (JPEG syntax allows 1023 × 255). Every kernel including the
  scalar reference absorbs that as defined wrapping/saturating arithmetic —
  see `IdctFn` for what is and is not guaranteed about the resulting pixels.

**Key optimisations:**
- `JpegDecoderState` persists across frames — `McuScratch` buffers grow to the
  high-water mark and are reused. Huffman LUTs live in `HuffmanCache` and are
  moved into the parse result for the decode, then restored: when the DHT
  payload is unchanged (typical of a camera stream), the second frame onwards
  allocates nothing for table construction. After the first decode at a given
  resolution, the JPEG decoder performs zero heap allocations (per-scan Huffman
  table pointers and DC predictors live in fixed-size stack arrays).
- Combined **fast-AC LUT** (stb_image model): where a Huffman code *and* its
  magnitude bits both fit in the tier's lookahead window, a single table hit
  yields the sign-extended coefficient, zero-run, and total bits to consume —
  one peek + one consume per coefficient on the fast path, for DC and AC
  alike. Size-0 symbols are baked in rather than treated as misses: DC size 0
  (predictor unchanged) is a plain value-0 hit, ZRL is a run-15/value-0 hit
  (the written zero lands in a slot the clear discipline already guarantees is
  zero), and **EOB carries a run-0xFF sentinel** that pushes `k` past 63 so the
  block decoder's existing bounds branch resolves end-of-block — the common
  block exit costs no extra hot-path compare and never touches the two-step
  LUT. Only long codes (> `fast_bits`) still fall back.
- **Paired-coefficient probe** (`HuffmanTable::pair_ac`, `NeonTier::High`
  only): decodes up to two coefficients per table hit when both fit the
  lookahead window (~54% of COCO AC coefficients at 10 bits). Singles bake as
  phantom pairs (value-0 second write into the next undecoded slot, `k`
  advanced by `(val2 != 0)` — branchless) so the probe keeps the single
  table's ~94% hit rate. A pair whose first coefficient lands on zigzag 63
  rewinds to a single via the `fast_ac` probe: the entry's second symbol was
  built from the next block's bits. Per-tier because in-order A53/A55 retire
  the phantom overhead with no OoO slack to hide it (measured −1-3%), while
  the A76 gains ~2.7%; the instantiation is selected once per decode as a fn
  pointer — keeping a second live call site in the MCU loop cost ~7% on the
  A53.
- **Register-resident bit cursor** (`BitCursor`): the block decoder copies the
  `BitStream` state into a function-local cursor and commits it back on every
  exit. `&mut BitStream` is caller-visible memory, and `perf annotate` on
  Cortex-A53 showed `avail` written back to the struct on every decoded
  coefficient; the cursor keeps the whole bit buffer in registers
  (libjpeg-turbo `BITREAD_LOAD_STATE`/`SAVE_STATE` discipline).
  `decode_block` is `#[inline(never)]` to keep its register-allocation problem
  out of the MCU loop's already-spilling frame — letting it inline cost ~20%
  of A53 decode time (measured 7.68 → 9.23 ms p50 on imx8mp COCO).
- **Entropy-derived IDCT sparsity** (`BlockInfo::last_k`): the highest zigzag
  index written is a free by-product of the scan order, and because zigzag
  fills the top-left corner first, a small `last_k` proves whole regions zero.
  The NEON kernel picks a tier from it without re-deriving sparsity from the
  coefficient vectors (`last_k ≤ 1` DC-broadcast left half, `≤ 9` sparse
  butterfly, `≤ 13` full-left/zero-right, else the value-detection kernel),
  loading only the rows that participate — 64-bit `vld1_s16` loads on the
  proven-sparse tiers, which matches the Cortex-A53's 64-bit load path. The
  de-zigzag table is padded (`ZIGZAG_EXT[320]`) so the natural-index load can
  issue before the bounds branch instead of address-stalling the coefficient
  store.
- Runtime **`NeonTier`** / **`IntelTier`** (`cpu.rs`): one binary picks Huffman
  lookahead width, entropy prefetch distance, and SIMD kernels per
  micro-architecture / ISA. AArch64 uses HWCAP `dotprod`/`i8mm` plus a Linux
  sysfs MIDR probe that promotes big out-of-order cores (A76/A78AE/X1…) that
  HWCAP alone cannot distinguish from the A55. x86_64 probes `avx2` >
  `sse4.1` > `sse2`. Override with `EDGEFIRST_CODEC_FORCE_NEON=…` or
  `EDGEFIRST_CODEC_FORCE_INTEL=scalar|sse2|sse41|avx2` for A/B.
- Dequantisation lives inside the IDCT kernels (turbo `JCOEF` model): the
  entropy decoder emits raw quantised `i16` coefficients (128-byte block clear,
  no per-coefficient multiply) and the SIMD kernels dequantise 8 lanes at a
  time on load.
- DC-only IDCT fast path: when all 63 AC coefficients are zero, the IDCT reduces
  to a constant fill (single multiply + shift). The NEON 8×8 kernel further
  special-cases each 4×8 half whose rows 1–7 are zero (turbo islow case 2).
- On AArch64 the MCU loop binds the NEON IDCT as a fn-item (direct `bl` per
  block) rather than an `IdctFn` pointer; other ISAs still select a pointer
  once at decode start.

### NV Output Path

For a colour JPEG the decoder produces a semi-planar NV format directly,
skipping any YCbCr→RGB conversion:

- The Y plane is copied from the IDCT luma output at the tensor's row stride
  (or written directly during IDCT for full MCU-row bands).
- The Cb/Cr components are interleaved pair-wise into the UV plane that follows
  the Y plane, at the source's own chroma resolution. Standard 4:4:4 (1×1)
  interleaves each 8×8 chroma block as it is produced (`write_nv24_uv_block`);
  4:2:2 and non-1×1 4:4:4 still gather a planar MCU-row band then `vst2`
  a full row. `avg_block` downsamples first only for the exotic subsamplings
  that have no matching NV format.

These NV formats are the codec's only colour output. The chroma carries the
source's JFIF (BT.601 full-range) colorimetry; mapping that to RGB is the
downstream `convert()` step's responsibility, not the codec's.

### Chroma Subsampling Support

The Cb/Cr components are decoded at their native sampling and downsampled to
4:2:0 when written into `Nv12`:

| Source sampling | Description              | Native output            |
|-----------------|--------------------------|--------------------------|
| 4:2:0           | Horizontal + vertical 2× | `Nv12` (passthrough, no resample) |
| 4:2:2           | Horizontal 2×            | `Nv16` (passthrough, no resample) |
| 4:4:4           | No subsampling           | `Nv24` (passthrough, no resample) |
| 4:4:0, 4:1:1, 4:1:0, mismatched Cb/Cr | Everything else | `Nv12` via `avg_block` block-average |
| Greyscale       | Single component         | No chroma (`Grey` output) |

The passthrough rows are the point: `native_format()` reads the chroma-to-luma
sampling ratio and picks the NV format that already matches, so the common
cases never touch `avg_block` at all. It runs only for the last row, where no
NV format fits the source.

### IDCT and Write-Path SIMD Kernels

SIMD is used for IDCT, fused YCbCr→RGB (`color.rs`), UV interleave, and chroma
downsample (`mcu.rs`). Native NV output still skips RGB for the usual decode
path; fused RGB and exotic NV12 averaging take the color/downsample kernels.

**NEON (AArch64)** — gated by `NeonTier` +
`std::arch::is_aarch64_feature_detected!("neon")`:

| Kernel   | Strategy                                          | Throughput              |
|----------|---------------------------------------------------|-------------------------|
| **IDCT** | 16-bit coefficients end-to-end (libjpeg-turbo `jidctint-neon` model): 8-lane `vmulq_s16` dequant, Loeffler butterfly via `vmull_s16`/`vmlal_s16` 16×16→32 multiplies, `vrshrn` descale, register-resident `int16x8` workspace (no memory round-trip), `trn`-based 16-bit and 8-bit 8×8 transposes. Sparse shortcuts skip the butterfly for all-zero bottom rows (pass 1) and all-zero right columns (both passes); a further per-half DC-only path (rows 1–7 of a 4×8 half all zero) broadcasts `saturate(dequant(row0) << PASS1_BITS)`, matching turbo's islow special case 2 — the remaining in-order gap after whole-block DC-only is taken in the MCU loop. DC-only fills 8 bytes via `vdup`/`vst1` | 8 columns per pass-1 sweep, 8 rows per pass-2 sweep |

The butterfly constants are held in two `int16x8` registers and addressed by
lane (`vmull_laneq_s16`), not splatted per multiply: the butterfly is inlined
four times, so `vdup_n_s16(FIX_x)` at each use site cost 64 `dup`s per block.
The `black_box` on the constant pointers in `idct_8x8_neon_inner` is what makes
that stick — see the comment there.
| **Color / UV** | Q14 `SQRDMULH` YCbCr→RGB; `vst2` UV interleave; NEON 2×2 / 1×2 chroma downsample | 8–16 samples/iter |

**x86-64** — gated by `IntelTier` (`EDGEFIRST_CODEC_FORCE_INTEL` for A/B):

| Kernel   | Tier   | Strategy                                          | Throughput              |
|----------|--------|---------------------------------------------------|-------------------------|
| **IDCT** | SSE2 / SSE4.1 / AVX2 | 16-bit-lane `islow` Loeffler: `pmullw` dequant folded into the load, every butterfly term a `pmaddwd` | whole 8-wide row per register |
| **Color** | SSE4.1 / AVX2 | Bit-exact Q14 `qrdmulh` matching the scalar/NEON model | 16 pixels/iter |
| **UV** | SSE2+ | Unpack interleave; SSE2 2×2 / 1×2 truncating downsample | 8–16 samples/iter |

All three x86 tiers share one IDCT kernel. In 16-bit lanes a 128-bit register
already holds a full row of eight coefficients, and `pmaddwd` needs only SSE2,
so SSE4.1 has nothing to add and a 256-bit *single-block* kernel would only buy
lane-crossing shuffles; AVX2 gets the same code VEX-encoded. Genuine 256-bit
width would require processing two blocks side by side. The constant pairs are
an exact integer regrouping of the scalar butterfly, so all tiers are bit-exact
with `idct/scalar.rs`.

### PNG Decode Flow

1. Parse PNG headers via `zune-png` → dimensions, colorspace, bit depth, and the
   `eXIf` orientation tag (reported only).
2. Map the colorspace to the native `PixelFormat` (Luma/LumaA → `Grey`, RGB →
   `Rgb`, RGBA → `Rgba`) and reconfigure the tensor via `configure_image()`.
3. Choose a decode strategy based on the target type and source bit depth:
   - **u8/i8 targets**: `decode_into(&mut [u8])` — fast u8 path with optional
     XOR for i8 and LumaA→Grey alpha stripping.
   - **u16/i16/f32 targets**: `decode()` → `DecodingResult`, preserving native
     16-bit data from 16-bit PNGs.
4. Row-copy from decoded data → tensor buffer at stride offsets, with pixel-type
   conversion via `from_u8()` / `from_u16()` per source depth.
5. Return `ImageInfo` with decoded dimensions and reported EXIF orientation.

### Format Auto-Detection

The decoder inspects magic bytes:
- `FF D8 FF` → JPEG
- `89 50 4E 47` → PNG
- Otherwise → `CodecError::InvalidData`

## V4L2 Hardware Backend

On Linux, the `v4l2` feature (default-on) adds a mem2mem hardware JPEG-decode
backend in `jpeg/v4l2/`. It drives any device that exposes a JPEG decoder
through the standard V4L2 M2M API; the lead target is i.MX `mxc-jpeg`
(`/dev/video11`) but nothing about the node, driver, or output format is
hardcoded.

### Discovery (capability-based, portable)

`V4l2Probe` is probed lazily on the first JPEG decode and at most once per
`ImageDecoder`. `device.rs`:

1. Honours `EDGEFIRST_DISABLE_V4L2=1` (skip → CPU) and
   `EDGEFIRST_CODEC_V4L2_DEVICE=<path>` (probe only that node); otherwise
   enumerates `/dev/video*`.
2. `VIDIOC_QUERYCAP` requires `V4L2_CAP_STREAMING` and a multi-planar M2M
   capability. (Single-planar-only M2M devices currently fall back to CPU.)
3. `VIDIOC_ENUM_FMT` on the OUTPUT (coded) queue must advertise
   `V4L2_PIX_FMT_JPEG` — this, not the device name, is the "is this a JPEG
   decoder" test. Nodes without JPEG are skipped (so camera, HEVC/H264, and
   ISI-M2M nodes on the same board are correctly rejected).

### Persistent streaming session (DMABUF-only CAPTURE)

The CAPTURE queue always uses `V4L2_MEMORY_DMABUF`, where `REQBUFS` is
allocation-free vb2 bookkeeping (measured 5-8 µs on i.MX95) — the buffers are
ours, imported at `QBUF` time. The OUTPUT (coded) buffer is one MMAP buffer
allocated once with headroom (`OUT_SIZE_FLOOR`, 2 MiB) that survives geometry
changes. Hardware decode therefore requires DMA buffer allocation (dma_heap),
in line with the HAL's DMABUF-centric design; platforms without it fail fast
to the CPU decoder. There is no cheap MMAP alternative: `S_FMT` returns
`EBUSY` while MMAP buffers exist on a queue, so an MMAP CAPTURE would re-pay a
~110 ms kernel buffer reallocation on every geometry change.

`decode()` picks one of three tiers per frame:

- **reuse** — same geometry, native format, and (for zero-copy) destination
  pitch: stage the JPEG, `QBUF`, done.
- **reconfigure** — geometry changed: `STREAMOFF(CAPTURE)` → `REQBUFS(0)` →
  stage + `QBUF` the JPEG (OUTPUT keeps streaming) → `SOURCE_CHANGE` wait →
  `G_FMT` (with a stale-geometry guard) → renegotiate → `REQBUFS(1, DMABUF)`
  → `STREAMON`. ~1 ms of ioctls; any failure recovers with a full rebuild.
- **rebuild** — first frame, OUTPUT overflow, or recovery: full teardown and
  setup of both queues.

Per-image lifecycle: stage the JPEG into the OUTPUT buffer (see metadata
stripping below) → bounded `DQEVENT` drain after `SOURCE_CHANGE` (an unbounded
drain hangs — capped at `MAX_EVENTS`) → `G_FMT(CAPTURE)` for the driver-chosen
format and MCU-rounded dimensions → queue the CAPTURE dmabuf → poll/`DQBUF` →
copy out if needed (cropped to the logical image) → recycle the OUTPUT buffer.

### JPEG staging strips metadata segments

The i.MX `mxc-jpeg` bitstream parser does not skip APPn payloads by their
length field: an APP13 (Photoshop IRB) carrying an embedded thumbnail JPEG
(nested SOI/SOF/SOS markers) wedges the hardware until the decode timeout.
`stage_jpeg()` drops APP1–APP13/APP15 and COM segments during the
OUTPUT-buffer copy that happens anyway, keeping JFIF APP0 and Adobe APP14
(whose transform flag changes component-colour interpretation). A stream that
doesn't parse as plain marker segments is staged verbatim.

### Capture targets

`format.rs::classify()` maps the driver's CAPTURE FourCC to a `CapKind`
(`NV12`/`NV12M` → `Nv12`, `GREY` → `Grey`, `YUV24` → 4:4:4-packed). The
CAPTURE queue is always configured single-plane — the driver's *default*
`NV12M` 2-plane layout does **not** compose into one buffer, so contiguous
`NV12` is renegotiated via `S_FMT`. Two decode targets, tried in order:

- **Zero-copy (`CaptureTarget::DstDma`):** when the destination is a DMA
  tensor with MCU(16)-aligned dimensions and the driver accepts a single-plane
  contiguous CAPTURE (`V4L2_PIX_FMT_NV12`/`GREY`) at the tensor pitch, the
  tensor's dmabuf fd is imported as the CAPTURE buffer and the hardware
  decodes straight in — no copy.
- **Scratch (`CaptureTarget::Scratch`):** the hardware decodes into a
  persistent codec-owned DMA scratch buffer (grown geometrically to the
  high-water mark, kept across stream resets), then the planes are copied out
  cropped: `Nv12`/`Grey` row copies, or the NEON-accelerated YUV24→NV24
  deinterleave for 4:4:4 JPEGs (`vld3q`/`vst2q`, 16 px/iteration).

A 4:4:4 JPEG requested as `Nv12` (and only then) attempts an `S_FMT(NV12)`
negotiation so the hardware downconverts; an `Nv24` request keeps the
driver's YUV3 and deinterleaves, preserving chroma resolution and the
native-format contract with the CPU decoder.

### Fallback & recovery

`try_decode()` returns `Ok(Some(info))` on a hardware decode, `Ok(None)` when no
device is available, or `Err(V4l2Decode::Fallback(reason))` on a transient
failure (the device is reset and the caller transparently runs the CPU decoder
on the same tensor). `Fatal` is reserved for deterministic input errors. After
`MAX_CONSECUTIVE_FAILURES` (8) the device is demoted to CPU for the rest of the
session (circuit breaker).

### ABI note

The raw UAPI structs in `ioctl.rs` must match the kernel's `sizeof`, which a
compile-time `size_of` assert checks against our arithmetic, **not** the
kernel's. `v4l2_format` is **208 bytes** (its union contains `v4l2_window`,
whose pointers force 8-byte alignment and 4 bytes of padding after `type`); a
wrong size yields the wrong ioctl request number → `ENOTTY` → a silent CPU
fallback that makes parity tests pass trivially. On-target `strace` is the only
reliable verification of the raw ioctl ABI — see `TESTING.md`.

## nvJPEG GPU Backend

`jpeg/nvjpeg/` offloads JPEG decode to the CUDA nvJPEG library on NVIDIA
platforms (lead target: Jetson Orin). When enabled it is preferred ahead of
V4L2/CPU, but it is **opt-in and off by default** (`EDGEFIRST_ENABLE_NVJPEG`,
see [Discovery](#discovery-dlopen-capability-based)) so it never silently
contends with CUDA inference. It is entirely `dlopen`-based — no link-time CUDA
dependency, so one binary runs on Jetson (nvJPEG), i.MX (V4L2), and a laptop
(CPU).

### Discovery (dlopen, capability-based)

`loader.rs` opens `libnvjpeg.so.12`, trying explicit CUDA install paths first
(the soname is not on JetPack's default loader path) and **requiring** the
`nvjpeg*` symbols to resolve — this rejects the libjpeg-turbo *decoy*
(`/usr/lib/.../nvidia/libnvjpeg.so`) that shares the name but exports `jpeg_*`.
nvJPEG is **opt-in and off by default** — it decodes on the same GPU as CUDA
inference (TensorRT etc.), so sharing the device can cost a concurrent inference
engine more than the decode speedup returns. Set `EDGEFIRST_ENABLE_NVJPEG=1`
(`true`/`yes`) to enable it on decode-bound workloads or where no concurrent GPU
compute runs. (V4L2 stays opt-out via `EDGEFIRST_DISABLE_V4L2`: it is a separate
hardware block and does not contend with CUDA.) The `NvJpegProbe` (on
`JpegDecoderState`) is probed once per `ImageDecoder`; a ready `NvJpegContext`
persists the nvJPEG handle, a reusable decode state (keeping nvJPEG's internal
device scratch hot), and **one CUDA stream per decoder** so concurrent decode
workers do not serialise on the default stream.

### RGB output — a deliberate exception to the native-format contract

Unlike the CPU/V4L2 paths (native NV12/Grey/NV24, never colour-converted), the
nvJPEG path emits packed **`Rgb`** via `NVJPEG_OUTPUT_RGBI`. nvJPEG performs the
YCbCr→RGB on the GPU at near-zero marginal cost, the result is GPU-resident, and
`ImageProcessor::convert()` accepts an `Rgb` source directly (removing the
downstream NV12→RGB step). `Rgbi` maps onto the existing `PixelFormat::Rgb` — no
new enum variant. The backend reconfigures the destination NV12→Rgb and, on any
post-reconfigure failure, restores the native format so the V4L2/CPU
fall-through writes into a correctly-shaped tensor. (The L4T nvJPEG 12.3.3 build
has no `NVJPEG_OUTPUT_NV12`, so RGB is also the only single-call option here.)

### PBO + `cuda_map` zero-copy chain

The destination is a `TensorMemory::Pbo` tensor — what
`ImageProcessor::create_image` yields on Jetson (no dma-heap) — whose CUDA
GL-buffer registration is mapped to a device pointer via `Tensor::cuda_map()`.
nvJPEG decodes a single interleaved RGB plane into that pointer (honouring any
batch `plane_offset`), the stream is synchronised, and the PBO is unmapped so
`convert()` can sample it. The nvJPEG row pitch is read back from
`Tensor::effective_row_stride()` *after* the `Rgb` reconfigure rather than
assumed — `configure_image` keeps a PBO tight (`width*3`) but rounds a
CUDA-backed DMA destination up to a 64-byte-aligned pitch, and writing at the
wrong pitch would shear the rows `convert()` then samples. Because
`cuda_map`/unmap on a GL-buffer route to the GL worker thread that owns the PBO,
each frame pays **two GL-thread round-trips** (map + unmap). The capacity of the
packed RGB write is bounds-checked against the mapping length explicitly —
`configure_image` does not guard a packed format on GL (PBO) memory — and an
NV12-sized buffer too small for 3 B/px RGB falls back to V4L2/CPU rather than
erroring.

### Orin: GPU_HYBRID, not the NVJPG ASIC

On Orin the dedicated-hardware backend (`NVJPEG_BACKEND_HARDWARE`) is
unsupported (`nvjpegCreateEx` returns status 7), so the context uses
`NVJPEG_BACKEND_DEFAULT` (GPU-hybrid: CUDA-core Huffman). The throughput win is
**GPU-resident output + freed CPU**, not raw decode speed (measured RGBI
decode-only: 720p 4:4:4 ~7 ms; 4K ~18–74 ms, entropy-dependent). Whether it
beats CPU-decode-NV12 + GPU-upload is a per-resolution call, gated on the
on-target `codec/jpeg/nvjpeg/*` benchmarks.

### Overlap model

Per-decoder CUDA streams let concurrent decode workers interleave on the GPU;
the primary throughput mechanism is pipeline overlap via the consumer's ring
buffer (decode worker N+1 runs while worker N's frame is converted).
Intra-frame CUDA/GL overlap on the single Orin GPU is unproven and must be
trace-verified. To keep the per-frame `cuda_map` round-trips off the `convert()`
hot path, the consumer should own the decode-source pool on a GL thread distinct
from the convert GL thread (a consumer-side concern).

### Fallback & recovery

`try_decode()` returns `Ok(None)` when nvJPEG is unavailable or the destination
is not CUDA-backed (untouched tensor → V4L2/CPU), `Err(Fallback)` on a transient
nvJPEG error (the native format is restored and the CPU decoder — which handles
progressive/non-baseline JPEGs nvJPEG rejects — runs on the same tensor), and
`Fatal` for deterministic tensor errors. After `MAX_CONSECUTIVE_FAILURES` (8)
the backend is demoted for the rest of the session (circuit breaker).

A future increment (not built) could defer the `cudaStreamSynchronize`/unmap and
hand a CUDA event to `convert()`, gated on a measured demonstration that
per-decode sync stalls the worker.

## Tracing Spans

`ImageDecoder::decode_into` (and the trait-method `Tensor::load_image`) emits a
[`tracing::trace_span!`] tree describing each phase of the JPEG/PNG decode.
Spans are captured by
[`edgefirst_tensor::trace::start_tracing`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/src/trace.rs)
into Chrome JSON for Perfetto. The cost when no subscriber is active is a single
relaxed atomic load per call site.

### Naming convention

Span names follow `<crate>.<function>[.<operation>]`:

- **`<crate>.<function>`** — top-level span: the public function the user
  invoked. The codec exposes format-specific entry points (`codec.decode_jpeg`,
  `codec.decode_png`) selected automatically from the magic bytes.
- **`<crate>.<function>.<operation>`** — meaningful internal work
  (`codec.decode_jpeg.parse_markers`, `codec.decode_jpeg.mcu_loop`).

### Span tree

```text
codec.decode_jpeg                                       [user-facing fn]
│ fields: dtype = "u8", n_bytes
│
├── codec.decode_jpeg.parse_markers                     ← parse SOF0/DQT/DHT/DRI/SOS/APP1, read EXIF orientation
├── codec.decode_jpeg.nvjpeg                            ← nvJPEG GPU decode into the PBO device pointer (RGB)
│   │ fields: w, h, n_bytes, target = "rgbi"
│   ├── codec.decode_jpeg.nvjpeg_map                    ← cuda_map(): GL-thread round-trip → device pointer
│   ├── codec.decode_jpeg.nvjpeg_submit                 ← nvjpegDecode enqueue on the per-decoder stream
│   ├── codec.decode_jpeg.nvjpeg_sync                   ← cudaStreamSynchronize (GPU decode time)
│   └── codec.decode_jpeg.nvjpeg_unmap                  ← drop(CudaMap): GL-thread round-trip, PBO freed for convert()
├── codec.decode_jpeg.v4l2_rebuild                      ← full V4L2 session setup (first frame, OUTPUT overflow, recovery)
│   fields: w, h
├── codec.decode_jpeg.v4l2_reconfigure                  ← CAPTURE-only retarget on geometry change (~1 ms, DMABUF)
│   fields: w, h
├── codec.decode_jpeg.v4l2_collect                      ← per-frame QBUF → poll → DQBUF (hardware decode wait)
│   │ field: target = "dst_dma" | "scratch"
│   └── codec.decode_jpeg.v4l2_copy_out                 ← scratch → tensor write (absent on the zero-copy target)
│       field: kind = "nv12" | "grey" | "yuv24_to_nv24"
└── codec.decode_jpeg.mcu_loop                          ← Huffman + IDCT + write stage (CPU path only)
    ├── codec.decode_jpeg.mcu_row.huffman_idct          ← per MCU-row entropy + IDCT (4:4:4 1×1 also writes NV24 UV / fused RGB here)
    ├── codec.decode_jpeg.write_nv12 / write_nv16 / write_nv24 / write_grey / write_rgb
        (write_nv24 / write_rgb absent for standard 1×1 4:4:4 — those writes are folded into huffman_idct)

codec.decode_png                                        [user-facing fn]
│ fields: dtype, n_bytes
│
└── codec.decode_png.zune_decode                        ← delegate to zune-png
    field: path = "u8" | "native_u16"
```

The `nvjpeg*`, `v4l2_*`, and `mcu_loop` spans are mutually exclusive — exactly
one backend handles a given JPEG. An nvJPEG frame shows the `nvjpeg` subtree
(and no `v4l2_*`/`mcu_loop`); a V4L2 frame shows the `v4l2_*` spans; a CPU frame
(no GPU/device, non-Linux, or a hardware fallback) shows only `mcu_loop`. A
steady-state V4L2 frame shows only `v4l2_collect` (the reuse path);
`v4l2_reconfigure` appears on geometry changes and `v4l2_rebuild` on the first
frame or error recovery. For nvJPEG, `nvjpeg_sync` is the GPU decode time and
`nvjpeg_map`/`nvjpeg_unmap` isolate the per-frame GL-thread round-trip cost.

### What each span measures

| Span                              | What is happening inside | Reference equivalent |
|-----------------------------------|--------------------------|----------------------|
| `codec.decode_jpeg`               | Full JPEG decode: marker parsing then either the V4L2 hardware decode or the CPU MCU loop, writing the native NV format or `Grey`. | Baseline JPEG decode per ITU T.81. |
| `codec.decode_jpeg.parse_markers` | Walk the JPEG byte stream once: parse SOF0, DQT, DHT, DRI, SOS, and APP1 (EXIF) segments; build Huffman LUTs and quant tables; read the EXIF orientation tag. | Equivalent to libjpeg's `jpeg_read_header` + DHT/DQT table builds. |
| `codec.decode_jpeg.nvjpeg`        | The nvJPEG GPU decode: map the PBO, `nvjpegGetImageInfo`, decode interleaved `Rgb` into the device pointer, synchronise, unmap. `target = "rgbi"`. Absent when V4L2/CPU decodes. | nvJPEG single-image GPU decode (`nvjpegDecode`). |
| `codec.decode_jpeg.nvjpeg_map` / `nvjpeg_unmap` | The `cuda_map()` / drop round-trips to the PBO's owning GL worker thread (`cudaGraphicsMapResources` / `Unmap`). Isolates the per-frame GL-thread interop cost. | CUDA-GL interop buffer map/unmap. |
| `codec.decode_jpeg.nvjpeg_submit` / `nvjpeg_sync` | `nvjpegDecode` enqueue on the per-decoder CUDA stream, then `cudaStreamSynchronize`. `nvjpeg_sync`'s duration is the GPU decode time (resolution/entropy dependent). | Async CUDA decode + stream sync. |
| `codec.decode_jpeg.mcu_loop`      | The CPU core loop: per MCU row, Huffman-decode (quantised `i16`) → dequantising two-pass Loeffler IDCT (scalar / NEON / x86 islow) → native `Grey`/`Nv12`/`Nv16`/`Nv24` write (or the fused `Rgb`/`Nv12` write), strided into the tensor. Allocation-free after warmup. Absent when hardware decodes. Child spans: `mcu_row.huffman_idct`, `write_nv12` / `write_nv16` / `write_nv24` / `write_grey` / `write_rgb`. | Equivalent to libjpeg's `jpeg_read_scanlines` loop, IDCT handwritten and SIMD-dispatched. |
| `codec.decode_jpeg.v4l2_rebuild`  | Full V4L2 stream setup: OUTPUT buffer allocation + mmap + `STREAMON`, JPEG staging, `SOURCE_CHANGE` wait, CAPTURE format negotiation, `REQBUFS(DMABUF)`, `STREAMON`. Runs on the first frame, OUTPUT overflow, or error recovery. | A stateful V4L2 decoder session open + format negotiation. |
| `codec.decode_jpeg.v4l2_reconfigure` | The geometry-change hot path: `STREAMOFF`/`REQBUFS(0)` on CAPTURE only, JPEG staging + `QBUF` while OUTPUT keeps streaming, `SOURCE_CHANGE` wait, `G_FMT` stale-geometry guard, renegotiation, `REQBUFS(1, DMABUF)`, `STREAMON`. ~1 ms — DMABUF makes `REQBUFS` allocation-free, vs ~110 ms for an MMAP buffer reallocation. | The V4L2 stateful-decoder dynamic-resolution-change sequence. |
| `codec.decode_jpeg.v4l2_collect`  | The per-frame hardware decode: queue the CAPTURE dmabuf (`target` names where the hardware writes — the destination tensor for zero-copy, or the persistent scratch), poll for completion, dequeue both buffers. Dominated by the hardware decode itself. | A V4L2 M2M `QBUF` → `poll` → `DQBUF` cycle. |
| `codec.decode_jpeg.v4l2_copy_out` | Scratch → tensor write, cropped to the logical image: `nv12`/`grey` strided row copies, or the NEON-accelerated `yuv24_to_nv24` deinterleave for 4:4:4 captures. Absent on the zero-copy target. | A libyuv-style plane copy / deinterleave. |
| `codec.decode_png`                | Full PNG decode: header parse, native or 8-bit `zune-png` decode, native-format row write. | PNG decode per ISO/IEC 15948 (RFC 2083). |
| `codec.decode_png.zune_decode`    | The bulk of PNG cost: zlib inflate + PNG filter reversal inside [`zune-png`](https://docs.rs/zune-png). `path = "u8"` is the strided-output fast path; `path = "native_u16"` preserves 16-bit-per-channel PNGs for u16/i16/f32 tensor targets. | Equivalent to libpng's `png_read_image`. |

[`tracing::trace_span!`]: https://docs.rs/tracing/latest/tracing/macro.trace_span.html

## Supported Pixel Formats

| Output Format | JPEG | PNG  | Notes                              |
|---------------|------|------|------------------------------------|
| Nv12          | ✓    | —    | 4:2:0 colour JPEG (and the fallback for exotic subsamplings): Y + interleaved UV. Also the fused-output target for any colour JPEG (opt-in). |
| Nv16          | ✓    | —    | 4:2:2 colour JPEG: Y + interleaved UV at half horizontal chroma |
| Nv24          | ✓    | —    | 4:4:4 colour JPEG: Y + interleaved UV at full resolution |
| Grey          | ✓    | ✓    | Single luma component              |
| Rgb           | opt-in | ✓  | PNG native; JPEG via the fused 4:4:4 output (`set_output_format`) |
| Rgba          | —    | ✓    | Native PNG RGBA                    |

`Rgba`/`Bgra` from a JPEG, `Rgb` from non-4:4:4 JPEGs, and any resize/rotation
come from `ImageProcessor::convert()` applied to the native decode — they are
deliberately not codec responsibilities.

## Supported Source Features

The codec implements a **strict subset** of the JPEG and PNG specifications.
Inputs that fall outside the subset surface a typed
`CodecError::Unsupported(UnsupportedFeature)`. See the per-feature table in
[`README.md`](README.md#decoder-limitations) for the full matrix and the typed
error variant that each rejected case carries.

The codec does **not** transparently fall back to another *software* decoder for
unsupported inputs and does **not** transcode them. (The V4L2→CPU fallback is a
different mechanism: both decoders implement the same strict subset and produce
identical native output; the fallback only changes *where* the supported decode
runs.) The contract is "accept this strict subset; reject everything else with a
precise typed error."

## Data Type Support

JPEG decodes to `u8` only (the native NV formats and `Grey` are byte
layouts); a non-`u8` destination is
rejected with `CodecError::UnsupportedDtype`. PNG supports the full set:

| Type  | PNG (8-bit source)   | PNG (16-bit source) |
|-------|----------------------|---------------------|
| `u8`  | Direct copy          | `>> 8`              |
| `u16` | `* 257` scaling      | Direct copy         |
| `i8`  | XOR 0x80             | `(>> 8) XOR 0x80`  |
| `i16` | `* 257` then XOR     | XOR 0x8000          |
| `f32` | `/ 255.0`            | `/ 65535.0`         |

### XOR Trick for Signed Types

Signed integer decoding uses a bit-flip to convert unsigned pixel data into the
signed range, which is the standard approach for ML quantization:

- **i8**: `(u8_value ^ 0x80) as i8` — maps `0→-128`, `128→0`, `255→127`
- **i16**: `(u16_value ^ 0x8000) as i16` — maps `0→-32768`, `32768→0`, `65535→32767`

### u16 Scaling from u8

When 8-bit PNG data is decoded into `u16`, each byte is scaled to the full
16-bit range: `u8_value as u16 * 257`. This maps `0→0`, `128→32896`, `255→65535`
exactly (257 = 0x0101).

## Scratch Buffer Strategy

### JPEG (`JpegDecoderState`)

The custom JPEG decoder uses `JpegDecoderState`, which persists across frames:

- `mcu_scratch` (`McuScratch`): per-component IDCT output for one MCU row band,
  grown to the high-water mark and reused. After the first decode at a given
  resolution, subsequent CPU JPEG decodes perform **zero heap allocations**.
- `v4l2` (Linux + feature): the lazily-probed hardware backend, holding the
  persistent streaming session (mapped OUTPUT buffer) and the CAPTURE DMA
  scratch across frames.

There is no EXIF rotation scratch — the codec reads the orientation tag and
reports it without allocating a rotation workspace.

### PNG (`zune-png`)

PNG decoding uses `zune-png`, which allocates internal decoder state on each
call. The edgefirst-codec PNG layer reuses `ImageDecoder.input_buffer` for
`Read`-based input, but the zune-png library itself allocates per-frame.

### Allocation Sources by Layer

| Layer                    | After Warmup     | Notes                        |
|--------------------------|------------------|------------------------------|
| JPEG `McuScratch`        | No allocations   | Grows to high-water mark     |
| JPEG Huffman/quant tables| No allocations   | LUTs reused when DHT unchanged; otherwise rebuilt from marker data |
| JPEG IDCT workspace      | No allocations   | Stack-allocated (`[i32; 64]` scalar/SSE; NEON keeps it in registers) |
| JPEG native row write    | No allocations   | Strided into pre-allocated tensor |
| V4L2 streaming session   | No allocations   | OUTPUT buffer + DMA scratch persist; geometry changes are ioctl-only |
| zune-png `decode()`      | 1 `Vec` / call   | Returns owned `Vec<u16/u8>`  |
| zune-png `decode_into()` | ~3 `brk` / call  | Internal filter state        |
