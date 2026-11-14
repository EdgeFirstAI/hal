# edgefirst-codec

JPEG and PNG decoding straight into pre-allocated tensors, JPEG encoding, and hardware H.264 and MJPEG video with tensors as the pictures — no per-frame allocations, zero-copy where the platform allows.

[![PyPI](https://img.shields.io/pypi/v/edgefirst-codec.svg)](https://pypi.org/project/edgefirst-codec/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Part of the EdgeFirst HAL

`edgefirst-codec` is one of five Python packages built from the [EdgeFirst Hardware Abstraction Layer](https://github.com/EdgeFirstAI/hal).

**The [`EdgeFirstAI/hal`](https://github.com/EdgeFirstAI/hal) repository is the home for all of them** — source, issue tracker, architecture documentation and release notes.

| Package | Provides |
|---|---|
| [`edgefirst-tensor`](https://pypi.org/project/edgefirst-tensor/) | Zero-copy tensor allocation and host/GPU/CUDA mapping |
| [`edgefirst-codec`](https://pypi.org/project/edgefirst-codec/) | JPEG and PNG decoding into pre-allocated tensors, JPEG encoding, and H.264 and MJPEG video (this package) |
| [`edgefirst-image`](https://pypi.org/project/edgefirst-image/) | GPU-accelerated colour conversion, resize, letterbox, tiling and drawing |
| [`edgefirst-decoder`](https://pypi.org/project/edgefirst-decoder/) | YOLO and ModelPack output decoding |
| [`edgefirst-tracker`](https://pypi.org/project/edgefirst-tracker/) | ByteTrack multi-object tracking |

## Installation

```bash
pip install edgefirst-codec
```

Requires Python 3.8 or newer; `edgefirst-tensor` and NumPy are installed automatically. Wheels are published for Linux (x86_64, aarch64), macOS (arm64), and Windows (x86_64), and are self-contained — there is no system JPEG library to install.

Packages install under the [PEP 420](https://peps.python.org/pep-0420/) `edgefirst.*` namespace, so the import is `edgefirst.codec`.

## Quick start

For maximum performance, decode straight into a tensor allocated by `edgefirst.image`'s `ImageProcessor.create_image()` — DMA/PBO-backed and GPU-pitch-aligned — then hand it to `ImageProcessor.convert()` for colour conversion and resize. `decode_file_into` / `decode_into` are free functions rather than `Tensor` methods precisely so they can take a tensor from another `edgefirst.*` package:

```python
import numpy as np
from edgefirst.codec import Tensor, decode_file_into
from edgefirst.image import Flip, ImageProcessor, PixelFormat, Rotation

processor = ImageProcessor()

# peek_image_info_file reads the header only — no pixels are decoded.
info = Tensor.peek_image_info_file("frame.jpg")
print(info.width, info.height, info.format)  # e.g. 1280 720 PixelFormat.Nv16

# Allocate once, outside the loop. The decoder reconfigures the tensor's
# dimensions and format within this allocation, so one tensor sized for the
# largest expected frame can receive smaller images without reallocating.
src = processor.create_image(
    info.width, info.height, PixelFormat.Nv12, "uint8", "readwrite"
)
dst = processor.create_image(640, 640, PixelFormat.Rgb, "uint8", "readwrite")

info = decode_file_into(src, "frame.jpg")  # or decode_into(src, jpeg_bytes)
# convert() performs colour conversion (native → RGB) and resize; the codec
# reports EXIF orientation in `info` but does not apply it, so pass it on.
rotation = Rotation.degrees_clockwise(info.rotation_degrees)
flip = Flip.Horizontal if info.flip_horizontal else Flip.NoFlip
processor.convert(src, dst, rotation, flip)

with dst.map() as view:
    data = np.frombuffer(view, dtype=np.uint8)
```

Same-package pipelines that never leave `edgefirst.codec` can use the equivalent `Tensor.decode_image_file()` method instead — `tensor.decode_image_file("frame.jpg")` — but `decode_image_file` is a method, and its `self` must literally be an `edgefirst.codec.Tensor`; a DMA/PBO-backed destination from `edgefirst.image.ImageProcessor.create_image()` is a different package's type and can never be that `self`, so it must go through the free functions `decode_into` / `decode_file_into` instead.

Images decode in their **native** pixel format and are never colour-converted, rotated or resized. A colour JPEG lands on the NV format matching its own chroma sampling (4:2:0 → `Nv12`, 4:2:2 → `Nv16`, 4:4:4 → `Nv24`), greyscale on `Grey`, and PNG on `Rgb` / `Rgba` / `Grey`. Nothing is resampled on the way out.

EXIF orientation is **reported, never applied**: `info.rotation_degrees` and `info.flip_horizontal` carry the transform your pipeline should apply downstream, and the reported dimensions are unrotated.

### Tuning the decode

```python
from edgefirst.codec import DctMethod, PixelFormat, set_dct_method, set_output_format

set_dct_method(DctMethod.Fast)  # faster IDCT, small bounded accuracy cost
set_output_format(PixelFormat.Rgb)  # fused colour conversion inside the decode
```

Both settings are **thread-local** — apply them on every thread that decodes. `set_output_format` fuses colour conversion into the decode's MCU write stage, which is a pure-CPU single-pass path; pass `None` to restore native output.

`PixelFormat` and the other value types (`TensorMemory`, `Region`, the colour axis enums) are accepted from any `edgefirst.*` package, not just this one — they compare and hash equal across packages by value, so `==`, dict keys and set membership all work regardless of which package's copy you pass. Tensors, `Decoder` and `ProtoData` cross packages too, through the capsule protocols. Importing `PixelFormat` from `edgefirst.codec` when calling into this package is still good style for readability, not a requirement. See the Interoperability section below for the one thing that does not cross: `isinstance` against a concrete class.

## Encoding JPEG

```python
from edgefirst.codec import ImageEncoder

encoder = ImageEncoder(quality=90)
jpeg = encoder.encode(image)  # bytes; image is a tensor from any edgefirst.* package
```

The CPU encodes `uint8` NV12, NV16, YUYV, GREY, RGB and RGBA images at any row stride; under `Engine.Auto` a V4L2 JPEG encoder takes the DMA-BUF sources it can read.

## Video

H.264 and MJPEG streams are encoded and decoded on the platform's hardware: V4L2 memory-to-memory devices on Linux, Media Foundation on Windows. There is no software video codec, so where no engine runs a codec, opening it raises `NoDeviceError` or `UnsupportedError`.

```python
from edgefirst.codec import Codec, PixelFormat, StreamInfo, VideoDecoder, VideoEncoder, access_units

encoder = VideoEncoder(Codec.H264, 1920, 1080, PixelFormat.Nv12, 30.0, bitrate=8_000_000)
for pts, picture in enumerate(pictures):
    packet = encoder.encode(picture, pts, keyframe=pts == 0)
    if packet is not None:
        out.write(packet.data)
for packet in encoder.flush():
    out.write(packet.data)

decoder = VideoDecoder(Codec.H264)
for pts, unit in enumerate(access_units(stream)):
    for event in decoder.decode(unit, pts):
        if isinstance(event, StreamInfo):
            print(event.width, event.height, event.transfer)
        else:
            use(event)  # a Frame: a Tensor with .pts, read in place
```

Sources are read in place where the engine can (DMA-BUF on Linux, D3D11 textures on Windows), and `encoder.info` says how they reach it. A decoded `Frame` is a `Tensor` over a buffer of the decoder's pool: pass it to `ImageProcessor.convert()` or `map()` it like any tensor. The buffer goes back to the decoder when the frame is no longer referenced, so hold frames only as long as their pictures are needed; a decoder whose every buffer is held raises `PoolExhaustedError`. `VideoDecoder(..., pool=[...])` decodes into the caller's own tensors instead. Feed H.264 as Annex B access units in decode order (`access_units()` splits a byte stream) and MJPEG as one JPEG picture per call.

## Engines and capabilities

Every codec runs on an `Engine`. `Engine.Auto` takes the first engine that can do the request; the environment variable `EDGEFIRST_CODEC_ENGINE` (for example `v4l2,cpu`) narrows and orders that choice. An engine passed explicitly is used alone. `capabilities()` lists what this machine's engines do, with each one's device, pixel formats and how pictures move (`Transfer.ZeroCopy`, `GpuCopy`, `CpuCopy`). Probing opens devices, so call it once.

## Errors

Codec failures raise `CodecError`, a `RuntimeError`, through one subclass per kind: `NoDeviceError`, `UnsupportedError`, `PoolExhaustedError`, `CodecTimeoutError`, `InvalidConfigError`, `InvalidInputError` and `InvalidDataError`. A CPU access a tensor's declaration does not cover raises `PermissionError`, as it does across every `edgefirst.*` package.

## What this package provides

| API | Purpose |
|---|---|
| `Tensor.peek_image_info()` / `peek_image_info_file()` | Read dimensions, format and EXIF orientation from the header alone |
| `Tensor.decode_image()` / `decode_image_file()` | Decode into a pre-allocated `edgefirst.codec.Tensor` |
| `decode_into()` / `decode_file_into()` | Decode into a tensor from *any* `edgefirst.*` package (e.g. `edgefirst.image.ImageProcessor.create_image()`) |
| `ImageInfo` | Decoded dimensions, native format, row stride, EXIF orientation |
| `set_dct_method()` / `DctMethod` | IDCT accuracy/speed selection |
| `set_output_format()` | Fused `Rgb` / `Nv12` decode output |
| `is_v4l2_available()` | Whether a V4L2 hardware JPEG decoder is present |
| `ImageEncoder` | JPEG encoding from a tensor |
| `VideoEncoder` / `EncodedFrame` / `EncoderInfo` | H.264 and MJPEG encoding on hardware |
| `VideoDecoder` / `Frame` / `StreamInfo` | H.264 and MJPEG decoding on hardware into tensor frames |
| `access_units()` | Split an H.264 Annex B byte stream into access units |
| `capabilities()` / `Capability` / `Engine` / `Codec` / `Transfer` | What this machine's codec engines do |
| `CodecError` and its subclasses | Typed codec failures |

### Hardware acceleration

On Linux the decoder transparently tries hardware backends before the software path and falls back without any API change:

- **V4L2 mem2mem** — SoC JPEG blocks such as the i.MX `mxc-jpeg`. Discovery is capability-based, with no hardcoded device node. Opt out with `EDGEFIRST_DISABLE_V4L2=1`.
- **nvJPEG** — CUDA GPU decode on NVIDIA platforms such as Jetson Orin. Loaded via `dlopen`, so there is no link-time CUDA dependency. Opt **in** with `EDGEFIRST_ENABLE_NVJPEG=1`.

## Footprint

Adding `edgefirst-codec` to a project costs roughly 1.5 MB of downloads (this package plus `edgefirst-tensor`), excluding NumPy. That is roughly an order of magnitude smaller than Pillow or OpenCV, and a few times larger than a bare libjpeg-turbo binding — while bundling PNG, EXIF and the hardware backends, with no system libraries to install. See [the Rust crate README](https://github.com/EdgeFirstAI/hal/blob/main/crates/codec/README.md#footprint) for the comparison table.

## Supported inputs

JPEG decoding covers baseline DCT, 8-bit precision, 1 or 3 components. Progressive, lossless, hierarchical and arithmetic-coded JPEG, CMYK/YCCK and non-8-bit precision are rejected with a typed error rather than mis-decoded. PNG goes through `zune-png` and supports 8-bit and 16-bit Luma / LumaA / RGB / RGBA. The [Rust crate README](https://github.com/EdgeFirstAI/hal/blob/main/crates/codec/README.md#decoder-limitations) documents the full matrix.

## Interoperability

`decode_into` / `decode_file_into` accept a tensor from *any* `edgefirst.*` package because each extension module registers its own `Tensor` type object ([PyO3 issue #1444](https://github.com/PyO3/pyo3/issues/1444) — `isinstance` across packages is always `False`, even for two objects wrapping the same Rust type). Acceptance goes through the `__edgefirst_tensor__` capsule protocol every `Tensor` implements:

```python
# CORRECT — works regardless of which edgefirst.* package produced obj
if hasattr(obj, "__edgefirst_tensor__"):
    ...

# WRONG — always False for a tensor from a sibling package
if isinstance(obj, edgefirst.image.Tensor):
    ...
```

`edgefirst.tensor.EdgeFirstTensorExportable` (re-exported here as `edgefirst.codec.EdgeFirstTensorExportable`) is a `typing.Protocol` you can annotate a cross-package parameter with. See [`crates/python-common/INTEROP.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/python-common/INTEROP.md) for the full protocol.

## Versioning and changelog

All four `edgefirst-*` packages are versioned and released together with the HAL itself, so a given version number refers to the same source tree in every language. Because of that there is no per-package changelog: release notes for every version live in the single [**CHANGELOG.md**](https://github.com/EdgeFirstAI/hal/blob/main/CHANGELOG.md) in the `hal` repository, which follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Links

- [Changelog](https://github.com/EdgeFirstAI/hal/blob/main/CHANGELOG.md) — release notes for all packages
- [Source](https://github.com/EdgeFirstAI/hal) — the `hal` monorepo
- [Issue tracker](https://github.com/EdgeFirstAI/hal/issues)
- [Package documentation](https://github.com/EdgeFirstAI/hal/blob/main/crates/codec/README.md) — the underlying Rust crate
- [EdgeFirst](https://edgefirst.ai)

## License

Apache-2.0

<img referrerpolicy="no-referrer" src="https://px.edgefirst.ai/a.png?x-pxid=37aa7de2-6935-484e-b31d-9fd9a57fcd63" alt="" width="1" height="1" style="position:absolute; width:1px; height:1px; opacity:0; pointer-events:none;" />
