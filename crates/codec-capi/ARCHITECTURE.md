<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# edgefirst-codec-capi — Architecture

C ABI for `edgefirst-codec`. Ships as **`libedgefirst_codec`** with `edgefirst/codec.h`.

This library decodes JPEG/PNG into a caller-supplied `ef_tensor`, encodes JPEG from one, and runs the video encoders and decoders. It links `libedgefirst_tensor.so` (`DT_NEEDED` + `RUNPATH $ORIGIN`) instead of embedding a private tensor copy. pkg-config: `Requires: edgefirst-tensor`.

Exported symbols are `ef_*` only (version script / exported-symbols list). Constructors (`ef_image_decoder_new`, `ef_video_decoder_new`, …) live here; tensor accessors live only in `libedgefirst_tensor`.

## Modules

| Module | C surface |
|---|---|
| `error.rs` | errno mapping of `CodecError`, the thread-local `ef_codec_last_error`, and the guards every entry point runs in (a panic becomes `EINVAL`) |
| `engine.rs` | `ef_codec`, `ef_codec_engine`, `ef_codec_direction`, `ef_codec_transfer`, names, and the `ef_codec_capabilities` list |
| `decoder.rs` | `ef_image_decoder`, the availability probes, `ef_codec_save_jpeg`, `ef_codec_colorimetry_from_v4l2` |
| `encoder.rs` | `ef_image_encoder` |
| `video.rs` | `ef_video_encoder_config`, `ef_video_encoder`, `ef_video_decoder`, `ef_video_frame` |

Enums cross the ABI as `uint32_t` and are checked on the way in, so a value outside an enum is `EINVAL` rather than undefined behaviour in Rust.

## Video: push in, pull out

The Rust encoder returns access units from `encode` and `flush`, and the decoder returns `DecodeEvent`s from `decode` and `flush`. The C objects queue them, and the caller pulls one at a time (`ef_video_encoder_next_packet`, `ef_video_decoder_next_event`), so no call hands back a caller-freed array. A packet points into the encoder and lives until the next pull. A frame is its own object owning the `DecodedFrame`, so its pool buffer stays out of the pool until `ef_video_frame_free`; `ef_video_frame_tensor` lends the frame's tensor handle (`TensorDyn::as_raw`) without taking a reference, so the pool's slot accounting is unchanged. `ef_video_decoder_stream_info` returns the info of the last `RECONFIGURED` event pulled, not the backend's newest, so it always describes the frames being pulled.

`ef_video_decoder_new_with_pool` takes over the caller's reference to each tensor only on success: it retains each, opens the decoder, and releases the caller's references after. Taking them over rather than sharing them keeps each pool tensor's handle count at one, which the tensor library requires for the CPU write maps software decoders make.

## Structs

`ef_codec_capability`, `ef_video_packet`, `ef_video_encoder_info` and `ef_video_stream_info` are owned by the library and read through the pointers it returns, so fields may be appended without breaking callers. `ef_codec_region` is the one struct a caller fills (a crop passed by pointer); it is four `uint32_t` and does not change.

**ABI versioning.** `ef_codec_abi_version()` returns `3`. It moved from `2` with the engine, capability, image encoder and video entry points, and the errno mapping image decoding now shares (`EBADMSG` and `ENOTSUP` where it answered `EIO`); from `1` to `2` when `ef_codec_colorimetry_from_v4l2` gained its `is_rgb_or_hsv` argument. The rules that govern when it moves, and what mixing versions across the five libraries is allowed to do, are in [§ C ABI Stability and Versioning](https://github.com/EdgeFirstAI/hal/blob/main/ARCHITECTURE.md#c-abi-stability-and-versioning).

See [README.md](README.md) for usage and [tests/c/](tests/c/) for the consumers; `tests/c/test_codec_api.c` runs the whole surface, including an H.264 round trip where the machine has hardware for it.
