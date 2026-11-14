<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# edgefirst-codec (C API)

Image and video encode and decode with tensors as the pictures: JPEG and PNG decode, JPEG encode, and H.264 and MJPEG streams, on the platform's hardware (V4L2 on Linux, Media Foundation on Windows, nvJPEG on Linux with CUDA) or the CPU for still images. Ships as `libedgefirst_codec` with `edgefirst/codec.h`.

```sh
cc my_app.c $(pkg-config --cflags --libs edgefirst-codec) -o my_app
```

`edgefirst-codec` requires `edgefirst-tensor`: decoding means producing tensors, so `codec.h` includes `edgefirst/tensor.h` and the pkg-config pulls the tensor library in. It does **not** require `edgefirst-image` or `edgefirst-decoder`: an application that only decodes JPEGs links two libraries and no more. That is the whole reason these ship separately.

## Decoding images

```c
#include <edgefirst/codec.h>

ef_image_decoder *d = ef_image_decoder_new();

ef_tensor *dst = ef_tensor_image_alloc(1920, 1080, "NV12", EF_DTYPE_U8,
                                       /* has_memory */ 0, 0,
                                       EF_CPU_ACCESS_READ_WRITE);

int rc = ef_image_decoder_decode_into(d, jpeg_bytes, jpeg_len, dst);
if (rc != 0) {
    fprintf(stderr, "decode failed: %s\n", ef_codec_last_error());
    return 1;
}

ef_tensor_free(dst);
ef_image_decoder_free(d);
```

`ef_image_decoder_decode_file_into` takes a path instead of a buffer, and `ef_image_decoder_set_engine` picks the engine (`EF_CODEC_ENGINE_AUTO` by default).

`decode_into` does not allocate. The caller supplies the destination, which is what lets the decode target be a dma-buf the GPU reads next with no copy in between. The destination may be minted by any EdgeFirst library and backed any way it likes: host memory, dma-buf, PBO or D3D11 texture. Every library reaches the same tensor implementation in `libedgefirst_tensor`, so there is no "foreign" destination to re-import.

A destination too small is `ENOSPC`, so a caller who sized it from a JPEG header can tell "your buffer is too small" apart from "this file is not decodable" (`EBADMSG`).

## Encoding JPEG

```c
ef_image_encoder *e = ef_image_encoder_new(EF_CODEC_JPEG);
ef_image_encoder_set_quality(e, 90);

const uint8_t *jpeg;
size_t len;
if (ef_image_encoder_encode(e, image, &jpeg, &len) == 0)
    fwrite(jpeg, 1, len, file);   /* valid until the next encode */

ef_image_encoder_free(e);
```

The CPU encodes `u8` NV12, NV16, YUYV, GREY, RGB and RGBA images at any row stride; under `EF_CODEC_ENGINE_AUTO` a V4L2 JPEG encoder takes the DMA-BUF sources it can read. The bytes belong to the encoder, whose buffer the next encode reuses.

## Encoding video

```c
ef_video_encoder_config *cfg =
    ef_video_encoder_config_new(EF_CODEC_H264, 1920, 1080, "NV12", 30.0);
ef_video_encoder_config_set_bitrate(cfg, 8000000);   /* 0: chosen from size and rate */
ef_video_encoder *enc = ef_video_encoder_new(cfg);
ef_video_encoder_config_free(cfg);
if (!enc) { /* ENODEV, ENOTSUP or EINVAL; ef_codec_last_error() says why */ }

for (uint64_t pts = 0; next_frame(src); pts++) {
    ef_video_encoder_encode(enc, src, pts, pts == 0 ? EF_VIDEO_ENCODE_KEYFRAME : 0, NULL);
    const ef_video_packet *p;
    while ((p = ef_video_encoder_next_packet(enc)))
        write_access_unit(p->data, p->len, p->pts, p->keyframe);
}
ef_video_encoder_flush(enc);
/* ... pull the remaining packets the same way ... */
ef_video_encoder_free(enc);
```

Sources are read in place where the engine can (DMA-BUF on Linux, D3D11 textures on Windows); `ef_video_encoder_get_info` says which engine and device encode and how sources reach them. A packet stays valid until the next `ef_video_encoder_next_packet` call.

## Decoding video

```c
ef_video_decoder *dec = ef_video_decoder_new(EF_CODEC_H264, EF_CODEC_ENGINE_AUTO, 4);

ef_video_decoder_decode(dec, access_unit, len, pts);
for (;;) {
    uint32_t kind;
    ef_video_frame *f;
    ef_video_decoder_next_event(dec, &kind, &f);
    if (kind == EF_VIDEO_EVENT_NONE) break;
    if (kind == EF_VIDEO_EVENT_RECONFIGURED) {
        const ef_video_stream_info *s = ef_video_decoder_stream_info(dec);
        continue;   /* s->width, s->height, s->format, s->transfer, ... */
    }
    ef_tensor *picture = ef_video_frame_tensor(f);   /* owned by the frame */
    use_picture(picture, ef_video_frame_pts(f));
    ef_video_frame_free(f);                          /* gives the buffer back */
}
```

Frames come from a pool: the caller may hold up to the pool count at once, and a decode with every buffer held is `EAGAIN` (free some frames and decode the same access unit again). `ef_video_decoder_new_with_pool` decodes into the caller's own tensors instead. Feed H.264 as Annex B access units in decode order, and MJPEG as one JPEG picture per call.

## Engines and capabilities

Every codec runs on an engine (`ef_codec_engine`). `EF_CODEC_ENGINE_AUTO` takes the first engine that can do the request; the environment variable `EDGEFIRST_CODEC_ENGINE` (for example `v4l2,cpu`) narrows and orders that choice. An engine named explicitly is used alone.

| Codec | Linux | Windows | CPU |
|---|---|---|---|
| H.264 encode / decode | V4L2 | Media Foundation | no |
| MJPEG encode / decode | V4L2 JPEG encoder / decoder | no / Media Foundation | no |
| JPEG encode | V4L2 JPEG encoder | no | yes |
| JPEG / PNG decode | nvJPEG, V4L2 | no | yes |

There is no software video codec: where no engine runs a codec, opening it fails with `ENOTSUP` or `ENODEV`. `ef_codec_capabilities_new` lists what this machine's engines do, with each one's device, pixel formats and how frames move (`EF_CODEC_TRANSFER_ZERO_COPY`, `_GPU_COPY`, `_CPU_COPY`). Probing opens devices, so do it once.

`ef_codec_v4l2_available` and `ef_codec_nvjpeg_available` remain as quick probes for the JPEG decoders.

## Errors

Calls return 0 or an errno; functions that create an object return `NULL` and set `errno` (on Windows the thread's last-error code, read with `GetLastError`). `ef_codec_last_error` returns the calling thread's message for its last failed call.

| errno | Meaning |
|---|---|
| `ENODEV` | no engine or device here for the request |
| `ENOTSUP` | the engine exists, but not for this codec, format or feature |
| `EAGAIN` | the caller holds every frame of the decoder's pool |
| `ETIMEDOUT` | the device did not finish in time |
| `EINVAL` | a bad argument, configuration or source frame |
| `EBADMSG` | data the decoder cannot parse |
| `ENOSPC` | the destination tensor is too small |
| `EACCES` | the CPU must read or write a tensor not declared for that access |

Each object is used from one thread at a time.

<img referrerpolicy="no-referrer" src="https://px.edgefirst.ai/a.png?x-pxid=37aa7de2-6935-484e-b31d-9fd9a57fcd63" alt="" width="1" height="1" style="position:absolute; width:1px; height:1px; opacity:0; pointer-events:none;" />
