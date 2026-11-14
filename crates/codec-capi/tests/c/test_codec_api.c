/* SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
   SPDX-License-Identifier: Apache-2.0

   The codec C API end to end: the capability list, a JPEG encoded from a
   tensor and decoded back, the refusal of a codec no engine runs, and, where
   this machine has an H.264 encoder and decoder, a stream encoded from
   tensors and decoded into frames.

   A part this machine cannot run prints "SKIP: <reason>" and passes. */
#include <edgefirst/codec.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

#define FAIL(...)                                                              \
    do {                                                                       \
        fprintf(stderr, "FAIL: " __VA_ARGS__);                                 \
        const char *why = ef_codec_last_error();                               \
        fprintf(stderr, " (%s)\n", why ? why : "no codec error");              \
        return 1;                                                              \
    } while (0)

static int has(const ef_codec_capabilities *list, uint32_t codec, uint32_t direction) {
    for (size_t i = 0; i < ef_codec_capabilities_count(list); i++) {
        const ef_codec_capability *c = ef_codec_capabilities_get(list, i);
        if (c->codec == codec && c->direction == direction &&
            c->engine != EF_CODEC_ENGINE_CPU)
            return 1;
    }
    return 0;
}

/* A gradient, so an encoder has something to code and a decoder something
   to get wrong. */
static int fill(ef_tensor *t, unsigned seed) {
    ef_tensor_view v = { 0 };
    if (ef_tensor_map(t, EF_CPU_ACCESS_WRITE, &v) != 0) return 0;
    for (size_t i = 0; i < v.len; i++) v.ptr[i] = (uint8_t)(i * 7 + seed);
    ef_tensor_unmap(t);
    return 1;
}

static int capabilities(void) {
    ef_codec_capabilities *list = ef_codec_capabilities_new();
    if (!list) FAIL("ef_codec_capabilities_new returned NULL");
    int cpu_jpeg = 0;
    for (size_t i = 0; i < ef_codec_capabilities_count(list); i++) {
        const ef_codec_capability *c = ef_codec_capabilities_get(list, i);
        printf("  %-16s %-6s %-8s %s, %zu formats, transfer %u\n",
               ef_codec_engine_name(c->engine), ef_codec_name(c->codec),
               c->direction == EF_CODEC_DIRECTION_ENCODE ? "encode" : "decode",
               c->device, c->format_count, c->transfer);
        if (c->engine == EF_CODEC_ENGINE_CPU && c->codec == EF_CODEC_JPEG &&
            c->direction == EF_CODEC_DIRECTION_ENCODE && c->format_count > 0)
            cpu_jpeg = 1;
    }
    if (ef_codec_capabilities_get(list, ef_codec_capabilities_count(list)) != NULL)
        FAIL("an index past the end returned an entry");
    ef_codec_capabilities_free(list);
    if (!cpu_jpeg) FAIL("the CPU JPEG encoder is not listed");
    return 0;
}

static int jpeg_round_trip(void) {
    ef_image_encoder *e = ef_image_encoder_new(EF_CODEC_JPEG);
    if (!e) FAIL("ef_image_encoder_new(JPEG) returned NULL");
    if (ef_image_encoder_set_engine(e, EF_CODEC_ENGINE_CPU) != 0) FAIL("set_engine");
    if (ef_image_encoder_set_quality(e, 0) != EINVAL) FAIL("quality 0 accepted");
    if (ef_image_encoder_set_quality(e, 90) != 0) FAIL("set_quality(90)");

    ef_tensor *src = ef_tensor_image_alloc(64, 48, "NV12", EF_DTYPE_U8, 0, 0,
                                           EF_CPU_ACCESS_READ_WRITE);
    if (!src) FAIL("allocating the NV12 source: %s", ef_tensor_last_error_message());
    if (!fill(src, 0)) FAIL("mapping the source");

    const uint8_t *jpeg = NULL;
    size_t len = 0;
    if (ef_image_encoder_encode(e, src, &jpeg, &len) != 0) FAIL("encode");
    if (len < 4 || jpeg[0] != 0xff || jpeg[1] != 0xd8) FAIL("not a JPEG");

    ef_tensor *dst = ef_tensor_image_alloc(64, 48, "NV12", EF_DTYPE_U8, 0, 0,
                                           EF_CPU_ACCESS_READ_WRITE);
    if (!dst) FAIL("allocating the destination");
    ef_image_decoder *d = ef_image_decoder_new();
    if (ef_image_decoder_decode_into(d, jpeg, len, dst) != 0) FAIL("decoding the JPEG");
    if (strcmp(ef_tensor_format(dst), "NV12") != 0) FAIL("decoded as %s", ef_tensor_format(dst));

    /* Bytes that are not an image are malformed data, not an I/O failure. */
    static const uint8_t junk[16] = { 0xff, 0xd8, 0xff, 0x00 };
    if (ef_image_decoder_decode_into(d, junk, sizeof junk, dst) != EBADMSG)
        FAIL("a truncated JPEG was not EBADMSG");
    if (ef_codec_last_error() == NULL) FAIL("no message for the failure");

    ef_image_decoder_free(d);
    ef_tensor_free(dst);
    ef_tensor_free(src);
    ef_image_encoder_free(e);
    return 0;
}

static int refusals(void) {
    if (ef_video_decoder_new(EF_CODEC_AV1, EF_CODEC_ENGINE_AUTO, 0) != NULL)
        FAIL("an AV1 decoder opened");
    if (ef_codec_last_error() == NULL) FAIL("no message for the refusal");
    if (ef_image_encoder_new(EF_CODEC_PNG) != NULL) FAIL("a PNG encoder opened");
    if (ef_video_decoder_new(99, EF_CODEC_ENGINE_AUTO, 0) != NULL)
        FAIL("codec 99 accepted");
    uint32_t engine = 0;
    if (ef_codec_engine_from_name("mf", &engine) != 0 ||
        engine != EF_CODEC_ENGINE_MEDIA_FOUNDATION)
        FAIL("\"mf\" did not parse as Media Foundation");
    return 0;
}

enum { W = 320, H = 240, FRAMES = 8 };

static int h264_round_trip(void) {
    ef_codec_capabilities *list = ef_codec_capabilities_new();
    int able = has(list, EF_CODEC_H264, EF_CODEC_DIRECTION_ENCODE) &&
               has(list, EF_CODEC_H264, EF_CODEC_DIRECTION_DECODE);
    ef_codec_capabilities_free(list);
    if (!able) {
        printf("SKIP: no H.264 hardware encoder and decoder here\n");
        return 0;
    }

    ef_video_encoder_config *cfg =
        ef_video_encoder_config_new(EF_CODEC_H264, W, H, "NV12", 30.0);
    if (!cfg) FAIL("ef_video_encoder_config_new");
    if (ef_video_encoder_config_set_gop(cfg, FRAMES) != 0) FAIL("set_gop");
    ef_video_encoder *enc = ef_video_encoder_new(cfg);
    ef_video_encoder_config_free(cfg);
    if (!enc) FAIL("ef_video_encoder_new");
    const ef_video_encoder_info *info = ef_video_encoder_get_info(enc);
    printf("  encoder: %s on %s, transfer %u\n", info->device,
           ef_codec_engine_name(info->engine), info->transfer);

    static uint8_t stream[1 << 20];
    size_t sizes[FRAMES * 2] = { 0 };
    size_t used = 0, packets = 0;
    ef_tensor *src = ef_tensor_image_alloc(W, H, "NV12", EF_DTYPE_U8, 0, 0,
                                           EF_CPU_ACCESS_READ_WRITE);
    if (!src) FAIL("allocating the source: %s", ef_tensor_last_error_message());
    for (unsigned i = 0; i <= FRAMES; i++) {
        if (i < FRAMES) {
            if (!fill(src, i * 3)) FAIL("mapping the source");
            /* One frame names its window explicitly: the whole picture. */
            const ef_codec_region whole = { 0, 0, W, H };
            if (ef_video_encoder_encode(enc, src, i, i == 0 ? EF_VIDEO_ENCODE_KEYFRAME : 0,
                                        i == 1 ? &whole : NULL) != 0)
                FAIL("encoding frame %u", i);
        } else if (ef_video_encoder_flush(enc) != 0) {
            FAIL("flushing the encoder");
        }
        const ef_video_packet *p;
        while ((p = ef_video_encoder_next_packet(enc)) != NULL) {
            if (packets == 0 && !p->keyframe) FAIL("the first access unit is not a key frame");
            if (used + p->len > sizeof stream || packets == FRAMES * 2) FAIL("stream overflow");
            memcpy(stream + used, p->data, p->len);
            used += p->len;
            sizes[packets++] = p->len;
        }
    }
    ef_tensor_free(src);
    ef_video_encoder_free(enc);
    if (packets != FRAMES) FAIL("%zu access units for %d frames", packets, FRAMES);

    ef_video_decoder *dec = ef_video_decoder_new(EF_CODEC_H264, EF_CODEC_ENGINE_AUTO, 0);
    if (!dec) FAIL("ef_video_decoder_new");
    unsigned frames = 0, reconfigured = 0;
    size_t at = 0;
    for (size_t i = 0; i <= packets; i++) {
        if (i < packets) {
            if (ef_video_decoder_decode(dec, stream + at, sizes[i], i) != 0)
                FAIL("decoding access unit %zu", i);
            at += sizes[i];
        } else if (ef_video_decoder_flush(dec) != 0) {
            FAIL("flushing the decoder");
        }
        for (;;) {
            uint32_t kind = 0;
            ef_video_frame *f = NULL;
            if (ef_video_decoder_next_event(dec, &kind, &f) != 0) FAIL("next_event");
            if (kind == EF_VIDEO_EVENT_NONE) break;
            if (kind == EF_VIDEO_EVENT_RECONFIGURED) {
                const ef_video_stream_info *s = ef_video_decoder_stream_info(dec);
                if (!s || s->width != W || s->height != H || strcmp(s->format, "NV12") != 0)
                    FAIL("stream info does not describe a %dx%d NV12 stream", W, H);
                printf("  decoder: %s, %u buffers, transfer %u\n",
                       ef_codec_engine_name(s->engine), s->buffers, s->transfer);
                reconfigured++;
                continue;
            }
            ef_tensor *t = ef_video_frame_tensor(f);
            if (!t || strcmp(ef_tensor_format(t), "NV12") != 0) FAIL("frame tensor");
            if (ef_video_frame_pts(f) != frames) FAIL("frame %u has pts %llu", frames,
                                                      (unsigned long long)ef_video_frame_pts(f));
            ef_video_frame_free(f);
            frames++;
        }
    }
    ef_video_decoder_free(dec);
    if (reconfigured == 0) FAIL("no RECONFIGURED event before the frames");
    if (frames != FRAMES) FAIL("%u frames decoded from %d", frames, FRAMES);
    return 0;
}

int main(void) {
    if (ef_codec_abi_version() != 3) {
        fprintf(stderr, "FAIL: ABI version %u\n", ef_codec_abi_version());
        return 1;
    }
    if (capabilities() || jpeg_round_trip() || refusals() || h264_round_trip()) return 1;
    printf("PASS: codec C API\n");
    return 0;
}
