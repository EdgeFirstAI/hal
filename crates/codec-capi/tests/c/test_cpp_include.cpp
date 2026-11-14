// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// codec.h compiled as C++17: every enum is used both as its type and through
// its constants, and a function taking an enum value is declared.

#include <edgefirst/codec.h>

static_assert(sizeof(ef_codec) == sizeof(uint32_t), "ef_codec is 32 bits");

static ef_codec codec = EF_CODEC_H264;
static ef_codec_engine engine = EF_CODEC_ENGINE_AUTO;
static ef_codec_direction direction = EF_CODEC_DIRECTION_DECODE;
static ef_codec_transfer transfer = EF_CODEC_TRANSFER_ZERO_COPY;
static ef_h264_profile profile = EF_H264_PROFILE_HIGH;
static ef_video_event event = EF_VIDEO_EVENT_FRAME;
static ef_video_memory memory = EF_VIDEO_MEMORY_EXPORTED;

int touch_every_enum(void);

int touch_every_enum(void) {
    return static_cast<int>(codec) + static_cast<int>(engine) + static_cast<int>(direction) +
           static_cast<int>(transfer) + static_cast<int>(profile) + static_cast<int>(event) +
           static_cast<int>(memory) + static_cast<int>(ef_codec_abi_version());
}
