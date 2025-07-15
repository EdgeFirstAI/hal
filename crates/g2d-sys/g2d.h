/*
 *  Copyright (C) 2013-2016 Freescale Semiconductor, Inc.
 *  Copyright 2017-2022 NXP
 *
 *  Permission is hereby granted, free of charge, to any person obtaining
 *  a copy of this software and associated documentation files (the
 *  'Software'), to deal in the Software without restriction, including
 *  without limitation the rights to use, copy, modify, merge, publish,
 *  distribute, sub license, and/or sell copies of the Software, and to
 *  permit persons to whom the Software is furnished to do so, subject
 *  to the following conditions:

 *  The above copyright notice and this permission notice (including the
 *  next paragraph) shall be included in all copies or substantial
 *  portions of the Software.

 *  THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
 *  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 *  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 *  IN NO EVENT SHALL VIVANTE AND/OR ITS SUPPLIERS BE LIABLE FOR ANY
 *  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 *  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 *  SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 */

/*
 *	g2d.h
 *	Gpu 2d header file declare all g2d APIs exposed to application
 *	History :
 *	Date(y.m.d)        Author            Version        Description
 *	2012-10-22         Li Xianzhong      0.1            Created
 *	2013-02-22         Li Xianzhong      0.2            g2d_copy API is added
 *	2013-03-21         Li Xianzhong      0.4            g2d clear/rotation/flip APIs are supported
 *	2013-04-09         Li Xianzhong      0.5            g2d alpha blending feature is enhanced
 *	2013-05-17         Li Xianzhong      0.6            support vg core in g2d library
 *	2013-12-23         Li Xianzhong      0.7            support blend dim feature
 *	2014-03-20         Li Xianzhong      0.8            support pre-multipied & de-mutlipy out alpha
 *	2015-04-10         Meng Mingming     0.9            support multiple source blit
 * 	2015-11-03         Meng Mingming     1.0            support query 2D hardware type and feature
 *	2016-05-24         Meng Mingming     1.1            support get g2d_buf from dma fd
 *	2017-07-04         Prabhu Sundararaj 1.2            support get g2d_buf to export dma fd
 *	2018-04-24         Yuchou Gan        1.3            Add AMPHION_TILED support
 *	2019-12-30         Yong Gan          1.4            Add G2D_TILED_STATUS support
 *	2020-01-08         Li Xianzhong      1.5            support BT_601 and BT_709
 *	2020-08-25         Petr Cach         1.6            support BGR888, support BT_601FR, BT_709FR
 *	2021-12-02         Li Xianzhong      2.0            Add g2d fence sync extension

*/

#ifndef __G2D_H__
#define __G2D_H__

#ifdef __cplusplus
extern "C"  {
#endif

#define G2D_VERSION_MAJOR   2
#define G2D_VERSION_MINOR   1
#define G2D_VERSION_PATCH   0

enum g2d_format
{
//rgb formats
     G2D_RGB565               = 0,    /* [0:4] Blue;  [5:10] Green; [11:15] Red                      */
     G2D_RGBA8888             = 1,    /* [0:7] Red;   [8:15] Green; [16:23] Blue; [23:31] Alpha      */
     G2D_RGBX8888             = 2,    /* [0:7] Red;   [8:15] Green; [16:23] Blue; [23:31] don't care */
     G2D_BGRA8888             = 3,    /* [0:7] Blue;  [8:15] Green; [16:23] Red; [23:31] Alpha       */
     G2D_BGRX8888             = 4,    /* [0:7] Blue;  [8:15] Green; [16:23] Red; [23:31] don't care  */
     G2D_BGR565               = 5,    /* [0:4] Red;   [5:10] Green; [11:15] Blue                     */

     G2D_ARGB8888             = 6,    /* [0:7] Alpha; [8:15] Red;   [16:23] Green; [23:31] Blue      */
     G2D_ABGR8888             = 7,    /* [0:7] Alpha; [8:15] Blue;  [16:23] Green; [23:31] Red       */
     G2D_XRGB8888             = 8,    /* [0:7] don't care; [8:15] Red;  [16:23] Green; [23:31] Blue  */
     G2D_XBGR8888             = 9,    /* [0:7] don't care; [8:15] Blue; [16:23] Green; [23:31] Red   */
     G2D_RGB888               = 10,   /* [0:7] Red;   [8:15] Green; [16:23] Blue                     */
     G2D_BGR888               = 11,   /* [0:7] Blue;  [8:15] Green; [16:23] Red                      */

     G2D_RGBA5551             = 12,    /* [0:4] Red;   [5:9] Green; [10:14] Blue; [15] Alpha      */
     G2D_RGBX5551             = 13,    /* [0:4] Red;   [5:9] Green; [10:14] Blue; [15] don't care */
     G2D_BGRA5551             = 14,    /* [0:4] Blue;  [5:9] Green; [10:14] Red;  [15] Alpha       */
     G2D_BGRX5551             = 15,    /* [0:4] Blue;  [5:9] Green; [10:14] Red;  [15] don't care  */

//yuv formats
     G2D_NV12                 = 20,   /* 2 plane 420 format; plane 1: [0:7] Y ; plane 2: [0:7] U; [8:15] V */
     G2D_I420                 = 21,   /* 3 plane 420 format; plane 1: [0:7] Y ; plane 2: [0:7] U; plane 3: [0:7] V */
     G2D_YV12                 = 22,   /* 3 plane 420 format; plane 1: [0:7] Y ; plane 2: [0:7] V; plane 3: [0:7] U */
     G2D_NV21                 = 23,   /* 2 plane 420 format; plane 1: [0:7] Y ; plane 2: [0:7] V; [8:15] U */
     G2D_YUYV                 = 24,   /* 1 plane 422 format; [0:7] Y; [8:15; U; [16:23] Y; [24:31] V */
     G2D_YVYU                 = 25,   /* 1 plane 422 format; [0:7] Y; [8:15; V; [16:23] Y; [24:31] U */
     G2D_UYVY                 = 26,   /* 1 plane 422 format; [0:7] U; [8:15; Y; [16:23] V; [24:31] Y */
     G2D_VYUY                 = 27,   /* 1 plane 422 format; [0:7] V; [8:15; Y; [16:23] U; [24:31] Y */
     G2D_NV16                 = 28,   /* 2 plane 422 format; plane 1: [0:7] Y ; plane 2: [0:7] U; [8:15] V */
     G2D_NV61                 = 29,   /* 2 plane 422 format; plane 1: [0:7] Y ; plane 2: [0:7] V; [8:15] U */
};

enum g2d_blend_func
{
//basic blend
    G2D_ZERO                  = 0,
    G2D_ONE                   = 1,
    G2D_SRC_ALPHA             = 2,
    G2D_ONE_MINUS_SRC_ALPHA   = 3,
    G2D_DST_ALPHA             = 4,
    G2D_ONE_MINUS_DST_ALPHA   = 5,

// extensive blend is set with basic blend together,
// such as, G2D_ONE | G2D_PRE_MULTIPLIED_ALPHA
    G2D_PRE_MULTIPLIED_ALPHA  = 0x10,
    G2D_DEMULTIPLY_OUT_ALPHA  = 0x20,
};

enum g2d_cap_mode
{
    G2D_BLEND                 = 0,
    G2D_DITHER                = 1,
    G2D_GLOBAL_ALPHA          = 2,//only support source global alpha
    G2D_BLEND_DIM             = 3,//support special blend effect
    G2D_BLUR                  = 4,//blur effect
    G2D_YUV_BT_601            = 5,//yuv BT.601
    G2D_YUV_BT_709            = 6,//yuv BT.709
    G2D_YUV_BT_601FR          = 7,//yuv BT.601 Full Range
    G2D_YUV_BT_709FR          = 8,//yuv BT.709 Full Range
    G2D_WARPING               = 9,//can perform warp/dewarp operations
};

enum g2d_feature
{
    G2D_SCALING               = 0,
    G2D_ROTATION,
    G2D_SRC_YUV,
    G2D_DST_YUV,
    G2D_MULTI_SOURCE_BLT,
    G2D_FAST_CLEAR,
    G2D_WARP_DEWARP,
};

enum g2d_rotation
{
    G2D_ROTATION_0            = 0,
    G2D_ROTATION_90           = 1,
    G2D_ROTATION_180          = 2,
    G2D_ROTATION_270          = 3,
    G2D_FLIP_H                = 4,
    G2D_FLIP_V                = 5,
};

enum g2d_cache_mode
{
    G2D_CACHE_CLEAN           = 0,
    G2D_CACHE_FLUSH           = 1,
    G2D_CACHE_INVALIDATE      = 2,
};

enum g2d_hardware_type
{
    G2D_HARDWARE_2D           = 0,//default type
    G2D_HARDWARE_VG           = 1,
};

enum g2d_status
{
    G2D_STATUS_FAIL           =-1,
    G2D_STATUS_OK             = 0,
    G2D_STATUS_NOT_SUPPORTED  = 1,
};

#if defined(__QNX__)
#include <sys/types.h>
typedef off64_t g2d_phys_addr_t;
#else
typedef int     g2d_phys_addr_t;
#endif

struct g2d_surface
{
    enum g2d_format format;

    g2d_phys_addr_t planes[3];//surface buffer addresses are set in physical planes separately
                  //RGB:  planes[0] - RGB565/RGBA8888/RGBX8888/BGRA8888/BRGX8888
                  //NV12: planes[0] - Y, planes[1] - packed UV
                  //I420: planes[0] - Y, planes[1] - U, planes[2] - V
                  //YV12: planes[0] - Y, planes[1] - V, planes[2] - U
                  //NV21: planes[0] - Y, planes[1] - packed VU
                  //YUYV: planes[0] - packed YUYV
                  //YVYU: planes[0] - packed YVYU
                  //UYVY: planes[0] - packed UYVY
                  //VYUY: planes[0] - packed VYUY
                  //NV16: planes[0] - Y, planes[1] - packed UV
                  //NV61: planes[0] - Y, planes[1] - packed VU

    //blit rectangle in surface
    int left;
    int top;
    int right;
    int bottom;
    int stride;                     ///< buffer stride, in Pixels
    int width;                      ///< surface width, in Pixels
    int height;                     ///< surface height, in Pixels
    enum g2d_blend_func blendfunc;  ///< alpha blending parameters
    int global_alpha;               ///< value is 0 ~ 255
    //clrcolor format is RGBA8888, used as dst for clear, as src for blend dim
    int clrcolor;

    //rotation degree
    enum g2d_rotation rot;
};

struct g2d_surface_pair
{
    struct g2d_surface s;
    struct g2d_surface d;
};

struct g2d_buf
{
    void *buf_handle;
    void *buf_vaddr;
    int  buf_paddr;
    int  buf_size;
};

int g2d_open(void **handle);
int g2d_close(void *handle);

int g2d_make_current(void *handle, enum g2d_hardware_type type);

int g2d_clear(void *handle, struct g2d_surface *area);
int g2d_blit(void *handle, struct g2d_surface *src, struct g2d_surface *dst);
int g2d_copy(void *handle, struct g2d_buf *d, struct g2d_buf* s, int size);
int g2d_multi_blit(void *handle, struct g2d_surface_pair *sp[], int layers);

int g2d_query_hardware(void *handle, enum g2d_hardware_type type, int *available);
int g2d_query_feature(void *handle, enum g2d_feature feature, int *available);
int g2d_query_cap(void *handle, enum g2d_cap_mode cap, int *enable);
int g2d_enable(void *handle, enum g2d_cap_mode cap);
int g2d_disable(void *handle, enum g2d_cap_mode cap);

int g2d_cache_op(struct g2d_buf *buf, enum g2d_cache_mode op);
struct g2d_buf *g2d_alloc(int size, int cacheable);
struct g2d_buf *g2d_buf_from_fd(int fd);
int g2d_buf_export_fd(struct g2d_buf *);
struct g2d_buf *g2d_buf_from_virt_addr(void *vaddr, int size);
int g2d_free(struct g2d_buf *buf);

int g2d_flush(void *handle);
int g2d_finish(void *handle);

#ifdef __cplusplus
}
#endif

#endif
