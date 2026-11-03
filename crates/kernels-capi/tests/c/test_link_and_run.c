/* SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
 * SPDX-License-Identifier: Apache-2.0
 *
 * Links and RUNS against libedgefirst_kernels only: one plan of every type,
 * one small tensor each, one hand-derived value each. */
#include <edgefirst/kernels.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL %s:%d: %s (last error: %s)\n", __FILE__,   \
                    __LINE__, #cond, ef_kernels_last_error() ? ef_kernels_last_error() : "none"); \
            return 1;                                                        \
        }                                                                    \
    } while (0)

static ef_kernel_view make_view(void *data, uint32_t dtype, size_t ndim, const size_t *shape) {
    ef_kernel_view v;
    v.data = data;
    v.dtype = dtype;
    v.ndim = ndim;
    v.shape = shape;
    v.strides = NULL;
    return v;
}

static int pool_calls;

static void counting_pool(void *ctx, size_t n_tasks, void (*task)(void *, size_t), void *arg) {
    (void)ctx;
    pool_calls++;
    for (size_t i = 0; i < n_tasks; i++) task(arg, i);
}

int main(void) {
    const ef_kernel_quant out_q = {1.0f / 256.0f, -128};

    /* softmax_i8: four equal logits -> 1/4 each; 0.25 * 256 = 64; 64 + (-128) = -64. */
    {
        ef_softmax_i8_params p = {{0.1f, 0}, out_q, 1.0f};
        ef_softmax_i8_plan *plan = NULL;
        int8_t in[4] = {0, 0, 0, 0}, out[4] = {0};
        size_t shape[2] = {1, 4};
        ef_kernel_view iv = make_view(in, EF_KERNEL_DTYPE_I8, 2, shape);
        ef_kernel_view ov = make_view(out, EF_KERNEL_DTYPE_I8, 2, shape);
        CHECK(ef_softmax_i8_plan_new(&p, &plan) == 0 && plan != NULL);
        CHECK(ef_softmax_i8_plan_run(plan, &iv, &ov, NULL) == 0);
        for (int i = 0; i < 4; i++) CHECK(out[i] == -64);

        ef_kernel_pool pool = {NULL, counting_pool};
        memset(out, 0, sizeof out);
        CHECK(ef_softmax_i8_plan_run(plan, &iv, &ov, &pool) == 0);
        for (int i = 0; i < 4; i++) CHECK(out[i] == -64);
        CHECK(pool_calls > 0);
        ef_softmax_i8_plan_free(plan);
    }

    /* softmax_i8 with output zero point 0 is valid but unhandled: ENOTSUP. */
    {
        ef_softmax_i8_params p = {{0.1f, 0}, {1.0f / 256.0f, 0}, 1.0f};
        ef_softmax_i8_plan *plan = NULL;
        errno = 0;
        CHECK(ef_softmax_i8_plan_new(&p, &plan) == -1);
        CHECK(errno == ENOTSUP);
        CHECK(plan == NULL);
        CHECK(ef_kernels_last_error() != NULL);
        ef_softmax_i8_plan_free(NULL);
    }

    /* softmax_i16: four equal logits -> 1/4 each; 0.25 * 32768 = 8192. */
    {
        ef_softmax_i16_params p = {{0.001f, 0}, {1.0f / 32768.0f, 0}, 1.0f};
        ef_softmax_i16_plan *plan = NULL;
        int16_t in[4] = {7, 7, 7, 7}, out[4] = {0};
        size_t shape[2] = {1, 4};
        ef_kernel_view iv = make_view(in, EF_KERNEL_DTYPE_I16, 2, shape);
        ef_kernel_view ov = make_view(out, EF_KERNEL_DTYPE_I16, 2, shape);
        CHECK(ef_softmax_i16_plan_new(&p, &plan) == 0);
        CHECK(ef_softmax_i16_plan_run(plan, &iv, &ov, NULL) == 0);
        for (int i = 0; i < 4; i++) CHECK(out[i] == 8192);
        ef_softmax_i16_plan_free(plan);
    }

    /* sigmoid: logit 0 -> 0.5; 0.5 * 256 = 128; 128 + (-128) = 0. */
    {
        ef_sigmoid_params p = {EF_KERNEL_DTYPE_I8, {0.1f, 0}, EF_KERNEL_DTYPE_I8, out_q};
        ef_sigmoid_plan *plan = NULL;
        int8_t in[3] = {0, 0, 0}, out[3] = {9, 9, 9};
        size_t shape[1] = {3};
        ef_kernel_view iv = make_view(in, EF_KERNEL_DTYPE_I8, 1, shape);
        ef_kernel_view ov = make_view(out, EF_KERNEL_DTYPE_I8, 1, shape);
        CHECK(ef_sigmoid_plan_new(&p, &plan) == 0);
        CHECK(ef_sigmoid_plan_run(plan, &iv, &ov, NULL) == 0);
        for (int i = 0; i < 3; i++) CHECK(out[i] == 0);
        ef_sigmoid_plan_free(plan);
    }

    /* sigmoid f16 -> f32: the half 0x0000 is 0.0 and sigmoid(0) = 0.5. */
    {
        ef_sigmoid_params p = {EF_KERNEL_DTYPE_F16, {1.0f, 0}, EF_KERNEL_DTYPE_F32, {1.0f, 0}};
        ef_sigmoid_plan *plan = NULL;
        uint16_t in[2] = {0, 0};
        float out[2] = {0};
        size_t shape[1] = {2};
        ef_kernel_view iv = make_view(in, EF_KERNEL_DTYPE_F16, 1, shape);
        ef_kernel_view ov = make_view(out, EF_KERNEL_DTYPE_F32, 1, shape);
        CHECK(ef_sigmoid_plan_new(&p, &plan) == 0);
        CHECK(ef_sigmoid_plan_run(plan, &iv, &ov, NULL) == 0);
        CHECK(fabsf(out[0] - 0.5f) < 1e-3f && fabsf(out[1] - 0.5f) < 1e-3f);
        ef_sigmoid_plan_free(plan);
    }

    /* dfl: all-zero logits -> uniform over bins 0..3; expectation (0+1+2+3)/4 = 1.5 per side. */
    {
        ef_dfl_params p = {EF_KERNEL_DTYPE_F32, {1.0f, 0}, EF_KERNEL_DTYPE_F32, {1.0f, 0}, 4};
        ef_dfl_plan *plan = NULL;
        float in[16] = {0}, out[4] = {0};
        size_t ishape[2] = {1, 16}, oshape[2] = {1, 4};
        ef_kernel_view iv = make_view(in, EF_KERNEL_DTYPE_F32, 2, ishape);
        ef_kernel_view ov = make_view(out, EF_KERNEL_DTYPE_F32, 2, oshape);
        CHECK(ef_dfl_plan_new(&p, &plan) == 0);
        CHECK(ef_dfl_plan_run(plan, &iv, &ov, NULL) == 0);
        for (int i = 0; i < 4; i++) CHECK(fabsf(out[i] - 1.5f) < 1e-5f);
        ef_dfl_plan_free(plan);
    }

    /* dist2bbox: ltrb (1,1,3,3), cell (0.5,0.5), stride 8: xc = (0.5+(3-1)/2)*8 = 12, w = (1+3)*8 = 32,
     * so xyxy = (12-16, 12-16, 12+16, 12+16) = (-4,-4,28,28). */
    {
        ef_kernel_level level = {1, 1, 8.0f};
        ef_dist2bbox_params p = {EF_KERNEL_DTYPE_F32, {1.0f, 0}, EF_KERNEL_DTYPE_F32, {1.0f, 0},
                                 &level, 1, EF_KERNEL_BOX_XYXY};
        ef_dist2bbox_plan *plan = NULL;
        float in[4] = {1, 1, 3, 3}, out[4] = {0};
        size_t shape[2] = {1, 4};
        ef_kernel_view iv = make_view(in, EF_KERNEL_DTYPE_F32, 2, shape);
        ef_kernel_view ov = make_view(out, EF_KERNEL_DTYPE_F32, 2, shape);
        CHECK(ef_dist2bbox_plan_new(&p, &plan) == 0);
        CHECK(ef_dist2bbox_plan_run(plan, &iv, &ov, NULL) == 0);
        CHECK(out[0] == -4.0f && out[1] == -4.0f && out[2] == 28.0f && out[3] == 28.0f);
        ef_dist2bbox_plan_free(plan);
    }

    /* dfl_box: one-hot bins (1,1,3,3) give the same distances; xywh = (12, 12, 32, 32). */
    {
        ef_kernel_level level = {1, 1, 8.0f};
        ef_dfl_box_params p = {EF_KERNEL_DTYPE_I8, {0.5f, 0}, EF_KERNEL_DTYPE_F32, {1.0f, 0},
                               4, &level, 1, EF_KERNEL_BOX_XYWH};
        ef_dfl_box_plan *plan = NULL;
        int8_t in[16];
        float out[4] = {0};
        const size_t bins[4] = {1, 1, 3, 3};
        memset(in, -128, sizeof in);
        for (size_t side = 0; side < 4; side++) in[side * 4 + bins[side]] = 127;
        size_t ishape[2] = {1, 16}, oshape[2] = {1, 4};
        ef_kernel_view iv = make_view(in, EF_KERNEL_DTYPE_I8, 2, ishape);
        ef_kernel_view ov = make_view(out, EF_KERNEL_DTYPE_F32, 2, oshape);
        CHECK(ef_dfl_box_plan_new(&p, &plan) == 0);
        CHECK(ef_dfl_box_plan_run(plan, &iv, &ov, NULL) == 0);
        CHECK(out[0] == 12.0f && out[1] == 12.0f && out[2] == 32.0f && out[3] == 32.0f);
        ef_dfl_box_plan_free(plan);
    }

    /* lut table: entry i is -i (mod 256), looked up by the raw byte: 1 -> -1, -2 -> 2, 0 -> 0, 127 -> -127. */
    {
        uint8_t table[256];
        for (int i = 0; i < 256; i++) table[i] = (uint8_t)(-i);
        ef_lut_i8_plan *plan = NULL;
        int8_t in[4] = {1, -2, 0, 127}, out[4] = {0};
        size_t shape[1] = {4};
        ef_kernel_view iv = make_view(in, EF_KERNEL_DTYPE_I8, 1, shape);
        ef_kernel_view ov = make_view(out, EF_KERNEL_DTYPE_I8, 1, shape);
        CHECK(ef_lut_i8_plan_new_table(EF_KERNEL_DTYPE_I8, table, &plan) == 0);
        CHECK(ef_lut_i8_plan_run(plan, &iv, &ov, NULL) == 0);
        CHECK(out[0] == -1 && out[1] == 2 && out[2] == 0 && out[3] == -127);
        ef_lut_i8_plan_free(plan);
    }

    /* lut sigmoid: logit 0 -> 0.5; 0.5 * 256 - 128 = 0. */
    {
        ef_lut_i8_plan *plan = NULL;
        int8_t in[2] = {0, 0}, out[2] = {9, 9};
        size_t shape[1] = {2};
        ef_kernel_view iv = make_view(in, EF_KERNEL_DTYPE_I8, 1, shape);
        ef_kernel_view ov = make_view(out, EF_KERNEL_DTYPE_I8, 1, shape);
        ef_kernel_quant in_q = {0.1f, 0};
        CHECK(ef_lut_i8_plan_new_sigmoid(EF_KERNEL_DTYPE_I8, in_q, out_q, &plan) == 0);
        CHECK(ef_lut_i8_plan_run(plan, &iv, &ov, NULL) == 0);
        CHECK(out[0] == 0 && out[1] == 0);
        ef_lut_i8_plan_free(plan);
    }

    CHECK(ef_kernels_abi_version() == 1);
    printf("OK\n");
    return 0;
}
