// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
// Windows: allocate a texture tensor, read its layout, share it, record and
// read a completion, wrap it again. Elsewhere: every export refuses with
// ENOTSUP and the declared-everywhere rule holds.
#include <edgefirst/tensor.h>
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>

// The WARP software adapter has no CUDA device behind it, so a texture
// allocated on it carries no CUDA handle and maps to NULL.
static int warp_adapter_requested(void) {
    static const char *const vars[2] = {"EDGEFIRST_D3D11_ADAPTER",
                                        "EDGEFIRST_ANGLE_ADAPTER"};
    char buf[16];
    int i;
    for (i = 0; i < 2; i++) {
        DWORD n = GetEnvironmentVariableA(vars[i], buf, (DWORD)sizeof(buf));
        if (n > 0 && n < (DWORD)sizeof(buf) && _stricmp(buf, "warp") == 0) {
            return 1;
        }
    }
    return 0;
}
#endif

int main(void) {
    uint64_t dims[3] = {32, 64, 4};
#ifndef _WIN32
    errno = 0;
    assert(ef_d3d11_device() == NULL && errno == ENOTSUP);
    assert(ef_d3d11_use_external_device((void *)1) == ENOTSUP);
    ef_d3d11_layout l;
    int layout_rc = ef_tensor_d3d11_layout(NULL, &l);
    assert(layout_rc == ENOTSUP || layout_rc == EINVAL);
    assert(ef_tensor_d3d11_texture(NULL) == NULL);
    assert(ef_tensor_d3d11_shared_handle(NULL) == NULL);
    void *fence = (void *)1;
    uint64_t value = 1;
    assert(ef_tensor_gpu_completion(NULL, &fence, &value) != 0);
    assert(ef_tensor_gpu_write_value(NULL) == 0);
    assert(ef_tensor_set_gpu_write(NULL, 1) != 0);
    assert(ef_tensor_from_d3d11_texture(NULL, EF_DTYPE_U8, dims, 3, "rgba8",
                                        EF_CPU_ACCESS_READ, NULL) == NULL);
    assert(ef_tensor_from_d3d11_shared_handle(NULL, EF_DTYPE_U8, dims, 3, "rgba8",
                                              EF_CPU_ACCESS_READ, NULL, 0,
                                              NULL) == NULL);
    printf("SKIP: D3D11 exports refuse off Windows as declared\n");
    return 0;
#else
    if (ef_is_gpu_buffer_available() == 0) {
        printf("SKIP: no D3D11 device\n");
        return 0;
    }
    assert(ef_d3d11_device() != NULL);

    ef_tensor_image_desc *d = ef_tensor_image_desc_new(64, 32, "rgba8", EF_DTYPE_U8);
    assert(d != NULL);
    assert(ef_tensor_image_desc_set_memory(d, EF_STORAGE_KIND_DMA_BUF) == 0);
    assert(ef_tensor_image_desc_set_access(d, EF_CPU_ACCESS_READ_WRITE) == 0);
    ef_tensor *t = ef_tensor_image_desc_alloc(d);
    ef_tensor_image_desc_free(d);
    assert(t != NULL);
    assert(ef_tensor_storage_kind(t) == EF_STORAGE_KIND_DMA_BUF);

    ef_d3d11_layout l = {0};
    assert(ef_tensor_d3d11_layout(t, &l) == 0);
    assert(l.texture_width == 64 && l.texture_height == 32 && l.bytes_per_texel == 4);

    void *tex = ef_tensor_d3d11_texture(t);
    assert(tex != NULL);

    ef_tensor_view v = {0};
    assert(ef_tensor_map(t, EF_CPU_ACCESS_READ_WRITE, &v) == 0);
    memset(v.ptr, 0x3C, v.len);
    assert(ef_tensor_unmap(t) == 0);

    // Ahead of the ef_tensor_set_gpu_write below: a CUDA map waits on the
    // tensor's recorded fence value before copying, and nothing in this test
    // signals the device fence, so a map taken after that write never
    // returns. Nothing above records a write.
    if (ef_is_cuda_available()) {
        void *m = ef_tensor_cuda_map_mut(t);
        if (m == NULL && warp_adapter_requested()) {
            printf("SKIP: WARP adapter: no CUDA device behind the texture\n");
        } else {
            size_t n = 0;
            assert(m != NULL);
            // The mapping is the texture's tight rows, not its staging pitch.
            assert(ef_tensor_cuda_device_ptr(m, &n) != NULL && n == 64 * 32 * 4);
            ef_tensor_cuda_unmap(m);
        }
    }

    void *h = ef_tensor_d3d11_shared_handle(t);
    assert(h != NULL);

    void *fence = (void *)1;
    uint64_t value = 1;
    assert(ef_tensor_gpu_completion(t, &fence, &value) == 0);
    assert(fence == NULL && value == 0);
    assert(ef_tensor_gpu_write_value(t) == 0);
    assert(ef_tensor_set_gpu_write(t, 42) == 0);
    assert(ef_tensor_gpu_completion(t, &fence, &value) == 0);
    assert(fence != NULL && value == 42);
    CloseHandle(fence);
    // The value alone, with no handle to close.
    assert(ef_tensor_gpu_write_value(t) == value);

    ef_tensor *again = ef_tensor_from_d3d11_shared_handle(
        h, EF_DTYPE_U8, dims, 3, "rgba8", EF_CPU_ACCESS_READ, NULL, 0, "again");
    assert(again != NULL);
    assert(ef_tensor_map(again, EF_CPU_ACCESS_READ, &v) == 0);
    assert(v.ptr[0] == 0x3C);
    assert(ef_tensor_unmap(again) == 0);
    ef_tensor_free(again);

    ef_tensor *wrapped = ef_tensor_from_d3d11_texture(
        tex, EF_DTYPE_U8, dims, 3, "rgba8", EF_CPU_ACCESS_READ, "wrapped");
    assert(wrapped != NULL);
    assert(ef_tensor_shape(wrapped)[0] == 32 && ef_tensor_shape(wrapped)[1] == 64);
    ef_tensor_free(wrapped);

    // dims that do not describe this texture are refused, not reinterpreted.
    uint64_t wrong[3] = {16, 16, 4};
    assert(ef_tensor_from_d3d11_texture(tex, EF_DTYPE_U8, wrong, 3, "rgba8",
                                        EF_CPU_ACCESS_READ, NULL) == NULL);
    assert(ef_tensor_from_d3d11_texture(tex, EF_DTYPE_U8, dims, 3, "no-such-format",
                                        EF_CPU_ACCESS_READ, NULL) == NULL);

    int rc = ef_tensor_try_map(t, EF_CPU_ACCESS_READ, &v);
    assert(rc == 0 || rc == EAGAIN);
    if (rc == 0) assert(ef_tensor_unmap(t) == 0);

    CloseHandle(h);
    ef_tensor_free(t);
    return 0;
#endif
}
