/* Links libedgefirst_tensor and exercises every exported accessor through
 * the opaque handle — the C-side proof that the header's declarations, the
 * exported symbols, and the behavior agree. */
#include <edgefirst/tensor.h>
#include <assert.h>
#include <string.h>

int main(void) {
    uint64_t dims[2] = {2, 3};
    ef_tensor *t = ef_tensor_new(0 /* U8 */, dims, 2);
    assert(t != NULL);

    assert(ef_tensor_ndim(t) == 2);
    const uint64_t *shape = ef_tensor_shape(t);
    assert(shape && shape[0] == 2 && shape[1] == 3);
    const int64_t *strides = ef_tensor_strides(t);
    assert(strides && strides[0] == 3 && strides[1] == 1); /* bytes */
    assert(ef_tensor_dtype(t) == 0);
    assert(ef_tensor_storage_kind(t) == EF_STORAGE_KIND_MEM);
    assert(ef_tensor_plane_count(t) == 1);
    assert(strcmp(ef_tensor_format(t), "") == 0);

    ef_tensor_plane plane;
    assert(ef_tensor_plane_at(t, 0, &plane) == 0);
    assert(plane.handle == -1);   /* mem: no shareable handle */
    assert(plane.size == 6);
    assert(ef_tensor_plane_at(t, 1, &plane) != 0); /* out of range */

    /* Colorimetry: undefined by default, round-trips through set/get. */
    assert(ef_tensor_colorimetry(t) == 0);
    uint32_t packed = 1u | (2u << 8) | (1u << 16) | (1u << 24);
    assert(ef_tensor_set_colorimetry(t, packed) == 0);
    assert(ef_tensor_colorimetry(t) == packed);
    assert(ef_tensor_set_colorimetry(t, 0) == 0);
    assert(ef_tensor_colorimetry(t) == 0);

    /* view_origin: a whole tensor (never viewed/batched) has none. */
    struct EfViewOrigin origin;
    memset(&origin, 0xAA, sizeof(origin));
    assert(ef_tensor_view_origin(t, &origin) == 0);
    assert(origin.has_origin == 0);

    /* Null-safety: documented benign results, not crashes. */
    assert(ef_tensor_ndim(NULL) == 0);
    assert(ef_tensor_colorimetry(NULL) == 0);
    assert(ef_tensor_set_colorimetry(NULL, packed) != 0);
    assert(ef_tensor_view_origin(NULL, &origin) != 0);
    ef_tensor_free(NULL);

    ef_tensor_free(t);
    return 0;
}
