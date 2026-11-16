/* The minimal two-library user: edgefirst-codec and edgefirst-tensor only.
   Decodes a JPEG into a host-memory tensor and reads the pixels back. */
#include <edgefirst/codec.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    uint64_t dims[3] = { 480, 640, 3 };
    ef_tensor *dst = ef_tensor_new(EF_DTYPE_U8, dims, 3);
    if (!dst) { fprintf(stderr, "FAIL: ef_tensor_new returned NULL\n"); return 1; }
    if (ef_tensor_storage_kind(dst) != EF_STORAGE_KIND_MEM) {
        fprintf(stderr, "FAIL: expected host memory, got kind %u\n",
                ef_tensor_storage_kind(dst));
        return 1;
    }

    struct ef_image_decoder *d = ef_image_decoder_new();
    if (!d) { fprintf(stderr, "FAIL: ef_image_decoder_new returned NULL\n"); return 1; }

    int rc = ef_image_decoder_decode_file_into(d, "testdata/coco_420_odd.jpg", dst);
    if (rc != 0) {
        fprintf(stderr, "FAIL: decode_file_into -> %d (%s)\n", rc, strerror(rc));
        return 2;
    }

    struct ef_tensor_view v = { 0 };
    int mrc = ef_tensor_map(dst, EF_CPU_ACCESS_READ, &v);
    if (mrc != 0) { fprintf(stderr, "FAIL: ef_tensor_map -> %d\n", mrc); return 3; }
    if (v.len == 0 || v.ptr == NULL) {
        fprintf(stderr, "FAIL: empty map view\n"); return 4;
    }
    /* The decoded pixels are not all zero. */
    size_t nonzero = 0;
    for (size_t i = 0; i < v.len; i++) if (v.ptr[i]) nonzero++;
    if (nonzero * 100 < v.len) {
        fprintf(stderr, "FAIL: decoded image is %zu/%zu nonzero\n", nonzero, v.len);
        return 5;
    }

    ef_tensor_unmap(dst);
    ef_image_decoder_free(d);
    ef_tensor_free(dst);
    printf("PASS: two-library JPEG path works on a host-memory tensor\n");
    return 0;
}
