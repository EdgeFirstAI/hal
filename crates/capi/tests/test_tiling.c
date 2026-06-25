// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// Tiling module tests for EdgeFirst HAL C API
//
// Covers the pure (no-GPU) tiling surface: grid geometry, plan metadata,
// box lift, IOS-vs-IOU merge, and the streaming accumulator fan-in/finalize.
// GPU paths (alloc_tile_batch / tile_into / tile_one) are exercised only for
// null-argument / validation behavior, since CI has no GPU.

#include "test_common.h"

// =============================================================================
// Default constructors
// =============================================================================

static void test_tiling_config_default(void) {
    TEST("tiling_config_default");

    struct hal_tiling_config cfg = hal_tiling_config_default(640, 480);
    ASSERT_EQ(640, cfg.tile_w);
    ASSERT_EQ(480, cfg.tile_h);
    // Deploy default: overlap 0.2, stretch fit, [114,114,114,255] pad.
    ASSERT_FLOAT_EQ(0.2f, cfg.overlap_ratio, 1e-6);
    ASSERT_EQ(HAL_FIT_STRETCH, cfg.fit);
    ASSERT_EQ(114, cfg.pad[0]);
    ASSERT_EQ(114, cfg.pad[1]);
    ASSERT_EQ(114, cfg.pad[2]);
    ASSERT_EQ(255, cfg.pad[3]);

    TEST_PASS();
}

static void test_merge_config_default(void) {
    TEST("merge_config_default");

    struct hal_merge_config cfg = hal_merge_config_default();
    // Library default: IOS / 0.5 / class-aware / 300 / 0.0.
    ASSERT_EQ(HAL_MATCH_METRIC_IOS, cfg.metric);
    ASSERT_FLOAT_EQ(0.5f, cfg.threshold, 1e-6);
    ASSERT_FALSE(cfg.class_agnostic);
    ASSERT_EQ(300, cfg.max_det);
    ASSERT_FLOAT_EQ(0.0f, cfg.score_threshold, 1e-6);

    TEST_PASS();
}

// =============================================================================
// Grid geometry
// =============================================================================

static void test_tile_grid_axis_worked_example(void) {
    TEST("tile_grid_axis_worked_example");

    // adis-uav-model sahi(1920, 640, 0.1) along one axis => origins
    // [0, 427, 853, 1280]. Use a 1-row frame (frame_h == tile_h) so the grid
    // is a single row: first origin x == 0, last origin x == 1280.
    struct hal_tile_spec_list* list = hal_tile_grid(640, 1920, 640, 640, 0.1f);
    ASSERT_NOT_NULL(list);

    size_t n = hal_tile_spec_list_len(list);
    ASSERT_EQ(4, n);  // 4 columns x 1 row

    struct hal_tile_spec first;
    struct hal_tile_spec last;
    ASSERT_EQ(0, hal_tile_spec_list_get(list, 0, &first));
    ASSERT_EQ(0, hal_tile_spec_list_get(list, n - 1, &last));
    ASSERT_EQ(0, first.source.x);
    ASSERT_EQ(1280, last.source.x);
    ASSERT_EQ(640, first.source.width);
    ASSERT_EQ(640, first.source.height);

    hal_tile_spec_list_free(list);
    TEST_PASS();
}

static void test_tile_grid_4k_count(void) {
    TEST("tile_grid_4k_count");

    // 3840x2160, 640 tiles, 0.2 overlap => 8 cols x 4 rows = 32 tiles.
    struct hal_tile_spec_list* list = hal_tile_grid(2160, 3840, 640, 640, 0.2f);
    ASSERT_NOT_NULL(list);
    ASSERT_EQ(32, hal_tile_spec_list_len(list));

    // Row-major: index 8 starts row 1 (x back to 0, y advanced).
    struct hal_tile_spec t8;
    ASSERT_EQ(0, hal_tile_spec_list_get(list, 8, &t8));
    ASSERT_EQ(0, t8.source.x);
    ASSERT_EQ(8, t8.index);
    ASSERT_EQ(1, t8.row);
    ASSERT_EQ(0, t8.col);

    hal_tile_spec_list_free(list);
    TEST_PASS();
}

static void test_tile_grid_invalid(void) {
    TEST("tile_grid_invalid");

    // Zero tile size.
    errno = 0;
    ASSERT_NULL(hal_tile_grid(640, 640, 0, 640, 0.2f));
    ASSERT_ERRNO(EINVAL);

    // overlap_ratio out of [0, 1).
    errno = 0;
    ASSERT_NULL(hal_tile_grid(640, 640, 640, 640, 1.0f));
    ASSERT_ERRNO(EINVAL);

    errno = 0;
    ASSERT_NULL(hal_tile_grid(640, 640, 640, 640, -0.1f));
    ASSERT_ERRNO(EINVAL);

    TEST_PASS();
}

static void test_tile_spec_list_get_bounds(void) {
    TEST("tile_spec_list_get_bounds");

    struct hal_tile_spec_list* list = hal_tile_grid(640, 640, 640, 640, 0.2f);
    ASSERT_NOT_NULL(list);
    ASSERT_EQ(1, hal_tile_spec_list_len(list));

    struct hal_tile_spec spec;
    errno = 0;
    ASSERT_EQ(-1, hal_tile_spec_list_get(list, 1, &spec));  // out of bounds
    ASSERT_ERRNO(EINVAL);
    errno = 0;
    ASSERT_EQ(-1, hal_tile_spec_list_get(list, 0, NULL));  // null out
    ASSERT_ERRNO(EINVAL);

    // NULL list is benign.
    ASSERT_EQ(0, hal_tile_spec_list_len(NULL));
    hal_tile_spec_list_free(NULL);

    hal_tile_spec_list_free(list);
    TEST_PASS();
}

// =============================================================================
// Plan metadata (pure, no GPU)
// =============================================================================

static void test_plan_tiles_metadata(void) {
    TEST("plan_tiles_metadata");

    struct hal_image_processor* proc = hal_image_processor_new();
    if (proc == NULL) {
        TEST_SKIP("no image processor backend available");
        return;
    }

    struct hal_tiling_config cfg = hal_tiling_config_default(640, 640);
    cfg.overlap_ratio = 0.2f;  // stretch fit by default

    struct hal_tile_placement_list* plan =
        hal_image_processor_plan_tiles(proc, 3840, 2160, &cfg);
    ASSERT_NOT_NULL(plan);
    ASSERT_EQ(32, hal_tile_placement_list_len(plan));

    struct hal_tile_placement p0;
    ASSERT_EQ(0, hal_tile_placement_list_get(plan, 0, &p0));
    ASSERT_EQ(0, p0.index);
    ASSERT_EQ(32, p0.count);
    ASSERT_FLOAT_EQ(0.0f, p0.origin_x, 1e-3);
    ASSERT_FLOAT_EQ(0.0f, p0.origin_y, 1e-3);
    ASSERT_FLOAT_EQ(640.0f, p0.crop_w, 1e-3);
    ASSERT_FLOAT_EQ(640.0f, p0.crop_h, 1e-3);
    ASSERT_FLOAT_EQ(3840.0f, p0.frame_w, 1e-3);
    ASSERT_FLOAT_EQ(2160.0f, p0.frame_h, 1e-3);
    // Stretch fit => no letterbox.
    ASSERT_FALSE(p0.has_letterbox);

    hal_tile_placement_list_free(plan);

    // Invalid config (zero tile) -> NULL + EINVAL.
    struct hal_tiling_config bad = hal_tiling_config_default(0, 640);
    errno = 0;
    ASSERT_NULL(hal_image_processor_plan_tiles(proc, 1920, 1080, &bad));
    ASSERT_ERRNO(EINVAL);

    // NULL config / processor.
    errno = 0;
    ASSERT_NULL(hal_image_processor_plan_tiles(proc, 1920, 1080, NULL));
    ASSERT_ERRNO(EINVAL);
    errno = 0;
    ASSERT_NULL(hal_image_processor_plan_tiles(NULL, 1920, 1080, &cfg));
    ASSERT_ERRNO(EINVAL);

    hal_image_processor_free(proc);
    TEST_PASS();
}

// =============================================================================
// GPU path null/validation guards (no GPU on CI)
// =============================================================================

static void test_gpu_paths_null_guards(void) {
    TEST("gpu_paths_null_guards");

    struct hal_tiling_config cfg = hal_tiling_config_default(640, 640);
    struct hal_tile_placement placement;
    memset(&placement, 0, sizeof(placement));

    // alloc_tile_batch: NULL processor / config -> NULL + EINVAL.
    errno = 0;
    ASSERT_NULL(hal_image_processor_alloc_tile_batch(
        NULL, 4, &cfg, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8, HAL_TENSOR_MEMORY_MEM));
    ASSERT_ERRNO(EINVAL);

    // tile_into / tile_one: NULL args -> NULL/-1 + EINVAL.
    errno = 0;
    ASSERT_NULL(hal_image_processor_tile_into(NULL, NULL, NULL, &cfg));
    ASSERT_ERRNO(EINVAL);

    errno = 0;
    ASSERT_EQ(-1, hal_image_processor_tile_one(NULL, NULL, NULL, &placement, &cfg));
    ASSERT_ERRNO(EINVAL);

    TEST_PASS();
}

// =============================================================================
// Box lift
// =============================================================================

static void test_lift_tile_boxes_null_input_is_empty(void) {
    TEST("lift_tile_boxes_null_input_is_empty");

    struct hal_tile_placement p;
    memset(&p, 0, sizeof(p));
    p.index = 0;
    p.count = 1;
    p.origin_x = 100.0f;
    p.origin_y = 200.0f;
    p.crop_w = 640.0f;
    p.crop_h = 640.0f;
    p.frame_w = 3840.0f;
    p.frame_h = 2160.0f;
    p.has_letterbox = false;

    // NULL boxes -> empty (not NULL) list.
    struct hal_detect_box_list* lifted = hal_lift_tile_boxes(NULL, &p);
    ASSERT_NOT_NULL(lifted);
    ASSERT_EQ(0, hal_detect_box_list_len(lifted));
    hal_detect_box_list_free(lifted);

    // NULL placement -> NULL + EINVAL.
    errno = 0;
    ASSERT_NULL(hal_lift_tile_boxes(NULL, NULL));
    ASSERT_ERRNO(EINVAL);

    TEST_PASS();
}

static void fill_placement(struct hal_tile_placement* p, size_t index, size_t count) {
    memset(p, 0, sizeof(*p));
    p->index = index;
    p->count = count;
    p->origin_x = 0.0f;
    p->origin_y = 0.0f;
    // 640x640 crop over a normalized [0,1] input maps norm -> [0,640] pixels.
    p->crop_w = 640.0f;
    p->crop_h = 640.0f;
    p->frame_w = 640.0f;
    p->frame_h = 640.0f;
    p->has_letterbox = false;
}

static void test_lift_origin_and_scale(void) {
    TEST("lift_origin_and_scale");

    // Rust doctest parity: origin (100,200), crop 640x640, box [0,0,1,1] (norm)
    // lifts to full-frame pixels [100,200,740,840]; box [0.25,0.5,0.75,1.0]
    // lifts to [260,520,580,840].
    struct hal_tile_placement p;
    memset(&p, 0, sizeof(p));
    p.index = 0;
    p.count = 1;
    p.origin_x = 100.0f;
    p.origin_y = 200.0f;
    p.crop_w = 640.0f;
    p.crop_h = 640.0f;
    p.frame_w = 3840.0f;
    p.frame_h = 2160.0f;
    p.has_letterbox = false;

    struct hal_detect_box in[2] = {
        {0.0f, 0.0f, 1.0f, 1.0f, 0.9f, 0},
        {0.25f, 0.5f, 0.75f, 1.0f, 0.8f, 0},
    };
    struct hal_detect_box_list* input = hal_detect_box_list_new(in, 2);
    ASSERT_NOT_NULL(input);

    struct hal_detect_box_list* lifted = hal_lift_tile_boxes(input, &p);
    ASSERT_NOT_NULL(lifted);
    // Input is borrowed, not consumed — still valid and freeable.
    ASSERT_EQ(2, hal_detect_box_list_len(input));
    ASSERT_EQ(2, hal_detect_box_list_len(lifted));

    struct hal_detect_box b0, b1;
    ASSERT_EQ(0, hal_detect_box_list_get(lifted, 0, &b0));
    ASSERT_EQ(0, hal_detect_box_list_get(lifted, 1, &b1));
    ASSERT_FLOAT_EQ(100.0f, b0.xmin, 1e-3);
    ASSERT_FLOAT_EQ(200.0f, b0.ymin, 1e-3);
    ASSERT_FLOAT_EQ(740.0f, b0.xmax, 1e-3);
    ASSERT_FLOAT_EQ(840.0f, b0.ymax, 1e-3);
    ASSERT_FLOAT_EQ(260.0f, b1.xmin, 1e-3);
    ASSERT_FLOAT_EQ(520.0f, b1.ymin, 1e-3);
    ASSERT_FLOAT_EQ(580.0f, b1.xmax, 1e-3);
    ASSERT_FLOAT_EQ(840.0f, b1.ymax, 1e-3);

    hal_detect_box_list_free(lifted);
    hal_detect_box_list_free(input);
    TEST_PASS();
}

static void test_merge_ios_vs_iou(void) {
    TEST("merge_ios_vs_iou");

    // Canonical case from the Rust reference: B fully inside A
    // (IoS=1.0, IoU=0.167). IOS merges to one box (group max score); IOU leaves
    // two. Boxes are already in full-frame pixel space.
    struct hal_detect_box in[2] = {
        {100.0f, 100.0f, 400.0f, 300.0f, 0.9f, 0},
        {350.0f, 100.0f, 400.0f, 300.0f, 0.7f, 0},
    };

    // IOS (default) -> 1 box.
    struct hal_detect_box_list* input = hal_detect_box_list_new(in, 2);
    ASSERT_NOT_NULL(input);
    struct hal_merge_config ios = hal_merge_config_default();  // IOS / 0.5
    struct hal_detect_box_list* merged_ios = hal_merge_tiled_detections(input, &ios);
    ASSERT_NOT_NULL(merged_ios);
    ASSERT_EQ(1, hal_detect_box_list_len(merged_ios));
    struct hal_detect_box m;
    ASSERT_EQ(0, hal_detect_box_list_get(merged_ios, 0, &m));
    ASSERT_FLOAT_EQ(0.9f, m.score, 1e-6);
    ASSERT_FLOAT_EQ(100.0f, m.xmin, 1e-3);
    ASSERT_FLOAT_EQ(400.0f, m.xmax, 1e-3);
    hal_detect_box_list_free(merged_ios);

    // IOU -> 2 boxes. Input still valid (borrowed) for reuse.
    struct hal_merge_config iou = hal_merge_config_default();
    iou.metric = HAL_MATCH_METRIC_IOU;
    struct hal_detect_box_list* merged_iou = hal_merge_tiled_detections(input, &iou);
    ASSERT_NOT_NULL(merged_iou);
    ASSERT_EQ(2, hal_detect_box_list_len(merged_iou));
    hal_detect_box_list_free(merged_iou);

    hal_detect_box_list_free(input);
    TEST_PASS();
}

static void test_merge_null_config(void) {
    TEST("merge_null_config");

    errno = 0;
    ASSERT_NULL(hal_merge_tiled_detections(NULL, NULL));
    ASSERT_ERRNO(EINVAL);

    // NULL boxes + valid config -> empty list (not NULL).
    struct hal_merge_config cfg = hal_merge_config_default();
    struct hal_detect_box_list* out = hal_merge_tiled_detections(NULL, &cfg);
    ASSERT_NOT_NULL(out);
    ASSERT_EQ(0, hal_detect_box_list_len(out));
    hal_detect_box_list_free(out);

    TEST_PASS();
}

// =============================================================================
// Accumulator fan-in + finalize-consumes
// =============================================================================

static void test_accumulator_fan_in(void) {
    TEST("accumulator_fan_in");

    struct hal_merge_config cfg = hal_merge_config_default();
    struct hal_tiled_frame_accumulator* acc =
        hal_tiled_frame_accumulator_new(640.0f, 640.0f, 3, &cfg, 8);
    ASSERT_NOT_NULL(acc);

    ASSERT_FALSE(hal_tiled_frame_accumulator_is_complete(acc));
    ASSERT_EQ(3, hal_tiled_frame_accumulator_remaining(acc));

    struct hal_tile_placement p0, p1, p2;
    fill_placement(&p0, 0, 3);
    fill_placement(&p1, 1, 3);
    fill_placement(&p2, 2, 3);

    // Push empty tiles (NULL box list == empty).
    ASSERT_EQ(1, hal_tiled_frame_accumulator_push_tile(acc, NULL, &p0));
    ASSERT_EQ(1, hal_tiled_frame_accumulator_push_tile(acc, NULL, &p1));
    ASSERT_EQ(1, hal_tiled_frame_accumulator_remaining(acc));
    ASSERT_FALSE(hal_tiled_frame_accumulator_is_complete(acc));

    // Duplicate index 1 is ignored (idempotent).
    ASSERT_EQ(0, hal_tiled_frame_accumulator_push_tile(acc, NULL, &p1));

    ASSERT_EQ(1, hal_tiled_frame_accumulator_push_tile(acc, NULL, &p2));
    ASSERT_TRUE(hal_tiled_frame_accumulator_is_complete(acc));
    ASSERT_EQ(0, hal_tiled_frame_accumulator_remaining(acc));

    // Finalize CONSUMES the handle. Output of an all-empty frame is empty.
    struct hal_detect_box_list* merged = hal_tiled_frame_accumulator_finalize(acc);
    ASSERT_NOT_NULL(merged);
    ASSERT_EQ(0, hal_detect_box_list_len(merged));
    hal_detect_box_list_free(merged);
    // acc is dead now; do NOT free it again.

    TEST_PASS();
}

static void test_accumulator_two_tile_merge(void) {
    TEST("accumulator_two_tile_merge");

    // Two overlapping tiles of a 640x640 frame: one sees the whole object, the
    // other a contained fragment. IOS merges them into a single detection.
    struct hal_merge_config cfg = hal_merge_config_default();  // IOS / 0.5
    struct hal_tiled_frame_accumulator* acc =
        hal_tiled_frame_accumulator_new(640.0f, 640.0f, 2, &cfg, 8);
    ASSERT_NOT_NULL(acc);

    // Distinct tile indices (the fan-in fence) but identical whole-frame
    // geometry, so both tiles lift their normalized boxes into the same pixel
    // region — mirroring the Rust e2e test where two overlapping tiles see the
    // same object (one whole, one fragment).
    struct hal_tile_placement p0, p1;
    memset(&p0, 0, sizeof(p0));
    p0.index = 0;
    p0.count = 2;
    p0.origin_x = 0.0f;
    p0.origin_y = 0.0f;
    p0.crop_w = 640.0f;
    p0.crop_h = 640.0f;
    p0.frame_w = 640.0f;
    p0.frame_h = 640.0f;
    p1 = p0;
    p1.index = 1;

    // Tile 0: the whole object [100,100]-[400,300] normalized to its 640 crop.
    struct hal_detect_box t0[1] = {
        {100.0f / 640.0f, 100.0f / 640.0f, 400.0f / 640.0f, 300.0f / 640.0f, 0.9f, 0},
    };
    // Tile 1: a contained fragment [350,100]-[400,300] of the same object.
    struct hal_detect_box t1[1] = {
        {350.0f / 640.0f, 100.0f / 640.0f, 400.0f / 640.0f, 300.0f / 640.0f, 0.7f, 0},
    };

    struct hal_detect_box_list* l0 = hal_detect_box_list_new(t0, 1);
    struct hal_detect_box_list* l1 = hal_detect_box_list_new(t1, 1);
    ASSERT_NOT_NULL(l0);
    ASSERT_NOT_NULL(l1);

    // Fan-in: each tile pushed under its own index but shared geometry, so both
    // lift into the same frame region (fan-in + lift + merge end to end).
    ASSERT_EQ(1, hal_tiled_frame_accumulator_push_tile(acc, l0, &p0));
    ASSERT_EQ(1, hal_tiled_frame_accumulator_push_tile(acc, l1, &p1));
    ASSERT_TRUE(hal_tiled_frame_accumulator_is_complete(acc));

    // Inputs are borrowed; free them now.
    hal_detect_box_list_free(l0);
    hal_detect_box_list_free(l1);

    struct hal_detect_box_list* merged = hal_tiled_frame_accumulator_finalize(acc);
    ASSERT_NOT_NULL(merged);
    // IOS merges the fragment into the full box => exactly one detection.
    ASSERT_EQ(1, hal_detect_box_list_len(merged));
    struct hal_detect_box m;
    ASSERT_EQ(0, hal_detect_box_list_get(merged, 0, &m));
    ASSERT_FLOAT_EQ(0.9f, m.score, 1e-6);
    hal_detect_box_list_free(merged);

    TEST_PASS();
}

static void test_accumulator_finalize_normalized(void) {
    TEST("accumulator_finalize_normalized");

    struct hal_merge_config cfg = hal_merge_config_default();
    struct hal_tiled_frame_accumulator* acc =
        hal_tiled_frame_accumulator_new(1280.0f, 640.0f, 1, &cfg, 8);
    ASSERT_NOT_NULL(acc);

    struct hal_tile_placement p;
    memset(&p, 0, sizeof(p));
    p.index = 0;
    p.count = 1;
    p.crop_w = 640.0f;
    p.crop_h = 640.0f;
    p.frame_w = 1280.0f;
    p.frame_h = 640.0f;

    ASSERT_EQ(1, hal_tiled_frame_accumulator_push_tile(acc, NULL, &p));
    ASSERT_TRUE(hal_tiled_frame_accumulator_is_complete(acc));

    struct hal_detect_box_list* norm =
        hal_tiled_frame_accumulator_finalize_normalized(acc);
    ASSERT_NOT_NULL(norm);
    ASSERT_EQ(0, hal_detect_box_list_len(norm));
    hal_detect_box_list_free(norm);

    TEST_PASS();
}

static void test_accumulator_abandon_free(void) {
    TEST("accumulator_abandon_free");

    struct hal_merge_config cfg = hal_merge_config_default();
    struct hal_tiled_frame_accumulator* acc =
        hal_tiled_frame_accumulator_new(640.0f, 640.0f, 4, &cfg, 8);
    ASSERT_NOT_NULL(acc);

    // Abandon without finalizing — _free is the correct path here.
    hal_tiled_frame_accumulator_free(acc);
    // NULL free is a no-op.
    hal_tiled_frame_accumulator_free(NULL);

    // NULL config -> NULL + EINVAL.
    errno = 0;
    ASSERT_NULL(hal_tiled_frame_accumulator_new(640.0f, 640.0f, 4, NULL, 8));
    ASSERT_ERRNO(EINVAL);

    // NULL acc finalize -> NULL + EINVAL.
    errno = 0;
    ASSERT_NULL(hal_tiled_frame_accumulator_finalize(NULL));
    ASSERT_ERRNO(EINVAL);

    TEST_PASS();
}

// =============================================================================
// Test runner
// =============================================================================

void run_tiling_tests(void) {
    TEST_SUITE("Tiling");

    // Defaults
    test_tiling_config_default();
    test_merge_config_default();

    // Grid
    test_tile_grid_axis_worked_example();
    test_tile_grid_4k_count();
    test_tile_grid_invalid();
    test_tile_spec_list_get_bounds();

    // Plan (pure)
    test_plan_tiles_metadata();

    // GPU guards
    test_gpu_paths_null_guards();

    // Lift / merge
    test_lift_tile_boxes_null_input_is_empty();
    test_lift_origin_and_scale();
    test_merge_ios_vs_iou();
    test_merge_null_config();

    // Accumulator
    test_accumulator_fan_in();
    test_accumulator_two_tile_merge();
    test_accumulator_finalize_normalized();
    test_accumulator_abandon_free();
}

#ifdef TEST_TILING_STANDALONE
// Define test result tracking variables for standalone mode
int tests_run = 0;
int tests_passed = 0;
int tests_failed = 0;
const char* current_test_name = NULL;
const char* current_suite_name = NULL;

int main(void) {
    run_tiling_tests();
    print_test_summary();
    return get_test_exit_code();
}
#endif
