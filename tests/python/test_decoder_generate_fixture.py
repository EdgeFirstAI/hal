"""Tests for scripts/decoder_generate_fixture.py.

Generator-internal pure functions are tested with synthetic inputs;
end-to-end generation with a real model lives in CI on dev hosts that
have the TFLite blobs.
"""

import importlib
import sys
from pathlib import Path

import pytest


def _import_generator():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts"))
    try:
        return importlib.import_module("decoder_generate_fixture")
    finally:
        sys.path.pop(0)


def test_generator_module_imports():
    mod = _import_generator()
    assert hasattr(mod, "main")
    assert hasattr(mod, "parse_args")
    assert hasattr(mod, "FixtureConfig")


def test_parse_args_minimum_required():
    mod = _import_generator()
    cfg = mod.parse_args(
        [
            "model.tflite",
            "image.jpg",
            "--output",
            "out.safetensors",
        ]
    )
    assert isinstance(cfg, mod.FixtureConfig)
    assert cfg.model_path == Path("model.tflite")
    assert cfg.image_path == Path("image.jpg")
    assert cfg.output_path == Path("out.safetensors")
    assert cfg.include_stages is True
    assert cfg.score_threshold == pytest.approx(0.001)
    assert cfg.iou_threshold == pytest.approx(0.7)
    assert cfg.expected_count_min == 10


def test_parse_args_no_stages_flag():
    mod = _import_generator()
    cfg = mod.parse_args(
        [
            "model.tflite",
            "image.jpg",
            "--output",
            "out.safetensors",
            "--no-stages",
        ]
    )
    assert cfg.include_stages is False


def test_parse_args_overrides():
    mod = _import_generator()
    cfg = mod.parse_args(
        [
            "model.tflite",
            "image.jpg",
            "--output",
            "out.safetensors",
            "--score-threshold",
            "0.05",
            "--iou-threshold",
            "0.5",
            "--expected-count-min",
            "42",
        ]
    )
    assert cfg.score_threshold == pytest.approx(0.05)
    assert cfg.iou_threshold == pytest.approx(0.5)
    assert cfg.expected_count_min == 42


def test_parse_args_missing_output_raises():
    mod = _import_generator()
    with pytest.raises(SystemExit):
        mod.parse_args(["model.tflite", "image.jpg"])


def _synthetic_layout_three_levels():
    """Hand-built object that quacks like PerScaleLayout for collect_raw_tensors_by_role."""

    class FakeLayout:
        pass

    layout = FakeLayout()
    layout.box_encoding = "dfl"
    layout.fpn_levels = [
        {
            "stride": 8.0,
            "h": 80,
            "w": 80,
            "reg_max": 16,
            "nc": 80,
            "nm": 32,
            "box_shape": (1, 80, 80, 64),
            "score_shape": (1, 80, 80, 80),
            "mc_shape": (1, 80, 80, 32),
            "box_quant": (0.157, -42),
            "score_quant": (0.178, 108),
            "mc_quant": (0.025, 31),
            "score_activation": "sigmoid",
        },
        {
            "stride": 16.0,
            "h": 40,
            "w": 40,
            "reg_max": 16,
            "nc": 80,
            "nm": 32,
            "box_shape": (1, 40, 40, 64),
            "score_shape": (1, 40, 40, 80),
            "mc_shape": (1, 40, 40, 32),
            "box_quant": (0.107, -36),
            "score_quant": (0.203, 112),
            "mc_quant": (0.024, 32),
            "score_activation": "sigmoid",
        },
        {
            "stride": 32.0,
            "h": 20,
            "w": 20,
            "reg_max": 16,
            "nc": 80,
            "nm": 32,
            "box_shape": (1, 20, 20, 64),
            "score_shape": (1, 20, 20, 80),
            "mc_shape": (1, 20, 20, 32),
            "box_quant": (0.094, -30),
            "score_quant": (0.239, 112),
            "mc_quant": (0.024, 35),
            "score_activation": "sigmoid",
        },
    ]
    layout.protos_shape = (1, 160, 160, 32)
    layout.protos_quant = (0.031, -119)
    layout.total_anchors = 80 * 80 + 40 * 40 + 20 * 20
    layout.nc = 80
    return layout


def test_collect_raw_tensors_from_synthetic_layout():
    import numpy as np

    mod = _import_generator()
    raw_outputs = {
        (1, 80, 80, 64): np.full((1, 80, 80, 64), -42, dtype=np.int8),
        (1, 40, 40, 64): np.full((1, 40, 40, 64), -36, dtype=np.int8),
        (1, 20, 20, 64): np.full((1, 20, 20, 64), -30, dtype=np.int8),
        (1, 80, 80, 80): np.full((1, 80, 80, 80), 108, dtype=np.int8),
        (1, 40, 40, 80): np.full((1, 40, 40, 80), 112, dtype=np.int8),
        (1, 20, 20, 80): np.full((1, 20, 20, 80), 112, dtype=np.int8),
        (1, 80, 80, 32): np.full((1, 80, 80, 32), 31, dtype=np.int8),
        (1, 40, 40, 32): np.full((1, 40, 40, 32), 32, dtype=np.int8),
        (1, 20, 20, 32): np.full((1, 20, 20, 32), 35, dtype=np.int8),
        (1, 160, 160, 32): np.full((1, 160, 160, 32), -119, dtype=np.int8),
    }
    layout = _synthetic_layout_three_levels()
    raw_table = mod.collect_raw_tensors_by_role(layout, raw_outputs)
    assert raw_table["raw.boxes_0"].shape == (1, 80, 80, 64)
    assert raw_table["raw.boxes_1"].shape == (1, 40, 40, 64)
    assert raw_table["raw.boxes_2"].shape == (1, 20, 20, 64)
    assert raw_table["raw.scores_0"].shape == (1, 80, 80, 80)
    assert raw_table["raw.mc_0"].shape == (1, 80, 80, 32)
    assert raw_table["raw.protos"].shape == (1, 160, 160, 32)


def test_collect_raw_tensors_skips_missing_mc_and_protos():
    """Detection-only models have no mc and no protos."""
    import numpy as np

    mod = _import_generator()

    class FakeLayout:
        pass

    layout = FakeLayout()
    layout.fpn_levels = [
        {
            "stride": 8.0,
            "h": 4,
            "w": 4,
            "reg_max": 16,
            "nc": 2,
            "nm": 0,
            "box_shape": (1, 4, 4, 64),
            "score_shape": (1, 4, 4, 2),
            "mc_shape": None,
            "box_quant": (0.1, 0),
            "score_quant": (0.2, 0),
            "mc_quant": None,
            "score_activation": "sigmoid",
        },
    ]
    layout.protos_shape = None
    layout.protos_quant = None
    raw_outputs = {
        (1, 4, 4, 64): np.zeros((1, 4, 4, 64), dtype=np.int8),
        (1, 4, 4, 2): np.zeros((1, 4, 4, 2), dtype=np.int8),
    }
    raw_table = mod.collect_raw_tensors_by_role(layout, raw_outputs)
    assert "raw.boxes_0" in raw_table
    assert "raw.scores_0" in raw_table
    assert "raw.mc_0" not in raw_table
    assert "raw.protos" not in raw_table


def test_capture_intermediates_dfl_one_level():
    import numpy as np

    mod = _import_generator()
    h, w, reg_max, nc, nm, stride = 2, 2, 16, 2, 4, 8
    box_q = np.zeros((1, h, w, 4 * reg_max), dtype=np.int8)
    sco_q = np.zeros((1, h, w, nc), dtype=np.int8)
    mc_q = np.zeros((1, h, w, nm), dtype=np.int8)
    inter = mod.capture_box_score_mc_intermediates(
        level_index=0,
        encoding="dfl",
        reg_max=reg_max,
        h=h,
        w=w,
        stride=stride,
        box_q=box_q,
        box_q_scale=0.1,
        box_q_zp=0,
        sco_q=sco_q,
        sco_q_scale=0.2,
        sco_q_zp=0,
        mc_q=mc_q,
        mc_q_scale=0.05,
        mc_q_zp=0,
    )
    assert inter["intermediate.boxes_0.dequant"].shape == (1, h, w, 4 * reg_max)
    assert inter["intermediate.boxes_0.ltrb"].shape == (h * w, 4)
    assert inter["intermediate.boxes_0.xywh"].shape == (h * w, 4)
    assert inter["intermediate.scores_0.dequant"].shape == (1, h, w, nc)
    assert inter["intermediate.scores_0.activated"].shape == (1, h, w, nc)
    assert inter["intermediate.mc_0.dequant"].shape == (1, h, w, nm)
    for arr in inter.values():
        assert arr.dtype == np.float32


def test_capture_intermediates_ltrb_one_level_omits_ltrb_key():
    import numpy as np

    mod = _import_generator()
    h, w, nc, nm, stride = 2, 2, 2, 4, 8
    box_q = np.zeros((1, h, w, 4), dtype=np.int8)
    sco_q = np.zeros((1, h, w, nc), dtype=np.int8)
    mc_q = np.zeros((1, h, w, nm), dtype=np.int8)
    inter = mod.capture_box_score_mc_intermediates(
        level_index=0,
        encoding="ltrb",
        reg_max=None,
        h=h,
        w=w,
        stride=stride,
        box_q=box_q,
        box_q_scale=0.1,
        box_q_zp=0,
        sco_q=sco_q,
        sco_q_scale=0.2,
        sco_q_zp=0,
        mc_q=mc_q,
        mc_q_scale=0.05,
        mc_q_zp=0,
    )
    assert "intermediate.boxes_0.dequant" in inter
    assert "intermediate.boxes_0.xywh" in inter
    assert "intermediate.boxes_0.ltrb" not in inter


def test_capture_intermediates_detection_only_omits_mc():
    import numpy as np

    mod = _import_generator()
    h, w, reg_max, nc, stride = 2, 2, 16, 2, 8
    box_q = np.zeros((1, h, w, 4 * reg_max), dtype=np.int8)
    sco_q = np.zeros((1, h, w, nc), dtype=np.int8)
    inter = mod.capture_box_score_mc_intermediates(
        level_index=0,
        encoding="dfl",
        reg_max=reg_max,
        h=h,
        w=w,
        stride=stride,
        box_q=box_q,
        box_q_scale=0.1,
        box_q_zp=0,
        sco_q=sco_q,
        sco_q_scale=0.2,
        sco_q_zp=0,
        mc_q=None,
        mc_q_scale=None,
        mc_q_zp=None,
    )
    assert "intermediate.boxes_0.dequant" in inter
    assert "intermediate.scores_0.dequant" in inter
    assert "intermediate.mc_0.dequant" not in inter


def test_capture_protos_intermediate():
    import numpy as np

    mod = _import_generator()
    proto_q = np.full((1, 16, 16, 32), -119, dtype=np.int8)
    out = mod.capture_protos_intermediate(proto_q, 0.031, -119)
    assert "intermediate.protos.dequant" in out
    deq = out["intermediate.protos.dequant"]
    assert deq.shape == (1, 16, 16, 32)
    assert deq.dtype == np.float32
    # value: (-119 - (-119)) * 0.031 = 0
    np.testing.assert_allclose(deq, 0.0)


def test_build_documentation_md_lists_all_keys():
    mod = _import_generator()
    keys = [
        "input.image",
        "raw.boxes_0",
        "raw.boxes_1",
        "raw.boxes_2",
        "raw.scores_0",
        "raw.mc_0",
        "raw.protos",
        "intermediate.boxes_0.dequant",
        "intermediate.boxes_0.ltrb",
        "intermediate.boxes_0.xywh",
        "decoded.boxes_xyxy",
        "decoded.scores",
        "decoded.classes",
    ]
    md = mod.build_documentation_md(
        decoder_family="per_scale_yolo_seg",
        model_basename="yolov8n-seg",
        encoding="dfl",
        keys=keys,
        nms_config={
            "iou_threshold": 0.7,
            "score_threshold": 0.001,
            "max_detections": 300,
        },
        expected_count_min=42,
    )
    assert "Physical output decomposition" in md
    assert "yolov8n-seg" in md
    for k in keys:
        assert k in md, f"documentation_md missing key {k!r}"
    assert "iou_threshold" in md
    assert "score_threshold" in md
    assert "expected_count_min" in md or "≥ 42" in md


def test_patch_schema_dshape_synthesizes_missing_entries():
    mod = _import_generator()
    raw_metadata = {
        "outputs": [
            {
                "name": "boxes",
                "type": "boxes",
                "shape": [1, 4, 8400],
                # no "dshape"
                "outputs": [{"name": "boxes_0", "shape": [1, 80, 80, 64]}],
            },
            {
                "name": "scores",
                "type": "scores",
                "shape": [1, 80, 8400],
                "outputs": [{"name": "scores_0", "shape": [1, 80, 80, 80]}],
            },
            {
                "name": "mask_coefs",
                "type": "mask_coefs",
                "shape": [1, 32, 8400],
                "outputs": [{"name": "mc_0", "shape": [1, 80, 80, 32]}],
            },
            {
                "name": "protos",
                "type": "protos",
                "shape": [1, 160, 160, 32],
                # no children, no dshape — must be untouched
            },
        ]
    }
    patched = mod._patch_schema_dshape(raw_metadata)

    boxes = patched["outputs"][0]
    assert "dshape" in boxes
    assert boxes["dshape"] == [{"batch": 1}, {"box_coords": 4}, {"num_boxes": 8400}]

    scores = patched["outputs"][1]
    assert scores["dshape"] == [{"batch": 1}, {"num_classes": 80}, {"num_boxes": 8400}]

    mc = patched["outputs"][2]
    assert mc["dshape"] == [{"batch": 1}, {"num_protos": 32}, {"num_boxes": 8400}]

    # Protos has no children — should not get a synthesized dshape.
    protos = patched["outputs"][3]
    assert "dshape" not in protos


def test_patch_schema_dshape_is_idempotent():
    mod = _import_generator()
    pre_patched = {
        "outputs": [
            {
                "name": "boxes",
                "type": "boxes",
                "shape": [1, 4, 8400],
                "dshape": [{"batch": 1}, {"box_coords": 4}, {"num_boxes": 8400}],
                "outputs": [{"name": "boxes_0", "shape": [1, 80, 80, 64]}],
            }
        ]
    }
    patched = mod._patch_schema_dshape(pre_patched)
    # Should be unchanged.
    assert patched["outputs"][0]["dshape"] == pre_patched["outputs"][0]["dshape"]


def test_patch_schema_dshape_does_not_mutate_input():
    mod = _import_generator()
    raw = {
        "outputs": [
            {
                "name": "boxes",
                "type": "boxes",
                "shape": [1, 4, 8400],
                "outputs": [{"name": "boxes_0", "shape": [1, 80, 80, 64]}],
            }
        ]
    }
    patched = mod._patch_schema_dshape(raw)
    # Original should not have gained dshape.
    assert "dshape" not in raw["outputs"][0]
    assert "dshape" in patched["outputs"][0]


def test_assemble_and_write_safetensors_round_trips(tmp_path):
    import numpy as np
    import safetensors

    mod = _import_generator()

    tensors = {
        "input.image": np.zeros((1, 8, 8, 3), dtype=np.uint8),
        "raw.boxes_0": np.zeros((1, 2, 2, 4), dtype=np.int8),
        "decoded.boxes_xyxy": np.zeros((0, 4), dtype=np.float32),
        "decoded.scores": np.zeros((0,), dtype=np.float32),
        "decoded.classes": np.zeros((0,), dtype=np.uint32),
    }
    metadata = {
        "format_version": "1",
        "decoder_family": "per_scale_yolo_seg",
        "model_basename": "synthetic",
        "schema_json": "{}",
        "quantization_json": "{}",
        "nms_config_json": '{"iou_threshold":0.7,"score_threshold":0.001,"max_detections":300}',
        "expected_count_min": "10",
        "documentation_md": "# test",
        "generated_at": "1970-01-01T00:00:00Z",
        "reference_script_sha256": "0" * 64,
        "generator_git_sha": "deadbeef",
    }

    out = tmp_path / "round.safetensors"
    mod.assemble_and_write(tensors, metadata, out)

    with safetensors.safe_open(str(out), framework="numpy") as f:
        loaded_meta = f.metadata()
        loaded_keys = list(f.keys())
        loaded_input = f.get_tensor("input.image")

    assert set(loaded_keys) == set(tensors.keys())
    assert loaded_meta["model_basename"] == "synthetic"
    assert loaded_meta["format_version"] == "1"
    assert loaded_input.dtype == np.uint8
    assert loaded_input.shape == (1, 8, 8, 3)
