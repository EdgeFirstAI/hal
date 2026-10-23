# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Model configuration discovery: ``resolve_model_config``, overrides,
provenance and the preprocessing spec it hands to ``edgefirst.image``."""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import numpy as np
import pytest
from edgefirst.decoder import (
    ConfigField,
    ConfigSource,
    DecodeMode,
    Decoder,
    LabelCountMismatchError,
    ModelConfigError,
    ModelFamily,
    ModelOverrides,
    ModelSignals,
    ModelTask,
    OverrideError,
    PreprocessSpec,
    ResolvedModelConfig,
    Tensor,
    UnresolvedConfigError,
    UnsupportedModelError,
    resolve_model_config,
)

ROOT = Path(__file__).resolve().parents[2]
SEGFORMER = ROOT / "testdata/hf/segformer-512"

IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)

SEGFORMER_SIGNALS = ModelSignals(
    "onnx",
    [("pixel_values", [1, 3, 512, 512], "f32")],
    [("logits", [1, 150, 128, 128], "f32")],
)


@pytest.fixture
def sidecar_model(tmp_path: Path) -> Path:
    """A model file with the SegFormer configs beside it (the ``optimum``
    export layout). The model itself carries no metadata."""
    for name in ("config.json", "preprocessor_config.json"):
        shutil.copy(SEGFORMER / name, tmp_path / name)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"")
    return model


@pytest.fixture
def bare_model(tmp_path: Path) -> Path:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"")
    return model


# --- resolution ------------------------------------------------------------


def test_sidecar_hf_resolves_semantic_segformer(sidecar_model: Path):
    cfg = resolve_model_config(sidecar_model, SEGFORMER_SIGNALS)
    assert isinstance(cfg, ResolvedModelConfig)
    assert cfg.family == ModelFamily.HuggingFace
    assert cfg.task == ModelTask.Semantic
    assert cfg.decode_mode == DecodeMode.Accurate
    assert len(cfg.labels) == 150
    assert cfg.labels[0] == "class_0"
    assert cfg.reduce_labels is True
    assert "segformer" in cfg.description.lower()
    assert all(isinstance(n, str) for n in cfg.notes)

    schema = cfg.schema
    assert schema["schema_version"] == 2
    (out,) = schema["outputs"]
    assert out["decoder"] == "semantic"
    assert out["shape"] == [1, 150, 128, 128]


def test_path_may_be_a_string(sidecar_model: Path):
    cfg = resolve_model_config(str(sidecar_model), SEGFORMER_SIGNALS)
    assert cfg.task == ModelTask.Semantic


def test_resolved_schema_builds_a_semantic_decoder(sidecar_model: Path):
    cfg = resolve_model_config(sidecar_model, SEGFORMER_SIGNALS)
    decoder = Decoder(cfg.schema)
    assert decoder.semantic_classes == 150
    assert decoder.semantic_mode == DecodeMode.Accurate


def test_preprocessing_is_plain_values(sidecar_model: Path):
    cfg = resolve_model_config(sidecar_model, SEGFORMER_SIGNALS)
    spec = cfg.preprocessing
    assert isinstance(spec, PreprocessSpec)
    assert spec.normalization == (IMAGENET_MEAN, IMAGENET_STD)
    assert spec.mean == IMAGENET_MEAN
    assert spec.std == IMAGENET_STD
    assert spec.fit == "stretch"
    assert spec.placement == "center"
    assert spec.pad == (114.0, 114.0, 114.0)
    assert spec.pad_space == "pixel"
    assert spec.interpolation == "bilinear"
    assert spec.size == (512, 512)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.fit = "letterbox"  # type: ignore[misc]


def test_preprocess_spec_defaults_are_the_edgefirst_defaults():
    spec = PreprocessSpec()
    assert spec.normalization == ((0.0, 0.0, 0.0), (255.0, 255.0, 255.0))
    assert (spec.fit, spec.placement, spec.pad_space) == (
        "letterbox",
        "center",
        "pixel",
    )
    assert spec.pad == (114.0, 114.0, 114.0)
    assert spec.interpolation == "bilinear"
    assert spec.size is None


def test_describe_lines_carry_values_and_sources(sidecar_model: Path):
    lines = resolve_model_config(sidecar_model, SEGFORMER_SIGNALS).describe()
    assert lines == [
        "normalize=imagenet [preprocessor_config.json]",
        "resize=stretch [preprocessor_config.json]",
        "placement=center [default]",
        "pad=114,114,114 pixel [default]",
        "interpolation=bilinear [preprocessor_config.json]",
        "task=semantic [config.json]",
        "decode=accurate [config.json]",
        "labels=150 classes [config.json]",
        "size=512x512 [model input]",
    ]


def test_provenance_maps_every_field(sidecar_model: Path):
    cfg = resolve_model_config(sidecar_model, SEGFORMER_SIGNALS)
    prov = cfg.provenance
    assert set(prov) == {
        ConfigField.Normalization,
        ConfigField.Resize,
        ConfigField.Placement,
        ConfigField.Pad,
        ConfigField.Interpolation,
        ConfigField.Task,
        ConfigField.DecodeMode,
        ConfigField.Labels,
        ConfigField.Size,
    }
    assert prov[ConfigField.Normalization] == (
        ConfigSource.SidecarHf,
        "preprocessor_config.json",
    )
    assert prov[ConfigField.Labels] == (ConfigSource.SidecarHf, "config.json")
    assert prov[ConfigField.Placement] == (ConfigSource.Default, None)
    assert cfg.source(ConfigField.Task) == (ConfigSource.SidecarHf, "config.json")


def test_heuristic_only_guesses_from_io_names(bare_model: Path):
    cfg = resolve_model_config(bare_model, SEGFORMER_SIGNALS)
    assert cfg.family == ModelFamily.HuggingFace
    assert cfg.task == ModelTask.Semantic
    assert cfg.labels == []
    assert any("labels" in n for n in cfg.notes)
    source, detail = cfg.source(ConfigField.Normalization)
    assert source == ConfigSource.Heuristic
    assert detail
    assert cfg.source(ConfigField.Task)[0] == ConfigSource.Heuristic
    assert "normalize=imagenet [guess: " in cfg.describe()[0]


def test_overrides_apply_last_and_are_recorded(sidecar_model: Path):
    overrides = ModelOverrides(
        normalize="unsigned",
        resize="letterbox-topleft",
        decode=DecodeMode.Fast,
        labels=["a"] * 150,
    )
    cfg = resolve_model_config(sidecar_model, SEGFORMER_SIGNALS, overrides)
    spec = cfg.preprocessing
    assert spec.normalization == ((0.0, 0.0, 0.0), (255.0, 255.0, 255.0))
    assert (spec.fit, spec.placement) == ("letterbox", "topleft")
    assert cfg.decode_mode == DecodeMode.Fast
    assert cfg.labels == ["a"] * 150
    for field in (
        ConfigField.Normalization,
        ConfigField.Resize,
        ConfigField.Placement,
        ConfigField.DecodeMode,
        ConfigField.Labels,
    ):
        assert cfg.source(field) == (ConfigSource.Override, None), field
    assert cfg.source(ConfigField.Interpolation)[0] == ConfigSource.SidecarHf


def test_conflicting_task_override_is_an_override_error(sidecar_model: Path):
    with pytest.raises(OverrideError, match="task=detect"):
        resolve_model_config(
            sidecar_model, SEGFORMER_SIGNALS, ModelOverrides(task="detect")
        )


def test_hf_configs_without_signals_are_unresolved(sidecar_model: Path):
    with pytest.raises(UnresolvedConfigError, match="model signals"):
        resolve_model_config(sidecar_model)


def test_nothing_found_is_unresolved_naming_the_fix(bare_model: Path):
    with pytest.raises(UnresolvedConfigError, match="task=semantic"):
        resolve_model_config(bare_model, None, ModelOverrides(task="semantic"))


def test_label_count_mismatch_has_its_own_error(sidecar_model: Path):
    with pytest.raises(LabelCountMismatchError) as info:
        resolve_model_config(
            sidecar_model, SEGFORMER_SIGNALS, ModelOverrides(labels=["a", "b"])
        )
    assert isinstance(info.value, OverrideError)
    assert (info.value.expected, info.value.found) == (150, 2)
    assert "labels names 2 classes but the model has 150" in str(info.value)


def test_empty_override_label_is_a_plain_override_error():
    labels = ["a"] * 150
    labels[3] = " "
    with pytest.raises(OverrideError, match="empty name at 3") as info:
        ModelOverrides(labels=labels)
    assert not isinstance(info.value, LabelCountMismatchError)


def test_untrimmed_hf_labels_are_trimmed(tmp_path: Path):
    legacy = ROOT / "testdata/hf/segformer-legacy-512"
    for name in ("config.json", "preprocessor_config.json"):
        shutil.copy(legacy / name, tmp_path / name)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"")
    cfg = resolve_model_config(model, SEGFORMER_SIGNALS)
    assert cfg.labels[7] == "bed"
    assert cfg.labels[33] == "chest of drawers"

    overridden = resolve_model_config(
        model, SEGFORMER_SIGNALS, ModelOverrides(labels=[f" c{i} " for i in range(150)])
    )
    assert overridden.labels[0] == "c0"


def test_empty_label_after_trim_is_noted_and_ignored(tmp_path: Path):
    config = (SEGFORMER / "config.json").read_text().replace('"class_7"', '"  "', 1)
    (tmp_path / "config.json").write_text(config)
    shutil.copy(SEGFORMER / "preprocessor_config.json", tmp_path)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"")
    cfg = resolve_model_config(model, SEGFORMER_SIGNALS)
    assert cfg.labels == []
    assert any("id2label[7]" in n for n in cfg.notes), cfg.notes
    assert any("labels unknown" in n for n in cfg.notes), cfg.notes


def test_missing_model_is_an_os_error(tmp_path: Path):
    missing = tmp_path / "absent" / "model.onnx"
    with pytest.raises(OSError) as info:
        resolve_model_config(missing)
    assert isinstance(info.value, FileNotFoundError)
    assert info.value.filename is not None


def test_a_directory_is_an_os_error(tmp_path: Path):
    with pytest.raises(IsADirectoryError):
        resolve_model_config(tmp_path)


def test_error_classes_are_value_errors():
    assert issubclass(ModelConfigError, ValueError)
    for cls in (OverrideError, UnresolvedConfigError, UnsupportedModelError):
        assert issubclass(cls, ModelConfigError)
    assert issubclass(UnsupportedModelError, NotImplementedError)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"labels": ""},
        {"labels": "a,,b"},
        {"labels": ["a", " "]},
        {"normalize": "0,0,0/-1,1,1"},
        {"normalize": "0,0,0/0,1,1"},
    ],
)
def test_bad_values_are_refused_when_set(kwargs):
    with pytest.raises(OverrideError):
        ModelOverrides(**kwargs)


def test_describe_round_trips_through_overrides(sidecar_model: Path):
    cfg = resolve_model_config(
        sidecar_model, SEGFORMER_SIGNALS, ModelOverrides(resize="letterbox")
    )
    keys = {"normalize", "resize", "task", "decode"}
    pairs = dict(
        line.split(" [")[0].split("=", 1)
        for line in cfg.describe()
        if line.split("=", 1)[0] in keys
    )
    assert pairs["resize"] == "letterbox-topleft"
    again = resolve_model_config(
        sidecar_model, SEGFORMER_SIGNALS, ModelOverrides(**pairs)
    )
    assert again.preprocessing == cfg.preprocessing


def test_signals_metadata_defaults_to_none():
    assert SEGFORMER_SIGNALS.metadata is None
    source, _, _, metadata = SEGFORMER_SIGNALS
    assert source == "onnx" and metadata is None


def test_signals_reject_dynamic_shapes(bare_model: Path):
    dynamic = ModelSignals(
        "onnx",
        [("pixel_values", ["batch", 3, 512, 512], "f32")],
        [("logits", [1, 150, 128, 128], "f32")],
    )
    with pytest.raises(ValueError, match="dynamic dimension"):
        resolve_model_config(bare_model, dynamic)


# --- overrides grammar -----------------------------------------------------


@pytest.mark.parametrize(
    "key,value",
    [
        ("task", "detect"),
        ("task", "segment"),
        ("task", "semantic"),
        ("normalize", "imagenet"),
        ("normalize", "unsigned"),
        ("normalize", "signed"),
        ("normalize", "raw"),
        ("normalize", "0.485,0.456,0.406/0.229,0.224,0.225"),
        ("resize", "stretch"),
        ("resize", "letterbox"),
        ("resize", "letterbox-center"),
        ("resize", "letterbox-topleft"),
        ("decode", "accurate"),
        ("decode", "fast"),
    ],
)
def test_override_grammar_round_trips(key: str, value: str):
    o = ModelOverrides(**{key: value})
    assert getattr(o, key) == value
    assert str(o) == f"{key}={value}"
    assert eval(repr(o), {"ModelOverrides": ModelOverrides}) == o
    again = ModelOverrides()
    again.set(key, value)
    assert again == o


def test_override_labels_accept_list_or_comma_string():
    assert ModelOverrides(labels="background, person").labels == [
        "background",
        "person",
    ]
    assert ModelOverrides(labels=["a,b", "c"]).labels == ["a,b", "c"]
    assert repr(ModelOverrides(labels=["it's"])) == "ModelOverrides(labels=['it\\'s'])"


def test_override_enums_are_accepted():
    o = ModelOverrides(task=ModelTask.Segment, decode=DecodeMode.Fast)
    assert (o.task, o.decode) == ("segment", "fast")


def test_empty_overrides():
    o = ModelOverrides()
    assert o.is_empty
    assert str(o) == ""
    assert repr(o) == "ModelOverrides()"
    assert o.task is None and o.labels is None
    assert not ModelOverrides(task="detect").is_empty


@pytest.mark.parametrize(
    "kwargs,accepted",
    [
        ({"task": "classify"}, "detect | segment | semantic"),
        (
            {"normalize": "imagenet2"},
            "unsigned | signed | raw | imagenet | R,G,B/R,G,B",
        ),
        ({"normalize": "1,2/3,4,5"}, "R,G,B/R,G,B"),
        ({"resize": "fill"}, "stretch | letterbox | letterbox-center"),
        ({"decode": "exact"}, "accurate | fast"),
    ],
)
def test_bad_override_lists_accepted_forms(kwargs, accepted):
    with pytest.raises(OverrideError, match=accepted.replace("|", r"\|")):
        ModelOverrides(**kwargs)


def test_unknown_override_key():
    with pytest.raises(ValueError, match="unknown override `colour`"):
        ModelOverrides().set("colour", "red")


# --- vocabularies ----------------------------------------------------------


def test_vocabularies_are_hashable_and_distinct():
    for enum in (ConfigSource, ModelFamily, ModelTask, ConfigField):
        members = [getattr(enum, n) for n in dir(enum) if not n.startswith("_")]
        members = [m for m in members if isinstance(m, enum)]
        assert len({int(m) for m in members}) == len(members)
        assert len(set(members)) == len(members)


# --- content region on decode_semantic -------------------------------------


class _Region:
    """Stands in for ``edgefirst.image.ContentRegion``: any object with a
    normalized attribute is accepted."""

    def __init__(self, normalized):
        self.normalized = normalized


def test_decode_semantic_takes_a_content_region_object(sidecar_model: Path):
    cfg = resolve_model_config(sidecar_model, SEGFORMER_SIGNALS)
    decoder = Decoder(cfg.schema)
    logits = np.zeros((1, 150, 128, 128), dtype=np.float32)
    logits[0, 7, :, :64] = 1.0
    logits[0, 3, :, 64:] = 1.0
    scores = Tensor(list(logits.shape), dtype="float32")
    scores.from_numpy(logits)

    def run(**kwargs):
        dst = Tensor([64, 64], dtype="uint8")
        decoder.decode_semantic([scores], dst, **kwargs)
        with dst.map() as m:
            return np.frombuffer(m, dtype=np.uint8).reshape(64, 64).copy()

    left_quarter = (0.0, 0.0, 0.25, 1.0)
    by_tuple = run(content_region=left_quarter)
    by_object = run(content_region=_Region(left_quarter))
    np.testing.assert_array_equal(by_tuple, by_object)
    assert (by_tuple == 7).all()
    full = run()
    assert (full[:, :28] == 7).all() and (full[:, -28:] == 3).all()
    with pytest.raises(TypeError):
        decoder.decode_semantic(
            [scores], Tensor([4, 4], dtype="uint8"), letterbox=left_quarter
        )
