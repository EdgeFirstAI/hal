# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Value types crossing edgefirst.* package boundaries.

Covers the exports, equality, hashing, type-name gating and discriminants of
`PixelFormat`, `TensorMemory`, `Region`, `Colorimetry` and the colorimetry axis enums.
"""

import importlib

import pytest

PACKAGES = ("tensor", "codec", "image", "decoder")
VALUE_TYPES = ("PixelFormat", "TensorMemory", "Region", "Colorimetry")


@pytest.mark.parametrize("mod", PACKAGES)
@pytest.mark.parametrize("name", VALUE_TYPES)
def test_shared_value_types_exported_everywhere(mod, name):
    m = importlib.import_module(f"edgefirst.{mod}")
    assert hasattr(m, name), f"edgefirst.{mod} is missing {name}"


def test_python_discriminants_equal_the_shared_codes():
    """The Python `TensorMemory` discriminants equal the shared codes."""
    from edgefirst.tensor import TensorMemory

    assert int(TensorMemory.MEM) == 0
    assert int(TensorMemory.SHM) == 1
    assert int(TensorMemory.DMABUF) == 2
    assert int(TensorMemory.IOSURFACE) == 3
    assert int(TensorMemory.PBO) == 4
    assert int(TensorMemory.CUDA) == 5
    assert int(TensorMemory.D3D11_TEXTURE) == 6


def test_every_variant_exists_on_every_platform():
    """Every `TensorMemory` variant exists on every platform."""
    from edgefirst.tensor import TensorMemory

    for name in ("MEM", "SHM", "DMABUF", "IOSURFACE", "PBO", "CUDA", "D3D11_TEXTURE"):
        assert hasattr(TensorMemory, name), f"{name} must exist on every platform"


def test_enum_accepted_by_value_across_packages():
    """A sibling package's PixelFormat is accepted by create_image()."""
    from edgefirst.image import ImageProcessor
    from edgefirst.tensor import PixelFormat as TensorPixelFormat

    proc = ImageProcessor()
    t = proc.create_image(32, 32, TensorPixelFormat.Rgb, "uint8", "readwrite")
    assert t.memory is not None
    assert t.format == TensorPixelFormat.Rgb


def test_create_image_with_explicit_memory_backend_using_only_image_names():
    """create_image() takes a memory backend using only edgefirst.image names."""
    from edgefirst.image import PixelFormat, Tensor, TensorMemory

    t = Tensor.image(16, 16, PixelFormat.Rgb, mem=TensorMemory.MEM, access="readwrite")
    assert t.memory is not None
    assert int(t.memory) == int(TensorMemory.MEM)


def test_tensor_memory_accepted_by_value_across_packages():
    """A sibling package's TensorMemory is accepted by value."""
    from edgefirst.decoder import TensorMemory as DecoderTensorMemory
    from edgefirst.image import PixelFormat, Tensor

    t = Tensor.image(
        16, 16, PixelFormat.Rgb, mem=DecoderTensorMemory.MEM, access="readwrite"
    )
    assert int(t.memory) == int(DecoderTensorMemory.MEM)


def test_region_accepted_by_value_across_packages():
    """edgefirst.codec's Region is accepted by an edgefirst.image convert()."""
    from edgefirst.codec import Region as CodecRegion
    from edgefirst.image import ImageProcessor, PixelFormat, Tensor

    src = Tensor.image(64, 64, PixelFormat.Rgb, access="readwrite")
    proc = ImageProcessor()
    dst = proc.create_image(32, 32, PixelFormat.Rgb, access="readwrite")

    region = CodecRegion(0, 0, 32, 32)  # edgefirst.codec's Region
    proc.convert(src, dst, source=region)


def test_colorimetry_and_axis_enums_accepted_across_packages():
    """Colorimetry and each of its axis values cross packages in one round trip."""
    from edgefirst.image import PixelFormat, Tensor
    from edgefirst.tensor import Colorimetry as TensorColorimetry
    from edgefirst.tensor import ColorRange, ColorSpace

    src = TensorColorimetry(space=ColorSpace.Bt709, range=ColorRange.Full)
    t = Tensor.image(16, 16, PixelFormat.Rgb, access="readwrite")
    t.colorimetry = (
        src  # edgefirst.tensor's Colorimetry, set on an edgefirst.image Tensor
    )

    got = t.colorimetry
    assert got is not None
    assert got.space == ColorSpace.Bt709
    assert got.range == ColorRange.Full


def test_enum_equality_crosses_packages_both_directions():
    """A sibling package's copy of a variant compares equal from both sides."""
    from edgefirst.image import PixelFormat as ImagePixelFormat
    from edgefirst.image import TensorMemory as ImageTensorMemory
    from edgefirst.tensor import PixelFormat as TensorPixelFormat
    from edgefirst.tensor import TensorMemory as TensorTensorMemory

    assert TensorTensorMemory.DMABUF == ImageTensorMemory.DMABUF
    assert ImageTensorMemory.DMABUF == TensorTensorMemory.DMABUF
    assert TensorTensorMemory.DMABUF != ImageTensorMemory.MEM
    assert ImageTensorMemory.MEM != TensorTensorMemory.DMABUF

    assert TensorPixelFormat.Rgb == ImagePixelFormat.Rgb
    assert ImagePixelFormat.Rgb == TensorPixelFormat.Rgb
    assert TensorPixelFormat.Rgb != ImagePixelFormat.Rgba
    assert ImagePixelFormat.Rgba != TensorPixelFormat.Rgb


def test_colour_axis_enum_equality_crosses_packages():
    from edgefirst.decoder import ColorSpace as DecoderColorSpace
    from edgefirst.tensor import ColorSpace as TensorColorSpace

    assert TensorColorSpace.Bt709 == DecoderColorSpace.Bt709
    assert DecoderColorSpace.Bt709 == TensorColorSpace.Bt709
    assert TensorColorSpace.Bt709 != DecoderColorSpace.Srgb
    assert DecoderColorSpace.Srgb != TensorColorSpace.Bt709


def test_enum_still_compares_equal_to_its_bare_int_discriminant():
    """An enum compares equal to its bare int discriminant."""
    from edgefirst.tensor import PixelFormat

    assert PixelFormat.Rgb == 1
    assert PixelFormat.Rgb != 2


def test_enum_equality_against_unrelated_type_is_false_not_an_exception():
    """An unrelated type compares unequal without raising."""
    from edgefirst.tensor import TensorMemory

    assert (TensorMemory.DMABUF == "dma") is False
    assert (TensorMemory.DMABUF != "dma") is True
    assert (TensorMemory.DMABUF == 12345) is False  # no such discriminant
    assert (TensorMemory.DMABUF == object()) is False


def test_region_equality_crosses_packages_both_directions():
    """Regions with equal fields compare equal across packages from both sides."""
    from edgefirst.codec import Region as CodecRegion
    from edgefirst.tensor import Region as TensorRegion

    assert TensorRegion(0, 0, 32, 32) == CodecRegion(0, 0, 32, 32)
    assert CodecRegion(0, 0, 32, 32) == TensorRegion(0, 0, 32, 32)
    assert TensorRegion(0, 0, 32, 32) != CodecRegion(1, 0, 32, 32)
    assert CodecRegion(1, 0, 32, 32) != TensorRegion(0, 0, 32, 32)


def test_region_equality_against_unrelated_type_is_false_not_an_exception():
    from edgefirst.tensor import Region

    assert (Region(0, 0, 32, 32) == "not a region") is False
    assert (Region(0, 0, 32, 32) != "not a region") is True
    assert (Region(0, 0, 32, 32) == object()) is False


def test_colorimetry_has_no_equality_at_all_left_untouched():
    """Colorimetry implements no __eq__, same-package or cross-package."""
    from edgefirst.tensor import Colorimetry, ColorSpace

    assert Colorimetry.__eq__ is object.__eq__
    left = Colorimetry(space=ColorSpace.Bt709)
    right = Colorimetry(space=ColorSpace.Bt709)
    assert left is not right
    assert left != right


# --- __hash__: equal objects must hash equal --------------------------------


def test_enum_hashable_as_dict_key_and_set_member():
    from edgefirst.tensor import PixelFormat, TensorMemory

    d = {TensorMemory.DMABUF: "x", PixelFormat.Rgb: "y"}
    assert d[TensorMemory.DMABUF] == "x"
    assert {TensorMemory.DMABUF, TensorMemory.MEM, TensorMemory.DMABUF} == {
        TensorMemory.DMABUF,
        TensorMemory.MEM,
    }


def test_enum_cross_package_equal_values_hash_equal_and_set_collapses():
    """Equal cross-package enum values hash equal and collapse in a set."""
    from edgefirst.decoder import ColorSpace as DecoderColorSpace
    from edgefirst.image import PixelFormat as ImagePixelFormat
    from edgefirst.image import TensorMemory as ImageTensorMemory
    from edgefirst.tensor import ColorSpace as TensorColorSpace
    from edgefirst.tensor import PixelFormat as TensorPixelFormat
    from edgefirst.tensor import TensorMemory as TensorTensorMemory

    for a, b in (
        (TensorTensorMemory.DMABUF, ImageTensorMemory.DMABUF),
        (TensorPixelFormat.Rgb, ImagePixelFormat.Rgb),
        (TensorColorSpace.Bt709, DecoderColorSpace.Bt709),
    ):
        assert a == b
        assert hash(a) == hash(b)
        assert {a, b} == {a}
        assert len({a, b}) == 1


def test_enum_hash_matches_its_bare_int_discriminant():
    """`hash(enum)` equals the hash of its int discriminant."""
    from edgefirst.tensor import PixelFormat

    assert PixelFormat.Rgb == 1
    assert hash(PixelFormat.Rgb) == hash(1)


def test_enum_unequal_values_hash_differently():
    """Unequal enum values hash differently."""
    from edgefirst.tensor import ColorSpace, PixelFormat, TensorMemory

    assert hash(TensorMemory.DMABUF) != hash(TensorMemory.MEM)
    assert hash(PixelFormat.Rgb) != hash(PixelFormat.Rgba)
    assert hash(ColorSpace.Bt709) != hash(ColorSpace.Srgb)


def test_region_hashable_and_cross_package_equal_values_hash_equal():
    from edgefirst.codec import Region as CodecRegion
    from edgefirst.tensor import Region as TensorRegion

    a = TensorRegion(0, 0, 32, 32)
    b = CodecRegion(0, 0, 32, 32)
    assert a == b
    assert hash(a) == hash(b)
    assert {a, b} == {a}
    assert len({a, b}) == 1

    d = {a: "seen"}
    assert d[b] == "seen"  # lookup by the cross-package-equal key


def test_region_unequal_values_hash_differently():
    from edgefirst.tensor import Region

    assert hash(Region(0, 0, 32, 32)) != hash(Region(1, 0, 32, 32))


def test_colorimetry_hash_is_still_objects_identity_default():
    """Colorimetry's hash is object's identity-based default."""
    from edgefirst.tensor import Colorimetry

    assert Colorimetry.__hash__ is object.__hash__
    left = Colorimetry()
    right = Colorimetry()
    assert left is not right
    assert hash(left) != hash(right)


@pytest.mark.parametrize(
    "mod,name,members",
    [
        ("decoder", "Nms", ("ClassAgnostic", "ClassAware")),
        ("decoder", "DecoderType", ("Ultralytics", "ModelPack")),
        ("decoder", "DecoderVersion", ("Yolov5", "Yolov8")),
        ("decoder", "DimName", ("Batch", "Height")),
        ("image", "NormalizationPreset", ("Unsigned", "Signed")),
        ("image", "Interpolation", ("Bilinear", "Nearest")),
        ("image", "PadSpace", ("Pixel", "Normalized")),
        ("image", "EglDisplayKind", ("Gbm", "Default")),
        ("image", "Rotation", ("Rotate0", "Clockwise90")),
        ("image", "Flip", ("NoFlip", "Horizontal")),
        ("image", "ColorMode", ("Class", "Instance")),
        ("image", "Fit", ("Stretch", "Letterbox")),
        ("decoder", "MatchMetric", ("Iou", "Ios")),
        ("decoder", "MergeMode", ("KeepBest", "Union")),
        ("decoder", "ConfigSource", ("Heuristic", "Override")),
        ("decoder", "ModelFamily", ("EdgeFirst", "HuggingFace")),
        ("decoder", "ModelTask", ("Detect", "Semantic")),
        ("decoder", "ConfigField", ("Normalization", "Labels")),
    ],
)
def test_single_package_eq_int_enum_hashable(mod, name, members):
    """The single-package enums are hashable and distinct members do not collapse."""
    m = importlib.import_module(f"edgefirst.{mod}")
    cls = getattr(m, name)
    a = getattr(cls, members[0])
    b = getattr(cls, members[1])

    keyed = {a: "x", b: "y"}
    assert keyed[a] == "x"
    assert keyed[b] == "y"
    assert len(keyed) == 2
    assert len({a, b, a}) == 2
    assert hash(a) != hash(b)


# --- type confusion: two unrelated eq_int enums sharing a discriminant ------


def test_unrelated_enums_sharing_a_discriminant_are_not_equal():
    from edgefirst.tensor import PixelFormat, TensorMemory

    assert PixelFormat.Rgb.__int__() == TensorMemory.SHM.__int__() == 1  # the setup
    assert (PixelFormat.Rgb == TensorMemory.SHM) is False
    assert (PixelFormat.Rgb != TensorMemory.SHM) is True
    assert (TensorMemory.SHM == PixelFormat.Rgb) is False
    assert (TensorMemory.SHM != PixelFormat.Rgb) is True


def test_unrelated_colour_axis_enums_sharing_a_discriminant_are_not_equal():
    """Unrelated colour axis enums sharing a discriminant are not equal."""
    from edgefirst.tensor import ColorRange, ColorSpace

    assert ColorSpace.Bt709.__int__() == ColorRange.Full.__int__() == 0  # the setup
    assert (ColorSpace.Bt709 == ColorRange.Full) is False
    assert (ColorSpace.Bt709 != ColorRange.Full) is True


def test_unrelated_enum_dict_lookup_does_not_hit():
    from edgefirst.tensor import PixelFormat, TensorMemory

    d = {PixelFormat.Rgb: "fmt"}
    assert d.get(TensorMemory.SHM) is None


def test_unrelated_enum_rejected_in_argument_position():
    """A TensorMemory is rejected where create_image() expects a PixelFormat."""
    from edgefirst.image import ImageProcessor
    from edgefirst.tensor import TensorMemory

    proc = ImageProcessor()
    with pytest.raises(TypeError):
        proc.create_image(8, 8, TensorMemory.SHM, "uint8", "readwrite")


def test_bare_int_rejected_in_argument_position():
    """Argument extraction does not accept a bare int."""
    from edgefirst.image import ImageProcessor

    proc = ImageProcessor()
    with pytest.raises(TypeError):
        proc.create_image(8, 8, 1, "uint8", "readwrite")


def test_cross_package_equality_and_hashing():
    """The type-name gate accepts the legitimate cross-package case."""
    from edgefirst.image import ImageProcessor
    from edgefirst.image import PixelFormat as ImagePixelFormat
    from edgefirst.tensor import PixelFormat as TensorPixelFormat

    t = ImageProcessor().create_image(
        32, 32, TensorPixelFormat.Rgb, "uint8", "readwrite"
    )
    assert t.format == TensorPixelFormat.Rgb
    assert TensorPixelFormat.Rgb == ImagePixelFormat.Rgb
    assert hash(TensorPixelFormat.Rgb) == hash(ImagePixelFormat.Rgb)
    assert {TensorPixelFormat.Rgb, ImagePixelFormat.Rgb} == {TensorPixelFormat.Rgb}


def test_region_rejects_duck_typed_unrelated_object():
    """Region rejects an unrelated object that only exposes x/y/width/height."""
    from edgefirst.tensor import Region

    class NotARegion:
        x = 0
        y = 0
        width = 32
        height = 32

    fake = NotARegion()
    assert (Region(0, 0, 32, 32) == fake) is False
    assert (Region(0, 0, 32, 32) != fake) is True

    from edgefirst.image import ImageProcessor, PixelFormat, Tensor

    proc = ImageProcessor()
    src = Tensor.image(64, 64, PixelFormat.Rgb, access="readwrite")
    dst = proc.create_image(32, 32, PixelFormat.Rgb, access="readwrite")
    with pytest.raises(TypeError):
        proc.convert(src, dst, source=fake)


def test_region_cross_package():
    from edgefirst.codec import Region as CodecRegion
    from edgefirst.tensor import Region as TensorRegion

    a = TensorRegion(0, 0, 32, 32)
    b = CodecRegion(0, 0, 32, 32)
    assert a == b
    assert hash(a) == hash(b)
    assert {a, b} == {a}
