# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Value types crossing edgefirst.* package boundaries (task 8 of the
cross-package handoff).

Earlier tasks fixed cross-package handoff for buffer-backed and opaque
types via a capsule protocol (`__edgefirst_tensor__`,
`__edgefirst_protodata__`, `__edgefirst_decoder__`). Enums and small value
structs -- `PixelFormat`, `TensorMemory`, `Region`, `Colorimetry` and the
four colorimetry axis enums -- are not buffers, so capsules are the wrong
tool. They cross packages instead by accepting the value structurally: a
`FromPyObject` impl that tries the native downcast first (same-package,
zero-cost), then falls back to reading the value back out of a sibling
package's copy of the same Python-level type (the `__int__()` discriminant
for `eq_int` enums, or the public getters for `Region`/`Colorimetry`).

This file also covers a second, independent defect: before this task
`edgefirst.image` and `edgefirst.codec` did not export `TensorMemory` or
`Colorimetry` *at all*, so there was no way to even name those types from
those packages, regardless of whether the identity problem above was fixed.

A third defect surfaced once the first two were fixed: `#[pyclass(eq_int)]`'s
auto-generated `__eq__`/`__ne__` only resolves `other` by native identity or
a bare int, so a sibling package's copy of the same enum compared *unequal*
-- silently, never an error. `tensor.memory != TensorMemory.DMABUF` returned
`True` for values that ARE equal. `PixelFormat`, `TensorMemory` and the four
colorimetry axis enums now get hand-written `__eq__`/`__ne__` that reuse the
same fallback as their `FromPyObject`. `Region` (a plain value struct, no
`eq_int` discriminant) had the identical bug via `#[pyclass(eq)]`'s
identity-only richcmp, fixed the same way by comparing field values instead.

`Colorimetry` does NOT get this treatment: it never implemented `__eq__` at
all (not same-package, not cross-package -- `Colorimetry() == Colorimetry()`
is `False` even within one package, via `object`'s identity default), so
there is nothing to fix without adding a new feature.

A fourth defect: all of the above were unhashable (`eq`/`eq_int` defines
`__eq__` without `__hash__`, and Python then sets `__hash__ = None`, mirroring
what a plain `class Foo: def __eq__(self, other): ...` does). `PixelFormat`,
`TensorMemory`, `Region` and the four colorimetry axis enums now hash the
same value their `__eq__` compares (the `__int__()` discriminant for the
enums, the field tuple for `Region`), so equal cross-package values hash
equal too -- required by Python's data model, and doubly so now that
equality is structural rather than identity-based. The single-package
`eq_int` enums the equality sweep also turned up (`Nms`, `DecoderType`,
`DecoderVersion`, `DimName`, `Normalization`, `EglDisplayKind`, `Rotation`,
`Flip`, `ColorMode`, `Fit`, `MatchMetric`) get the same discriminant-hash
treatment for the same reason, even though they have no cross-package
equality problem to fix (single-package -- no sibling copy is reachable).
`Colorimetry` is untouched here too: its hash is still `object`'s
identity-based default, consistent with its identity-based `__eq__`.

A fifth defect, a regression introduced by the cross-package fallback
itself: the `__int__()`/getattr fallback originally matched *any* object
exposing the right shape, not just a sibling package's copy of the *same*
type. Two unrelated `eq_int` enums sharing a discriminant (e.g.
`PixelFormat.Rgb` and `TensorMemory.SHM`, both `1`) compared equal, hashed
equal, and a `TensorMemory` was silently accepted anywhere a `PixelFormat`
was expected. `Region`'s getattr fallback had the analogous hole one level
up: pure structural duck typing accepted *any* object exposing
`x`/`y`/`width`/`height`, unrelated types included. Both fallbacks are now
gated on `type(obj).__name__` matching the expected type's name first --
every `edgefirst.*` package's copy of the same type shares the same
`#[pyclass(name = ...)]`, so this still accepts a sibling package's copy
while rejecting a same-shaped but unrelated type. This also closed an
accidental hole in argument extraction specifically: a bare Python `int`
used to be silently accepted wherever one of these enums was expected
(`int` has `__int__()` too); it no longer is. Equality's bare-int
comparison (`TensorMemory.MEM == 0`) is unaffected -- kept, but re-derived
from `Bound::extract::<i64>()` (`__index__`-based) rather than `__int__()`,
which a foreign enum still does not implement.

A sixth defect, closed by a later task in the same plan: this enum used to
have no explicit discriminants at all, plus a `#[cfg(unix)]` variant in the
middle of the list, so its numbering was a compile-time detail of the build
host (`TensorMemory.MEM` was `3` on unix, `2` elsewhere) and did not agree
with the C ABI's numbering for the same backings. `TensorMemory`'s
discriminants are now the shared codes from `edgefirst_tensor`'s
`TensorMemory::code()`, identical on every platform and every surface --
see `test_python_discriminants_equal_the_shared_codes` and
`test_every_variant_exists_on_every_platform` below.
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
    """The Python integer must BE the shared code, not a parallel numbering.

    Before this, `TensorMemory.MEM` was `3` while the C API's `PBO` was also
    `3`, so bridging the two surfaces by integer silently gave `PBO` where
    `MEM` was asked for. The value was also platform-dependent: a
    `#[cfg(unix)]` variant sat in the middle of an implicitly-numbered enum.

    This documents the intended values in a form readable from Python; it is
    NOT what enforces them. `PyTensorMemory`'s discriminants are still Rust
    literals (PyO3 requires one on a `#[pyclass]` enum -- there is no way to
    write `MEM = TensorMemory::Mem.code()`), so nothing here would catch the
    Rust and Python numbering drifting apart. The `const _: () = assert!(...)`
    items next to `PyTensorMemory`/`PyPixelFormat` in
    `crates/python-common/src/tensor.rs` are the actual guarantee -- they pin
    each literal to `TensorMemory::code()`/`PixelFormat::code()` at compile
    time, so a mismatch is a build failure, not a test result.
    """
    from edgefirst.tensor import TensorMemory

    assert int(TensorMemory.MEM) == 0
    assert int(TensorMemory.SHM) == 1
    assert int(TensorMemory.DMABUF) == 2
    assert int(TensorMemory.IOSURFACE) == 3
    assert int(TensorMemory.PBO) == 4
    assert int(TensorMemory.CUDA) == 5


def test_every_variant_exists_on_every_platform():
    """No variant may be cfg-gated away.

    A tensor recorded on Linux and replayed on macOS carries
    storage_kind = dmabuf. If the enum lacks the member on that build, the
    binding cannot report "cannot materialise here" -- it fails with an
    unknown code instead, turning a precise diagnostic into a corrupt-input
    error. Availability is a runtime question (the Rust
    `TensorMemory::is_available`), not a compile-time one.
    """
    from edgefirst.tensor import TensorMemory

    for name in ("MEM", "SHM", "DMABUF", "IOSURFACE", "PBO", "CUDA"):
        assert hasattr(TensorMemory, name), f"{name} must exist on every platform"


def test_enum_accepted_by_value_across_packages():
    """A sibling package's PixelFormat must be accepted structurally, not by
    identity: edgefirst.image never imports edgefirst.tensor's PixelFormat
    type object, so this only passes if create_image() falls back to the
    discriminant when the native downcast fails."""
    from edgefirst.image import ImageProcessor
    from edgefirst.tensor import PixelFormat as TensorPixelFormat

    proc = ImageProcessor()
    t = proc.create_image(32, 32, TensorPixelFormat.Rgb, "uint8", "readwrite")
    assert t.memory is not None
    assert t.format == TensorPixelFormat.Rgb


def test_create_image_with_explicit_memory_backend_using_only_image_names():
    """Regression test for the missing-export defect: before this task,
    edgefirst.image did not export TensorMemory at all, so an image tensor
    with an explicit memory backend was unconstructible using only names
    importable from edgefirst.image."""
    from edgefirst.image import PixelFormat, Tensor, TensorMemory

    t = Tensor.image(16, 16, PixelFormat.Rgb, mem=TensorMemory.MEM, access="readwrite")
    assert t.memory is not None
    assert int(t.memory) == int(TensorMemory.MEM)


def test_tensor_memory_accepted_by_value_across_packages():
    """Same identity story as PixelFormat above, but for TensorMemory --
    the enum defect (a)'s missing export was blocking in the first place."""
    from edgefirst.decoder import TensorMemory as DecoderTensorMemory
    from edgefirst.image import PixelFormat, Tensor

    t = Tensor.image(
        16, 16, PixelFormat.Rgb, mem=DecoderTensorMemory.MEM, access="readwrite"
    )
    assert int(t.memory) == int(DecoderTensorMemory.MEM)


def test_region_accepted_by_value_across_packages():
    """Region is a plain value struct (no discriminant), so the fallback
    reads its x/y/width/height getters back instead -- exercised here with
    edgefirst.codec's Region fed into an edgefirst.image convert()."""
    from edgefirst.codec import Region as CodecRegion
    from edgefirst.image import ImageProcessor, PixelFormat, Tensor

    src = Tensor.image(64, 64, PixelFormat.Rgb, access="readwrite")
    proc = ImageProcessor()
    dst = proc.create_image(32, 32, PixelFormat.Rgb, access="readwrite")

    region = CodecRegion(0, 0, 32, 32)  # edgefirst.codec's Region, not image's
    proc.convert(src, dst, source=region)


def test_colorimetry_and_axis_enums_accepted_across_packages():
    """Colorimetry itself crosses packages (via its four axis getters), and
    each axis value crosses independently too (via its own __int__()
    fallback) -- both exercised in one round trip."""
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
    """A sibling package's copy of the same variant must compare equal, and
    a different variant unequal -- checked from both sides, since a naive
    __eq__ override could end up asymmetric (Python does not require
    a.__eq__(b) and b.__eq__(a) to agree unless both are implemented)."""
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
    """`eq_int` already allowed comparing an enum to a plain int; the
    hand-written `__eq__`/`__ne__` must not regress that (`int.__int__()` is
    the identity function, so the same discriminant fallback covers it)."""
    from edgefirst.tensor import PixelFormat

    assert PixelFormat.Rgb == 1
    assert PixelFormat.Rgb != 2


def test_enum_equality_against_unrelated_type_is_false_not_an_exception():
    """An unrelated type must compare unequal, not raise -- NotImplemented
    on both sides falls back to Python's identity-based default."""
    from edgefirst.tensor import TensorMemory

    assert (TensorMemory.DMABUF == "dma") is False
    assert (TensorMemory.DMABUF != "dma") is True
    assert (TensorMemory.DMABUF == 12345) is False  # no such discriminant
    assert (TensorMemory.DMABUF == object()) is False


def test_region_equality_crosses_packages_both_directions():
    """Same story as the enums, but Region is a value struct: equal field
    values must compare equal cross-package, unequal values unequal --
    checked from both sides."""
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
    """Documents rather than exercises a fix: Colorimetry never implemented
    __eq__ (identity-based object default), same-package or cross-package.
    Adding equality where none existed is a new feature, not a fix -- left
    alone deliberately, see the module docstring."""
    from edgefirst.tensor import Colorimetry, ColorSpace

    assert Colorimetry.__eq__ is object.__eq__
    assert Colorimetry(space=ColorSpace.Bt709) != Colorimetry(space=ColorSpace.Bt709)


# --- __hash__: equal objects must hash equal --------------------------------
#
# eq/eq_int defines __eq__ without __hash__, so Python sets __hash__ = None
# (the same rule as a plain `class Foo: def __eq__(self, other): ...`).
# PixelFormat/TensorMemory/Region/the colour axis enums now hash the exact
# value their __eq__ compares, so this isn't just "hashable" -- a
# package-specific hash for a cross-package-equal value would be worse than
# unhashable (a set/dict lookup that silently misses), so the tests below
# check the actual hash values agree, not just that hash() doesn't raise.


def test_enum_hashable_as_dict_key_and_set_member():
    from edgefirst.tensor import PixelFormat, TensorMemory

    d = {TensorMemory.DMABUF: "x", PixelFormat.Rgb: "y"}
    assert d[TensorMemory.DMABUF] == "x"
    assert {TensorMemory.DMABUF, TensorMemory.MEM, TensorMemory.DMABUF} == {
        TensorMemory.DMABUF,
        TensorMemory.MEM,
    }


def test_enum_cross_package_equal_values_hash_equal_and_set_collapses():
    """The invariant that actually matters: a package-specific hash for an
    equal cross-package value would silently break dict/set lookups even
    though `==` reports them equal."""
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
    """eq_int already allowed `enum == int`; the data model then requires
    `hash(enum) == hash(int)` too."""
    from edgefirst.tensor import PixelFormat

    assert PixelFormat.Rgb == 1
    assert hash(PixelFormat.Rgb) == hash(1)


def test_enum_unequal_values_hash_differently():
    """Not required by the data model, but rules out a constant-hash
    implementation that would trivially pass the equal-hashes-equal checks
    above."""
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
    """Documents rather than exercises a fix: Colorimetry's hash is
    untouched -- still object's identity-based default, consistent with its
    identity-based __eq__ (see test_colorimetry_has_no_equality_at_all_left_untouched).
    Two equal-content-but-distinct instances are NOT required to hash equal
    here because they are not required to compare equal either."""
    from edgefirst.tensor import Colorimetry

    assert Colorimetry.__hash__ is object.__hash__
    assert hash(Colorimetry()) != hash(Colorimetry())


@pytest.mark.parametrize(
    "mod,name,members",
    [
        ("decoder", "Nms", ("ClassAgnostic", "ClassAware")),
        ("decoder", "DecoderType", ("Ultralytics", "ModelPack")),
        ("decoder", "DecoderVersion", ("Yolov5", "Yolov8")),
        ("decoder", "DimName", ("Batch", "Height")),
        ("image", "Normalization", ("DEFAULT", "SIGNED")),
        ("image", "EglDisplayKind", ("Gbm", "Default")),
        ("image", "Rotation", ("Rotate0", "Clockwise90")),
        ("image", "Flip", ("NoFlip", "Horizontal")),
        ("image", "ColorMode", ("Class", "Instance")),
        ("image", "Fit", ("Stretch", "Letterbox")),
        ("decoder", "MatchMetric", ("Iou", "Ios")),
    ],
)
def test_single_package_eq_int_enum_hashable(mod, name, members):
    """These carry the identical unhashable limitation as the cross-package
    enums above (eq_int defines __eq__ without __hash__), but are registered
    in exactly one package (see the sweep in the module docstring), so there
    is no cross-package equal-hash invariant to check -- only that they are
    hashable at all now, and that distinct members do not collapse."""
    m = importlib.import_module(f"edgefirst.{mod}")
    cls = getattr(m, name)
    a = getattr(cls, members[0])
    b = getattr(cls, members[1])

    assert {a: "x", b: "y"} == {a: "x", b: "y"}
    assert len({a, b, a}) == 2
    assert hash(a) != hash(b)


# --- type confusion: two unrelated eq_int enums sharing a discriminant ------
#
# The regression the whole-branch review caught: the cross-package fallback
# originally matched *any* object with the right shape (an __int__() for the
# enums, x/y/width/height for Region), not just a sibling package's copy of
# the SAME type. PixelFormat.Rgb and TensorMemory.SHM are both discriminant 1
# -- before the type-name gate, they compared equal, hashed equal, and a
# TensorMemory was silently accepted anywhere a PixelFormat was expected.


def test_unrelated_enums_sharing_a_discriminant_are_not_equal():
    from edgefirst.tensor import PixelFormat, TensorMemory

    assert PixelFormat.Rgb.__int__() == TensorMemory.SHM.__int__() == 1  # the setup
    assert (PixelFormat.Rgb == TensorMemory.SHM) is False
    assert (PixelFormat.Rgb != TensorMemory.SHM) is True
    assert (TensorMemory.SHM == PixelFormat.Rgb) is False
    assert (TensorMemory.SHM != PixelFormat.Rgb) is True


def test_unrelated_colour_axis_enums_sharing_a_discriminant_are_not_equal():
    """Same story, colour axis enums -- these all have low discriminants,
    so this is the likeliest place for it to bite in practice."""
    from edgefirst.tensor import ColorRange, ColorSpace

    assert ColorSpace.Bt709.__int__() == ColorRange.Full.__int__() == 0  # the setup
    assert (ColorSpace.Bt709 == ColorRange.Full) is False
    assert (ColorSpace.Bt709 != ColorRange.Full) is True


def test_unrelated_enum_dict_lookup_does_not_hit():
    from edgefirst.tensor import PixelFormat, TensorMemory

    d = {PixelFormat.Rgb: "fmt"}
    assert d.get(TensorMemory.SHM) is None


def test_unrelated_enum_rejected_in_argument_position():
    """The type-confusion bug's most concrete form: a TensorMemory silently
    accepted where an ImageProcessor.create_image() format argument
    (PixelFormat) belongs."""
    from edgefirst.image import ImageProcessor
    from edgefirst.tensor import TensorMemory

    with pytest.raises(TypeError):
        ImageProcessor().create_image(8, 8, TensorMemory.SHM, "uint8", "readwrite")


def test_bare_int_no_longer_accepted_in_argument_position():
    """Argument extraction never accepted a bare int as a matter of policy;
    the type-name gate closes an accidental hole where `__int__()` let one
    through (unlike equality, which keeps its bare-int comparison -- see
    test_enum_still_compares_equal_to_its_bare_int_discriminant)."""
    from edgefirst.image import ImageProcessor

    with pytest.raises(TypeError):
        ImageProcessor().create_image(8, 8, 1, "uint8", "readwrite")


def test_cross_package_equality_and_hashing_still_work_after_the_fix():
    """The type-name gate must not regress the legitimate cross-package
    case it exists alongside -- re-checked here after the fix."""
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
    """Region's getattr fallback had the analogous hole one level up: pure
    structural duck typing accepted any object exposing
    x/y/width/height, unrelated types included."""
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

    src = Tensor.image(64, 64, PixelFormat.Rgb, access="readwrite")
    dst = ImageProcessor().create_image(32, 32, PixelFormat.Rgb, access="readwrite")
    # Setup lives outside the `raises` block deliberately: if Tensor.image()
    # or create_image() ever raised TypeError for an unrelated reason, this
    # test would pass while never exercising the type-confusion gate it
    # exists to guard -- only the call under test belongs inside `raises`.
    with pytest.raises(TypeError):
        ImageProcessor().convert(src, dst, source=fake)


def test_region_cross_package_still_works_after_the_fix():
    from edgefirst.codec import Region as CodecRegion
    from edgefirst.tensor import Region as TensorRegion

    a = TensorRegion(0, 0, 32, 32)
    b = CodecRegion(0, 0, 32, 32)
    assert a == b
    assert hash(a) == hash(b)
    assert {a, b} == {a}
