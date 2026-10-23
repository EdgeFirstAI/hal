"""EdgeFirst decoder bindings.

Part of the PEP 420 `edgefirst.*` namespace. No package in the set ships an
`edgefirst/__init__.py`: a single one would shadow the namespace and break
every sibling.
"""

import dataclasses
import os
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Protocol, Sequence, Tuple

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    # Python 3.8+ loads extension modules with LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
    # | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR: the .pyd's own directory and the
    # directories registered via os.add_dll_directory() are searched, PATH is
    # not. _decoder.pyd lives here but edgefirst_tensor.dll ships in the sibling
    # edgefirst/tensor/ directory, so register that directory before the
    # import below loads the extension. The handle stays bound for the life
    # of the process (CPython's handle has no __del__, so dropping it would
    # not undo the registration, but holding it keeps the lifetime explicit
    # and remove_dll_directory() possible); the leading underscore keeps it
    # out of the public surface checked by tests/packaging.
    import edgefirst.tensor as _tensor_pkg

    _tensor_dll_dir = os.add_dll_directory(os.path.dirname(_tensor_pkg.__file__))
    del _tensor_pkg

from edgefirst.tensor import EdgeFirstTensorExportable as EdgeFirstTensorExportable

from ._decoder import *
from ._decoder import (
    __doc__,  # noqa: F401
    __tensor_backend__,  # noqa: F401
)


class EdgeFirstDecoderExportable(Protocol):
    """Structural type for anything that can hand a ``Decoder`` across an
    ``edgefirst.*`` package boundary via the ``__edgefirst_decoder__``
    capsule protocol. See
    ``crates/python-common/INTEROP.md`` in the
    `hal <https://github.com/EdgeFirstAI/hal>`_ repository.
    """

    def __edgefirst_decoder__(self) -> object: ...


class EdgeFirstProtoDataExportable(Protocol):
    """Structural type for anything that can hand mask-prototype data
    across an ``edgefirst.*`` package boundary via the
    ``__edgefirst_protodata__`` capsule protocol. See
    ``crates/python-common/INTEROP.md`` in the
    `hal <https://github.com/EdgeFirstAI/hal>`_ repository.
    """

    # `from __future__ import annotations` is deliberately not used in this
    # module -- see the `del` below.
    def __edgefirst_protodata__(self) -> Tuple[object, object, str]: ...  # noqa: FA100


class HasNormalizedRegion(Protocol):
    """Structural type for a ``content_region=`` argument given as an object,
    such as ``edgefirst.image.ContentRegion``: anything with a normalized
    ``(xmin, ymin, xmax, ymax)`` ``normalized`` attribute.
    """

    @property
    def normalized(self) -> Tuple[float, float, float, float]: ...  # noqa: FA100


class InferredSchema(NamedTuple):
    """What :func:`infer_ultralytics_schema` derived from a model's signals.

    A :class:`typing.NamedTuple`, so it unpacks positionally like a plain
    tuple while giving the three fields names -- two of them are strings,
    which positional unpacking alone cannot keep straight.
    """

    #: The inferred ``edgefirst.json`` schema v2 document, ready to hand to
    #: ``Decoder(schema)``.
    schema: Dict[str, Any]  # noqa: FA100
    #: Class names in index order. ``decode()`` returns class indices into
    #: this list.
    labels: List[str]  # noqa: FA100
    #: Human-readable summary, e.g. ``"Ultralytics YOLOv8/11 detect, 80
    #: classes"``.
    description: str


class ModelSignals(NamedTuple):
    """A model's inputs, outputs and metadata, as a runtime reports them.

    Passed to :func:`resolve_model_config` to build a schema from Hugging
    Face configs, Ultralytics metadata or input/output names. The fields
    take the same forms as :func:`infer_ultralytics_schema`'s arguments.
    """

    #: Container format: ``"onnx"``, ``"tflite"``, ``"coreml"`` or
    #: ``"other"``.
    source: str
    #: Input tensors as ``(name, shape, dtype)`` or
    #: ``(name, shape, dtype, quantization)``.
    inputs: Sequence[Tuple[Any, ...]]  # noqa: FA100
    #: Output tensors as ``(name, shape, dtype)`` or
    #: ``(name, shape, dtype, (scales, zero_points))``.
    outputs: Sequence[Tuple[Any, ...]]  # noqa: FA100
    #: Raw model metadata key/values, verbatim; ``None`` for none.
    metadata: Optional[Dict[str, str]] = None  # noqa: FA100


@dataclasses.dataclass(frozen=True)
class PreprocessSpec:
    """A model's input preprocessing as plain values.

    Returned by :attr:`ResolvedModelConfig.preprocessing`; consumed by
    ``edgefirst.image.ConvertOptions.from_spec``, which reads these
    attributes by name. The defaults are the ``edgefirst.json`` defaults:
    unsigned normalization, a centred letterbox padded with pixel 114,
    bilinear.
    """

    #: ``(mean, std)``, each an RGB triple on the 0-255 scale; the image is
    #: mapped as ``(x - mean) / std``.
    normalization: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (  # noqa: FA100
        (0.0, 0.0, 0.0),
        (255.0, 255.0, 255.0),
    )
    #: ``"stretch"`` or ``"letterbox"``.
    fit: str = "letterbox"
    #: Letterbox placement: ``"center"`` or ``"topleft"``.
    placement: str = "center"
    #: Letterbox pad per RGB channel, in the space :attr:`pad_space` names.
    pad: Tuple[float, float, float] = (114.0, 114.0, 114.0)  # noqa: FA100
    #: ``"pixel"`` (0-255, normalized like the image) or ``"normalized"``
    #: (written as-is).
    pad_space: str = "pixel"
    #: ``"bilinear"``, ``"nearest"``, ``"bicubic"`` or ``"area"``.
    interpolation: str = "bilinear"
    #: Model input ``(width, height)``, when known.
    size: Optional[Tuple[int, int]] = None  # noqa: FA100

    @property
    def mean(self) -> Tuple[float, float, float]:  # noqa: FA100
        """The per-channel mean, RGB order, on the 0-255 scale."""
        return self.normalization[0]

    @property
    def std(self) -> Tuple[float, float, float]:  # noqa: FA100
        """The per-channel standard deviation, RGB order, on the 0-255 scale."""
        return self.normalization[1]


class ModelConfigError(ValueError):
    """A model configuration that is malformed or cannot be resolved."""


class OverrideError(ModelConfigError):
    """An override outside its grammar or one that does not apply."""


class LabelCountMismatchError(OverrideError):
    """A ``labels`` override naming a different number of classes than the
    model has: ``expected`` is the model's class count, ``found`` the number
    of names given."""

    def __init__(self, message: str, expected: int, found: int) -> None:
        super().__init__(message)
        self.expected = expected
        self.found = found


class UnresolvedConfigError(ModelConfigError):
    """Nothing supplies a required piece of the configuration."""


class UnsupportedModelError(ModelConfigError, NotImplementedError):
    """A configuration that asks for something the library cannot run."""


# `os`/`sys`, `dataclasses` and the `typing` names must not remain bound as
# module attributes: both tests/packaging/test_stub_parity.py (via `dir()`)
# and tests/packaging/test_namespace.py's `test_pyclass_modules_are_named`
# (via `vars()`, which `Protocol.__module__ == "typing"` would fail) treat
# every top-level name here as this package's public surface.
del (
    Any,
    dataclasses,
    Dict,
    List,
    NamedTuple,
    Optional,
    os,
    Protocol,
    Sequence,
    sys,
    Tuple,
)
