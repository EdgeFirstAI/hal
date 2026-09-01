"""EdgeFirst decoder bindings.

Part of the PEP 420 `edgefirst.*` namespace. No package in the set ships an
`edgefirst/__init__.py`: a single one would shadow the namespace and break
every sibling.
"""

import os
import sys
from typing import Any, Dict, List, NamedTuple, Protocol, Tuple

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
from ._decoder import __doc__  # noqa: F401


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


# `os`/`sys` and the `typing` names must not remain bound as module
# attributes: both tests/packaging/test_stub_parity.py (via `dir()`) and
# tests/packaging/test_namespace.py's `test_pyclass_modules_are_named` (via
# `vars()`, which `Protocol.__module__ == "typing"` would fail) treat every
# top-level name here as this package's public surface.
del Any, Dict, List, NamedTuple, os, Protocol, sys, Tuple
