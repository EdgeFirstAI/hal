"""EdgeFirst decoder bindings.

Part of the PEP 420 `edgefirst.*` namespace. No package in the set ships an
`edgefirst/__init__.py`: a single one would shadow the namespace and break
every sibling.
"""

from typing import Protocol, Tuple

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


# `Protocol`/`Tuple` must not remain bound as module attributes: both
# tests/packaging/test_stub_parity.py (via `dir()`) and
# tests/packaging/test_namespace.py's `test_pyclass_modules_are_named` (via
# `vars()`, which `Protocol.__module__ == "typing"` would fail) treat every
# top-level name here as this package's public surface.
del Protocol, Tuple
