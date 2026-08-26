"""EdgeFirst tensor bindings.

Part of the PEP 420 `edgefirst.*` namespace. No package in the set ships an
`edgefirst/__init__.py`: a single one would shadow the namespace and break
every sibling.
"""

from typing import Optional, Protocol

from ._tensor import *  # noqa: F403
from ._tensor import __doc__  # noqa: F401


class EdgeFirstTensorExportable(Protocol):
    """Structural type for anything that can hand a tensor across an
    ``edgefirst.*`` package boundary via the ``__edgefirst_tensor__``
    capsule protocol.

    Every `edgefirst.*` extension module registers its own ``Tensor`` type
    object (see PyO3 issue #1444), so ``isinstance`` cannot recognise a
    tensor produced by a sibling package. Consumers accept this protocol by
    duck typing (``hasattr(obj, "__edgefirst_tensor__")``) instead. See
    ``crates/python-common/INTEROP.md`` in the
    `hal <https://github.com/EdgeFirstAI/hal>`_ repository for the full
    protocol, including the capsule's lifetime and versioning contract.
    """

    # `from __future__ import annotations` is deliberately not used in this
    # module -- see the `del` below.
    def __edgefirst_tensor__(self, access: Optional[str] = None) -> object: ...  # noqa: FA100


# `Optional`/`Protocol` must not remain bound as module attributes: both
# tests/packaging/test_stub_parity.py (via `dir()`) and
# tests/packaging/test_namespace.py's `test_pyclass_modules_are_named` (via
# `vars()`, which `Protocol.__module__ == "typing"` would fail) treat every
# top-level name here as this package's public surface.
del Optional, Protocol
