"""EdgeFirst image bindings.

Part of the PEP 420 `edgefirst.*` namespace. No package in the set ships an
`edgefirst/__init__.py`: a single one would shadow the namespace and break
every sibling.
"""

from typing import Protocol

from edgefirst.tensor import EdgeFirstTensorExportable as EdgeFirstTensorExportable

from ._image import *  # noqa: F403
from ._image import __doc__  # noqa: F401


class EdgeFirstProtoDataExportable(Protocol):
    """Structural type for mask-prototype data via ``__edgefirst_protodata__``."""

    def __edgefirst_protodata__(self) -> tuple: ...


del Protocol
