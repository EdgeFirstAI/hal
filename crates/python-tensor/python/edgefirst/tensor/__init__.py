"""EdgeFirst tensor bindings.

Part of the PEP 420 `edgefirst.*` namespace.
"""

from typing import Optional, Protocol

from ._tensor import *
from ._tensor import (
    __doc__,  # noqa: F401
    __tensor_backend__,  # noqa: F401
)


class EdgeFirstTensorExportable(Protocol):
    """Structural type for anything that can hand a tensor across an
    ``edgefirst.*`` package boundary via the ``__edgefirst_tensor__``
    capsule protocol.

    Every `edgefirst.*` extension module registers its own ``Tensor`` type
    object, so consumers accept this protocol by duck typing
    (``hasattr(obj, "__edgefirst_tensor__")``). See
    ``crates/python-common/INTEROP.md`` in the
    `hal <https://github.com/EdgeFirstAI/hal>`_ repository for the full
    protocol.
    """

    def __edgefirst_tensor__(self, access: Optional[str] = None) -> object: ...  # noqa: FA100


# Remove the helper imports from the module's public names.
del Optional, Protocol
