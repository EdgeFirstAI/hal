"""EdgeFirst image bindings.

Part of the PEP 420 `edgefirst.*` namespace.
"""

import os
import sys
from typing import Protocol

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    # Register the sibling edgefirst/tensor/ directory, which holds
    # edgefirst_tensor.dll, before the import below loads the extension.
    import edgefirst.tensor as _tensor_pkg

    _tensor_dll_dir = os.add_dll_directory(os.path.dirname(_tensor_pkg.__file__))
    del _tensor_pkg

from edgefirst.tensor import EdgeFirstTensorExportable as EdgeFirstTensorExportable

from ._image import *
from ._image import (
    __doc__,  # noqa: F401
    __tensor_backend__,  # noqa: F401
)


class EdgeFirstProtoDataExportable(Protocol):
    """Structural type for mask-prototype data via ``__edgefirst_protodata__``."""

    def __edgefirst_protodata__(self) -> tuple: ...


class PreprocessSpecLike(Protocol):
    """Structural type for :meth:`ConvertOptions.from_spec`: anything with
    all of the attributes of ``edgefirst.decoder.PreprocessSpec``.
    ``normalization`` may instead be given as ``mean`` and ``std``.
    """

    @property
    def normalization(self) -> object: ...
    @property
    def fit(self) -> object: ...
    @property
    def placement(self) -> object: ...
    @property
    def pad(self) -> object: ...
    @property
    def pad_space(self) -> object: ...
    @property
    def interpolation(self) -> object: ...
    @property
    def size(self) -> object: ...


# Remove the helper imports from the module's public names.
del os, sys, Protocol
