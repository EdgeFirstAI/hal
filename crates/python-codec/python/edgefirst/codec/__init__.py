"""EdgeFirst codec bindings.

Part of the PEP 420 `edgefirst.*` namespace.
"""

import os
import sys

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    # Register the sibling edgefirst/tensor/ directory, which holds
    # edgefirst_tensor.dll, before the import below loads the extension.
    import edgefirst.tensor as _tensor_pkg

    _tensor_dll_dir = os.add_dll_directory(os.path.dirname(_tensor_pkg.__file__))
    del _tensor_pkg

from edgefirst.tensor import EdgeFirstTensorExportable as EdgeFirstTensorExportable

from ._codec import *
from ._codec import (
    __doc__,  # noqa: F401
    __tensor_backend__,  # noqa: F401
)

# Remove the helper imports from the module's public names.
del os, sys
