"""EdgeFirst codec bindings.

Part of the PEP 420 `edgefirst.*` namespace. No package in the set ships an
`edgefirst/__init__.py`: a single one would shadow the namespace and break
every sibling.
"""

import os
import sys

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    # Python 3.8+ loads extension modules with LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
    # | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR: the .pyd's own directory and the
    # directories registered via os.add_dll_directory() are searched, PATH is
    # not. _codec.pyd lives here but edgefirst_tensor.dll ships in the sibling
    # edgefirst/tensor/ directory, so register that directory before the
    # import below loads the extension. The returned handle is dropped on
    # purpose: the registration lasts for the process, only close() undoes it.
    import edgefirst.tensor as _tensor_pkg

    os.add_dll_directory(os.path.dirname(_tensor_pkg.__file__))
    del _tensor_pkg

from edgefirst.tensor import EdgeFirstTensorExportable as EdgeFirstTensorExportable

from ._codec import *
from ._codec import __doc__  # noqa: F401

# `os`/`sys` must not remain bound as module attributes: both
# tests/packaging/test_stub_parity.py (via `dir()`) and
# tests/packaging/test_namespace.py's `test_pyclass_modules_are_named` (via
# `vars()`) treat every top-level name here as this package's public surface.
del os, sys
