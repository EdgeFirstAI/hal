"""EdgeFirst kernels bindings.

Part of the PEP 420 `edgefirst.*` namespace. No package in the set ships an
`edgefirst/__init__.py`: a single one would shadow the namespace and break
every sibling.
"""

from ._kernels import *
from ._kernels import __doc__  # noqa: F401
