"""EdgeFirst codec bindings.

Part of the PEP 420 `edgefirst.*` namespace. No package in the set ships an
`edgefirst/__init__.py`: a single one would shadow the namespace and break
every sibling.
"""

from edgefirst.tensor import EdgeFirstTensorExportable as EdgeFirstTensorExportable

from ._codec import *
from ._codec import __doc__  # noqa: F401
