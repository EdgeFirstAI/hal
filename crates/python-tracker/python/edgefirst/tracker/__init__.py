"""EdgeFirst tracker bindings.

Part of the PEP 420 `edgefirst.*` namespace. No package in the set ships an
`edgefirst/__init__.py`: a single one would shadow the namespace and break
every sibling.
"""

from ._tracker import *
from ._tracker import __doc__  # noqa: F401
