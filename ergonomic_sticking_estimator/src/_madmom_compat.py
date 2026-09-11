"""Compatibility shim for madmom 0.16.1 on modern Python/numpy.

madmom 0.16.1 (2018, last release) predates two breaking changes:
  - Python 3.10 moved the ABCs out of `collections` into `collections.abc`.
  - numpy 1.24 removed the deprecated `np.float`/`np.int`/... aliases.

Importing this module restores the removed names so madmom can import.
Must be imported *before* `import madmom`.
"""

import collections
import collections.abc

# Python 3.10+: restore ABCs that madmom imports from `collections`.
for _name in ("MutableSequence", "MutableMapping", "Mapping", "Sequence",
              "Iterable", "Callable", "MutableSet"):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

# numpy 1.24+: restore the removed scalar-type aliases madmom still references.
# Only `np.float`/`np.int` were fully removed; others may still exist (deprecated),
# so probe quietly and patch only the names that actually raise AttributeError.
import warnings
import numpy as np
for _alias, _builtin in (("float", float), ("int", int), ("object", object),
                         ("complex", complex)):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            getattr(np, _alias)
    except AttributeError:
        setattr(np, _alias, _builtin)
