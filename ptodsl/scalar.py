from .api import scalar as _scalar
from .api.scalar import __all__


def __getattr__(name):
    return getattr(_scalar, name)
