from .api import tile as _tile
from .api.tile import __all__


def __getattr__(name):
    return getattr(_tile, name)
