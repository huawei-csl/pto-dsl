# Adapted from https://github.com/llvm/eudsl/blob/main/projects/eudsl-python-extras/mlir/extras/util.py

import sys
import inspect
import functools
from pathlib import Path
from typing import Optional

from mlir.ir import Context, Location


def is_relative_to(self, other):
    return other == self or other in self.parents


def get_user_code_loc(user_base: Optional[Path] = None):
    from .. import utils

    if Context.current is None:
        return

    mlir_extras_root_path = Path(utils.__path__[0])

    prev_frame = inspect.currentframe().f_back
    if user_base is None:
        user_base = Path(prev_frame.f_code.co_filename)

    while prev_frame.f_back and (
        is_relative_to(Path(prev_frame.f_code.co_filename), mlir_extras_root_path)
        or is_relative_to(Path(prev_frame.f_code.co_filename), Path(sys.prefix))
        or is_relative_to(Path(prev_frame.f_code.co_filename), user_base)
    ):
        prev_frame = prev_frame.f_back
    frame_info = inspect.getframeinfo(prev_frame)
    if sys.version_info.minor >= 11:
        return Location.file(
            frame_info.filename, frame_info.lineno, frame_info.positions.col_offset
        )
    else:
        return Location.file(frame_info.filename, frame_info.lineno, col=0)


def with_loc(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        loc = get_user_code_loc()
        if loc is None:
            return fn(*args, **kwargs)
        with loc:
            return fn(*args, **kwargs)

    return wrapper


def apply_loc(cls):
    for name, val in vars(cls).items():
        if name == "__init__":
            continue
        if isinstance(val, staticmethod):
            setattr(cls, name, staticmethod(with_loc(val.__func__)))
        elif callable(val):
            setattr(cls, name, with_loc(val))
    return cls
