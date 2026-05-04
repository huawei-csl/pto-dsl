from contextlib import contextmanager

from mlir.dialects import scf
from mlir.ir import InsertionPoint

from .scalar import Value, _unwrap
from ..utils.codegen import get_user_code_loc


def range(start, stop, step):
    with get_user_code_loc():
        loop = scf.ForOp(_unwrap(start), _unwrap(stop), _unwrap(step))
    with InsertionPoint(loop.body):
        yield Value(loop.induction_variable)
        scf.YieldOp([])


class _IfElseBranch:
    def __init__(self, if_op):
        self._if_op = if_op

    @contextmanager
    def else_context(self):
        with InsertionPoint(self._if_op.else_block):
            yield
            scf.YieldOp([])


@contextmanager
def if_context(condition, has_else=False):
    if has_else:
        with get_user_code_loc():
            op = scf.IfOp(_unwrap(condition), [], hasElse=True)
        branch = _IfElseBranch(op)
    else:
        with get_user_code_loc():
            op = scf.IfOp(_unwrap(condition))
        branch = None

    with InsertionPoint(op.then_block):
        yield branch
        scf.YieldOp([])


def cond(condition, then_builder, else_builder):
    with get_user_code_loc():
        op = scf.IfOp(_unwrap(condition), [], hasElse=True)
    with InsertionPoint(op.then_block):
        then_builder()
        scf.YieldOp([])
    with InsertionPoint(op.else_block):
        else_builder()
        scf.YieldOp([])
    return op


__all__ = ["cond", "range", "if_context"]
