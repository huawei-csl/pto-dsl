import ast
import inspect
import textwrap
from contextlib import ExitStack

from mlir.dialects import arith, func, scf
from mlir.ir import InsertionPoint

from ..api.scalar import Value, _unwrap, const, wrap_value


class PTODSLAstError(ValueError):
    pass


_STATIC_RANGE_UNROLL_LIMIT = 256


def _unwrap_if_needed(value):
    return _unwrap(value)


def _is_runtime_value(value):
    raw = _unwrap_if_needed(value)
    return hasattr(raw, "type")


def _ensure_wrapped(value):
    return wrap_value(value) if _is_runtime_value(value) else value


def _as_loop_bound(value):
    return value if _is_runtime_value(value) else const(value)


def _expr_filename(fn):
    try:
        return inspect.getsourcefile(fn) or inspect.getfile(fn)
    except Exception:
        return f"<ptodsl:{fn.__name__}>"


def get_function_def(fn):
    source = textwrap.dedent(inspect.getsource(fn))
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn.__name__:
            return node
    raise PTODSLAstError(f"Could not locate AST for function '{fn.__name__}'.")


def _is_range_call(node):
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id == "range"
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        return node.func.value.id == "pto" and node.func.attr == "range"
    return False


def _extract_target_names(target):
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names = []
        for elt in target.elts:
            names.extend(_extract_target_names(elt))
        return names
    return []


def _assigned_names(statements):
    names = set()
    for stmt in statements:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    names.update(_extract_target_names(target))
            elif isinstance(node, ast.AnnAssign):
                names.update(_extract_target_names(node.target))
            elif isinstance(node, ast.AugAssign):
                names.update(_extract_target_names(node.target))
    return names


class AstLowerer:
    def __init__(self, fn, env):
        self.fn = fn
        self.filename = _expr_filename(fn)
        self.global_env = dict(fn.__globals__)
        try:
            closure_vars = inspect.getclosurevars(fn)
        except Exception:
            closure_vars = None
        if closure_vars is not None:
            self.global_env.update(closure_vars.globals)
            self.global_env.update(closure_vars.nonlocals)
        self.global_env.update(env)

    def eval_expr(self, node, env):
        unsupported_expr_nodes = (
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.Yield,
            ast.YieldFrom,
            ast.Lambda,
        )
        for child in ast.walk(node):
            if isinstance(child, unsupported_expr_nodes):
                raise PTODSLAstError(
                    f"Unsupported Python expression in PTODSL AST frontend: {type(child).__name__}"
                )
        if isinstance(node, ast.BoolOp):
            values = [self.eval_expr(v, env) for v in node.values]
            if all(not _is_runtime_value(v) for v in values):
                if isinstance(node.op, ast.And):
                    return all(values)
                return any(values)
            current = _ensure_wrapped(values[0])
            for value in values[1:]:
                rhs = _ensure_wrapped(value)
                if isinstance(node.op, ast.And):
                    current = Value(arith.AndIOp(_unwrap(current), _unwrap(rhs)).result)
                else:
                    current = Value(arith.OrIOp(_unwrap(current), _unwrap(rhs)).result)
            return current
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = self.eval_expr(node.operand, env)
            if _is_runtime_value(value):
                return ~_ensure_wrapped(value)
            return not value
        expr = ast.Expression(node)
        ast.fix_missing_locations(expr)
        code = compile(expr, self.filename, "eval")
        scope = dict(self.global_env)
        scope.update(env)
        return eval(code, scope, scope)

    def _bind_assignment(self, target, value, env):
        if isinstance(target, ast.Name):
            env[target.id] = value
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            if not isinstance(value, (tuple, list)):
                raise PTODSLAstError("Tuple/list assignment requires a tuple/list value.")
            if len(target.elts) != len(value):
                raise PTODSLAstError("Assignment target/value arity mismatch.")
            for elt, item in zip(target.elts, value):
                self._bind_assignment(elt, item, env)
            return
        raise PTODSLAstError(f"Unsupported assignment target: {ast.dump(target)}")

    def _static_range_values(self, args):
        if any(_is_runtime_value(arg) for arg in args):
            return None
        if len(args) == 1:
            lower_bound, upper_bound, step = 0, args[0], 1
        elif len(args) == 2:
            lower_bound, upper_bound = args
            step = 1
        elif len(args) == 3:
            lower_bound, upper_bound, step = args
        else:
            raise PTODSLAstError("range(...) expects 1 to 3 arguments.")
        if not all(isinstance(v, int) for v in (lower_bound, upper_bound, step)):
            return None
        if step == 0:
            raise PTODSLAstError("range(...) step must not be zero.")
        values = list(range(lower_bound, upper_bound, step))
        if len(values) > _STATIC_RANGE_UNROLL_LIMIT:
            return None
        return values

    def lower_statements(self, statements, env):
        terminated = False
        for stmt in statements:
            env, terminated = self.lower_statement(stmt, env)
            if terminated:
                break
        return env, terminated

    def lower_statement(self, stmt, env):
        if isinstance(stmt, ast.Assign):
            value = self.eval_expr(stmt.value, env)
            for target in stmt.targets:
                self._bind_assignment(target, value, env)
            return env, False

        if isinstance(stmt, ast.AnnAssign):
            value = self.eval_expr(stmt.value, env)
            self._bind_assignment(stmt.target, value, env)
            return env, False

        if isinstance(stmt, ast.AugAssign):
            if not isinstance(stmt.target, ast.Name):
                raise PTODSLAstError("Only simple names are supported in augmented assignment.")
            target_name = stmt.target.id
            expr = ast.BinOp(left=ast.Name(id=target_name, ctx=ast.Load()), op=stmt.op, right=stmt.value)
            value = self.eval_expr(expr, env)
            env[target_name] = value
            return env, False

        if isinstance(stmt, ast.Expr):
            self.eval_expr(stmt.value, env)
            return env, False

        if isinstance(stmt, ast.Pass):
            return env, False

        if isinstance(stmt, (ast.Try, ast.Raise, ast.AsyncFor, ast.AsyncWith)):
            raise PTODSLAstError(
                f"Unsupported Python statement in PTODSL AST frontend: {type(stmt).__name__}"
            )

        if isinstance(stmt, ast.With):
            with ExitStack() as stack:
                local_env = env.copy()
                for item in stmt.items:
                    manager = self.eval_expr(item.context_expr, local_env)
                    entered = stack.enter_context(manager)
                    if item.optional_vars is not None:
                        self._bind_assignment(item.optional_vars, entered, local_env)
                local_env, terminated = self.lower_statements(stmt.body, local_env)
                env.update(local_env)
                return env, terminated

        if isinstance(stmt, ast.If):
            condition = self.eval_expr(stmt.test, env)
            if not _is_runtime_value(condition):
                branch = stmt.body if condition else stmt.orelse
                return self.lower_statements(branch, env)

            carried_names = [
                name for name in sorted(_assigned_names(stmt.body + stmt.orelse))
                if name in env and _is_runtime_value(env[name])
            ]
            result_types = [_unwrap(env[name]).type for name in carried_names]
            has_else = bool(stmt.orelse) or bool(carried_names)
            if_op = scf.IfOp(_unwrap(condition), result_types, hasElse=has_else)

            with InsertionPoint(if_op.then_block):
                then_env = env.copy()
                then_env, terminated = self.lower_statements(stmt.body, then_env)
                if terminated:
                    raise PTODSLAstError("Dynamic branch returns are not supported yet in PTODSL AST lowering.")
                scf.YieldOp([_unwrap(then_env[name]) for name in carried_names])

            if has_else:
                with InsertionPoint(if_op.else_block):
                    else_env = env.copy()
                    if stmt.orelse:
                        else_env, terminated = self.lower_statements(stmt.orelse, else_env)
                        if terminated:
                            raise PTODSLAstError("Dynamic branch returns are not supported yet in PTODSL AST lowering.")
                    scf.YieldOp([_unwrap(else_env[name]) for name in carried_names])

            for name, result in zip(carried_names, if_op.results):
                env[name] = wrap_value(result)
            return env, False

        if isinstance(stmt, ast.For):
            if not _is_range_call(stmt.iter):
                raise PTODSLAstError("Only for-loops over range(...) are supported.")
            if any(isinstance(node, (ast.Break, ast.Continue)) for node in ast.walk(ast.Module(body=stmt.body, type_ignores=[]))):
                raise PTODSLAstError("break/continue inside for-loops are not supported yet in PTODSL AST lowering.")

            args = [self.eval_expr(arg, env) for arg in stmt.iter.args]
            static_values = self._static_range_values(args)
            if static_values is not None:
                for iv in static_values:
                    self._bind_assignment(stmt.target, iv, env)
                    env, terminated = self.lower_statements(stmt.body, env)
                    if terminated:
                        return env, True
                return env, False
            if len(args) == 1:
                lower_bound, upper_bound, step = const(0), args[0], const(1)
            elif len(args) == 2:
                lower_bound, upper_bound = args
                step = const(1)
            elif len(args) == 3:
                lower_bound, upper_bound, step = args
            else:
                raise PTODSLAstError("range(...) expects 1 to 3 arguments.")
            lower_bound = _as_loop_bound(lower_bound)
            upper_bound = _as_loop_bound(upper_bound)
            step = _as_loop_bound(step)

            carried_names = [
                name for name in sorted(_assigned_names(stmt.body))
                if name in env and _is_runtime_value(env[name])
            ]
            iter_args = [_unwrap(env[name]) for name in carried_names]
            loop = scf.ForOp(_unwrap(lower_bound), _unwrap(upper_bound), _unwrap(step), iter_args=iter_args)

            with InsertionPoint(loop.body):
                body_env = env.copy()
                body_env[stmt.target.id] = wrap_value(loop.induction_variable)
                for name, arg in zip(carried_names, loop.inner_iter_args):
                    body_env[name] = wrap_value(arg)
                body_env, terminated = self.lower_statements(stmt.body, body_env)
                if terminated:
                    raise PTODSLAstError("return inside for-loops is not supported yet in PTODSL AST lowering.")
                scf.YieldOp([_unwrap(body_env[name]) for name in carried_names])

            for name, result in zip(carried_names, loop.results):
                env[name] = wrap_value(result)
            return env, False

        if isinstance(stmt, ast.While):
            if any(isinstance(node, (ast.Break, ast.Continue)) for node in ast.walk(ast.Module(body=stmt.body, type_ignores=[]))):
                raise PTODSLAstError("break/continue inside while-loops are not supported yet in PTODSL AST lowering.")

            carried_names = [
                name for name in sorted(_assigned_names(stmt.body))
                if name in env and _is_runtime_value(env[name])
            ]
            result_types = [_unwrap(env[name]).type for name in carried_names]
            init_values = [_unwrap(env[name]) for name in carried_names]
            loop = scf.WhileOp(result_types, init_values)
            before = loop.before.blocks.append(*result_types)
            after = loop.after.blocks.append(*result_types)

            with InsertionPoint(before):
                before_env = env.copy()
                for name, arg in zip(carried_names, before.arguments):
                    before_env[name] = wrap_value(arg)
                condition = self.eval_expr(stmt.test, before_env)
                scf.ConditionOp(_unwrap(condition), [_unwrap(before_env[name]) for name in carried_names])

            with InsertionPoint(after):
                after_env = env.copy()
                for name, arg in zip(carried_names, after.arguments):
                    after_env[name] = wrap_value(arg)
                after_env, terminated = self.lower_statements(stmt.body, after_env)
                if terminated:
                    raise PTODSLAstError("return inside while-loops is not supported yet in PTODSL AST lowering.")
                scf.YieldOp([_unwrap(after_env[name]) for name in carried_names])

            for name, result in zip(carried_names, loop.results):
                env[name] = wrap_value(result)
            return env, False

        if isinstance(stmt, ast.Break):
            raise PTODSLAstError("break is not supported yet in PTODSL AST lowering.")

        if isinstance(stmt, ast.Continue):
            raise PTODSLAstError("continue is not supported yet in PTODSL AST lowering.")

        if isinstance(stmt, ast.Return):
            if stmt.value is None:
                func.ReturnOp([])
            else:
                value = self.eval_expr(stmt.value, env)
                if isinstance(value, (list, tuple)):
                    func.ReturnOp([_unwrap(v) for v in value])
                else:
                    func.ReturnOp([_unwrap(value)])
            return env, True

        raise PTODSLAstError(f"Unsupported statement in PTODSL AST frontend: {ast.dump(stmt)}")


def lower_function(fn, wrapped_args, env):
    fn_def = get_function_def(fn)
    lowerer = AstLowerer(fn, env)
    local_env = dict(env)
    for arg, param in zip(wrapped_args, fn_def.args.args):
        local_env[param.arg] = arg
    return lowerer.lower_statements(fn_def.body, local_env)
