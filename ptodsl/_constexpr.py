import hashlib
import inspect
from dataclasses import dataclass
from typing import Generic, TypeVar, get_origin


T = TypeVar("T")
_MISSING = object()


class Constexpr(Generic[T]):
    """Marker annotation for compile-time-only parameters."""


@dataclass(frozen=True)
class SignatureAnalysis:
    signature: inspect.Signature
    constexpr_params: tuple[inspect.Parameter, ...]
    runtime_params: tuple[inspect.Parameter, ...]

    @property
    def has_constexpr_params(self):
        return bool(self.constexpr_params)


@dataclass(frozen=True)
class BoundArguments:
    all_arguments: dict[str, object]
    constexpr_arguments: dict[str, object]
    runtime_arguments: tuple[object, ...]
    missing_runtime: tuple[str, ...]


def analyze_signature(fn_or_signature):
    signature = (
        fn_or_signature
        if isinstance(fn_or_signature, inspect.Signature)
        else inspect.signature(fn_or_signature)
    )
    constexpr_params = []
    runtime_params = []
    for param in signature.parameters.values():
        if is_constexpr_annotation(param.annotation):
            constexpr_params.append(param)
        else:
            runtime_params.append(param)
    return SignatureAnalysis(
        signature=signature,
        constexpr_params=tuple(constexpr_params),
        runtime_params=tuple(runtime_params),
    )


def is_constexpr_annotation(annotation):
    if isinstance(annotation, str):
        return annotation.startswith("Constexpr[") and annotation.endswith("]")
    return get_origin(annotation) is Constexpr


def unwrap_constexpr_annotation(annotation):
    if isinstance(annotation, str):
        if is_constexpr_annotation(annotation):
            return annotation[len("Constexpr[") : -1]
        return annotation
    if get_origin(annotation) is Constexpr:
        args = getattr(annotation, "__args__", ())
        if args:
            return args[0]
        return inspect._empty
    return annotation


def bind_constexpr_arguments(analysis, *args, **kwargs):
    bound = analysis.signature.bind_partial(*args, **kwargs)
    provided_runtime = [
        name for name in bound.arguments if name in {p.name for p in analysis.runtime_params}
    ]
    if provided_runtime:
        joined = ", ".join(provided_runtime)
        raise TypeError(
            "Specialization only accepts constexpr arguments; "
            f"got runtime arguments: {joined}."
        )

    values = {}
    missing = []
    for param in analysis.constexpr_params:
        if param.name in bound.arguments:
            values[param.name] = bound.arguments[param.name]
        elif param.default is not inspect._empty:
            values[param.name] = param.default
        else:
            missing.append(param.name)

    if missing:
        joined = ", ".join(missing)
        raise TypeError(f"Missing required constexpr arguments: {joined}.")

    return values


def bind_kernel_arguments(analysis, *args, **kwargs):
    bound = analysis.signature.bind_partial(*args, **kwargs)
    values = dict(bound.arguments)

    missing_runtime = []
    runtime_arguments = []
    constexpr_arguments = {}

    for param in analysis.signature.parameters.values():
        if param.name not in values and param.default is not inspect._empty:
            values[param.name] = param.default

    for param in analysis.constexpr_params:
        if param.name not in values:
            raise TypeError(f"Missing required constexpr argument '{param.name}'.")
        constexpr_arguments[param.name] = values[param.name]

    for param in analysis.runtime_params:
        if param.name in values:
            runtime_arguments.append(values[param.name])
        else:
            runtime_arguments.append(_MISSING)
            missing_runtime.append(param.name)

    return BoundArguments(
        all_arguments=values,
        constexpr_arguments=constexpr_arguments,
        runtime_arguments=tuple(runtime_arguments),
        missing_runtime=tuple(missing_runtime),
    )


def normalize_constexpr_bindings(bindings):
    return tuple((name, _normalize_constexpr_value(value)) for name, value in bindings.items())


def specialization_suffix(bindings):
    if not bindings:
        return "default"
    digest = hashlib.sha256(repr(normalize_constexpr_bindings(bindings)).encode("utf-8"))
    return digest.hexdigest()[:16]


def meta_kwargs_for(meta_fn, constexpr_bindings):
    signature = inspect.signature(meta_fn)
    if not signature.parameters:
        return None

    kwargs = {}
    missing = []
    for param in signature.parameters.values():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError("`meta_data` does not support positional-only parameters.")
        if param.name in constexpr_bindings:
            kwargs[param.name] = constexpr_bindings[param.name]
        elif param.default is inspect._empty:
            missing.append(param.name)

    if missing:
        joined = ", ".join(missing)
        raise TypeError(
            "`meta_data` requires unresolved constexpr parameters: "
            f"{joined}."
        )
    return kwargs


def is_dynamic_value(value):
    if isinstance(value, (list, tuple)):
        return any(is_dynamic_value(item) for item in value)
    if isinstance(value, dict):
        return any(
            is_dynamic_value(key) or is_dynamic_value(item)
            for key, item in value.items()
        )
    cls = value.__class__
    module = getattr(cls, "__module__", "")
    name = getattr(cls, "__name__", "")
    if module.startswith("ptodsl") and name == "Value":
        return True
    if module.startswith("mlir.") and (
        name in {"Value", "OpResult", "BlockArgument"} or "Value" in name
    ):
        return True
    return False


def require_constexpr_value(value, *, context):
    if is_dynamic_value(value):
        raise TypeError(f"`{context}` requires compile-time values, got dynamic PTODSL/MLIR values.")
    return value


def require_static_int(value, *, context):
    require_constexpr_value(value, context=context)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{context}` requires a Python int, got {type(value)!r}.")
    return value


def require_static_int_sequence(values, *, context):
    return [require_static_int(value, context=context) for value in values]


def const_expr(value):
    require_constexpr_value(value, context="const_expr")
    return bool(value)


def range_constexpr(start, stop=None, step=1):
    if stop is None:
        return range(
            require_static_int(start, context="range_constexpr"),
        )
    return range(
        require_static_int(start, context="range_constexpr"),
        require_static_int(stop, context="range_constexpr"),
        require_static_int(step, context="range_constexpr"),
    )


def _normalize_constexpr_value(value):
    require_constexpr_value(value, context="constexpr specialization")
    if value is None:
        return ("none", None)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    if isinstance(value, tuple):
        return ("tuple", tuple(_normalize_constexpr_value(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_normalize_constexpr_value(item) for item in value))
    if isinstance(value, dict):
        items = [
            (_normalize_constexpr_value(key), _normalize_constexpr_value(item))
            for key, item in value.items()
        ]
        items.sort(key=repr)
        return ("dict", tuple(items))

    cls = value.__class__
    module = getattr(cls, "__module__", "")
    if module.startswith("mlir.") or module.startswith("ptodsl."):
        return ("object", module, cls.__qualname__, str(value))

    raise TypeError(
        "Unsupported constexpr value type for specialization key: "
        f"{type(value)!r}."
    )


__all__ = [
    "BoundArguments",
    "Constexpr",
    "SignatureAnalysis",
    "_MISSING",
    "analyze_signature",
    "bind_constexpr_arguments",
    "bind_kernel_arguments",
    "const_expr",
    "is_constexpr_annotation",
    "is_dynamic_value",
    "meta_kwargs_for",
    "normalize_constexpr_bindings",
    "range_constexpr",
    "require_constexpr_value",
    "require_static_int",
    "require_static_int_sequence",
    "specialization_suffix",
    "unwrap_constexpr_annotation",
]
