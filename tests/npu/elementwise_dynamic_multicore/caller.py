"""Generate caller.cpp for a given op name."""

import sys

# Registry: op_name -> ctype
OPS = {
    "div": "float",
    "mul": "float",
    "sub": "float",
    "add": "float",
    "or": "int16_t",
}


def generate_caller(op_name, ctype=None):
    """Generate caller.cpp content.

    ctype: C type for pointer casts (default "float"). Use "int32_t" for integer ops.
    If not provided, looks up from OPS registry.
    """
    if ctype is None:
        if op_name not in OPS:
            raise ValueError(
                f"Unknown op: {op_name}. Register it in OPS or pass ctype explicitly."
            )
        ctype = OPS[op_name]
    return _binary_caller(op_name, ctype)


def _binary_caller(op_name, ctype):
    return f"""\
#include "{op_name}_1d.cpp"

#define ptoas_bitcast ptoas_bitcast_2d
#include "{op_name}_2d.cpp"
#undef ptoas_bitcast

extern "C" void call_kernel_1d(
    void *stream, uint8_t *x, uint8_t *y, uint8_t *z, int32_t N)
{{
    vec_{op_name}_1d_dynamic<<<20, nullptr, stream>>>(
        ({ctype} *)x, ({ctype} *)y, ({ctype} *)z, N
        );
}}

extern "C" void call_kernel_2d(
    void *stream, uint8_t *x, uint8_t *y, uint8_t *z, int32_t M, int32_t N)
{{
    vec_{op_name}_2d_dynamic<<<20, nullptr, stream>>>(
        ({ctype} *)x, ({ctype} *)y, ({ctype} *)z, M, N
        );
}}
"""


_DTYPE_TO_CTYPE = {
    "float32": "float",
    "float16": "half",
    "int32": "int32_t",
    "int16": "int16_t",
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python caller.py <op_name> [dtype]", file=sys.stderr)
        sys.exit(1)
    op_name = sys.argv[1]
    dtype = sys.argv[2] if len(sys.argv) > 2 else None
    ctype = _DTYPE_TO_CTYPE.get(dtype) if dtype else None
    print(generate_caller(op_name, ctype=ctype))
