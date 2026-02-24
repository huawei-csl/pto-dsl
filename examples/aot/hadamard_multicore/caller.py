import sys

_DTYPE_TO_CTYPE = {
    "float16": "__fp16",
    "float32": "float",
}


def case_id(dtype, n):
    return f"{dtype}_n{n}"


def generate_caller(dtype, n):
    src_ctype = _DTYPE_TO_CTYPE[dtype]
    fn = case_id(dtype, n)  
    return f"""\
    #include "{fn}.cpp"

    extern "C" void call_{fn}(
        void *stream, uint8_t *src, uint8_t *out)
    {{
        {fn}<<<20, nullptr, stream>>>(
            ({src_ctype} *)src, ({src_ctype} *)out);
    }}
    """


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python caller.py <dtype> <N>", file=sys.stderr)
        sys.exit(1)
    print(generate_caller(sys.argv[1], int(sys.argv[2])))
