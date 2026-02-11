from ptodsl import JitWrapper


class _FakeType:
    def __init__(self, text):
        self._text = text

    def __str__(self):
        return self._text


def test_generate_caller_cpp_for_multicore_add_signature():
    def vec_add_kernel(
        arg0: "ptr_type",
        arg1: "ptr_type",
        arg2: "ptr_type",
        arg_vrow_i32: "index_dtype",
        arg_vcol_i32: "index_dtype",
    ) -> None:
        return None

    wrapper = JitWrapper(vec_add_kernel, meta_data=lambda: {}, block_dim=20)
    wrapper._arg_types = [
        _FakeType("!pto.ptr<f32>"),
        _FakeType("!pto.ptr<f32>"),
        _FakeType("!pto.ptr<f32>"),
        _FakeType("i32"),
        _FakeType("i32"),
    ]

    caller_cpp = wrapper._generate_caller_cpp("kernel.cpp")

    assert '#include "kernel.cpp"' in caller_cpp
    assert (
        'extern "C" void call_kernel(void *stream, uint8_t *arg0, uint8_t *arg1, uint8_t *arg2, '
        "int32_t arg_vrow_i32, int32_t arg_vcol_i32)"
    ) in caller_cpp
    assert (
        "vec_add_kernel<<<20, nullptr, stream>>>((float *)arg0, (float *)arg1, (float *)arg2, "
        "arg_vrow_i32, arg_vcol_i32);"
    ) in caller_cpp


def test_generate_caller_cpp_maps_pointer_and_scalar_types():
    def mixed_kernel(data: "ptr_i8", count: "i64_type", idx: "index_dtype") -> None:
        return None

    wrapper = JitWrapper(mixed_kernel, meta_data=lambda: {}, block_dim=7)
    wrapper._arg_types = [
        _FakeType("!pto.ptr<i8>"),
        _FakeType("i64"),
        _FakeType("index"),
    ]

    caller_cpp = wrapper._generate_caller_cpp("generated.cpp")

    assert (
        'extern "C" void call_kernel(void *stream, uint8_t *data, int64_t count, int64_t idx)'
    ) in caller_cpp
    assert "mixed_kernel<<<7, nullptr, stream>>>((int8_t *)data, count, idx);" in caller_cpp
