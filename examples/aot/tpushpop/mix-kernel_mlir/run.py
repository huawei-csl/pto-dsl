import argparse
import ctypes
import os
import subprocess

import torch
import torch_npu  # noqa: F401

from ptodsl.utils.npu_info import get_test_device

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LIB_PATH = os.path.join(THIS_DIR, "build_artifacts", "tpushpop_mlir_lib.so")
DEFAULT_COMPILE_SCRIPT = os.path.join(THIS_DIR, "compile.sh")
DEFAULT_FIFO_BYTES = 4 * 1024
DEFAULT_FIFO_BYTES_BOTH = 8 * 1024
M = 16
N = 16
ATOL = 1e-4
RTOL = 1e-4
MODES = ("c2v", "c2v_add", "v2c", "bidi")


def torch_to_ctypes(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def compile_example(compile_script: str, mode: str) -> None:
    env = dict(os.environ, TPUSHPOP_MODE=mode)
    subprocess.run(
        ["bash", compile_script],
        check=True,
        cwd=THIS_DIR,
        env=env,
    )


def load_lib(lib_path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(lib_path)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.call_kernel.restype = None
    return lib


def make_gm_slot_buffer(*, fifo_bytes: int, device: str) -> torch.Tensor:
    fifo_elems = max(1, (fifo_bytes + 3) // 4)
    return torch.zeros((fifo_elems,), dtype=torch.float32, device=device)


def make_io_tensors(*, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.rand((M, N), dtype=torch.float32, device=device) - 0.5
    y = torch.zeros((M, N), dtype=torch.float32, device=device)
    return x, y


def fifo_bytes_for_mode(mode: str) -> int:
    return DEFAULT_FIFO_BYTES_BOTH if mode in ("v2c", "bidi") else DEFAULT_FIFO_BYTES


def run_kernel(
    lib: ctypes.CDLL, *, gm_slot_buffer: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> None:
    stream_ptr = torch.npu.current_stream()._as_parameter_
    lib.call_kernel(
        1,
        stream_ptr,
        torch_to_ctypes(gm_slot_buffer),
        torch_to_ctypes(x),
        torch_to_ctypes(y),
    )
    torch.npu.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=MODES, default="c2v")
    return parser.parse_args()


def reference(mode: str, x: torch.Tensor) -> torch.Tensor:
    y = x.cpu() @ x.cpu()
    if mode == "c2v":
        return y
    if mode == "v2c":
        return x.cpu()
    return 2 * y


def main() -> None:
    args = parse_args()
    compile_example(DEFAULT_COMPILE_SCRIPT, args.mode)

    device = get_test_device()
    torch.npu.set_device(device)

    lib = load_lib(DEFAULT_LIB_PATH)
    gm_slot_buffer = make_gm_slot_buffer(
        fifo_bytes=fifo_bytes_for_mode(args.mode),
        device=device,
    )
    torch.set_printoptions(precision=1, threshold=2000, linewidth=250, sci_mode=False)
    x, y = make_io_tensors(device=device)

    print(y)
    run_kernel(lib, gm_slot_buffer=gm_slot_buffer, x=x, y=y)
    print(y)

    y_ref = reference(args.mode, x)
    y_cpu = y.cpu()

    print(y_ref - y_cpu)
    max_abs = float(torch.max(torch.abs(y_cpu - y_ref)).item())
    ok = bool(torch.allclose(y_cpu, y_ref, atol=ATOL, rtol=RTOL))

    print(f"shape=({M}, {N}) max_abs={max_abs:.6f}")
    if not ok:
        raise SystemExit(
            f"Validation failed with atol={ATOL} rtol={RTOL}. max_abs={max_abs:.6f}"
        )

    print(f"Validation passed for mode={args.mode} using {DEFAULT_LIB_PATH}.")


if __name__ == "__main__":
    main()
