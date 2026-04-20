"""Helpers for invoking ptoas and bisheng from tests."""

import os
import shutil
import subprocess

# ---------------------------------------------------------------------------
# Tool availability (evaluated once at import time)
# ---------------------------------------------------------------------------

_PTOAS_BIN = shutil.which("ptoas")
_BISHENG_BIN = shutil.which("bisheng")
_PTO_LIB_PATH = os.environ.get("PTO_LIB_PATH", "/sources/pto-isa")


def ptoas_available():
    """``True`` when the ``ptoas`` CLI is on ``PATH``."""
    return _PTOAS_BIN is not None


def bisheng_available():
    """``True`` when ``bisheng`` is on ``PATH`` and PTO_LIB_PATH/include exists."""
    return _BISHENG_BIN is not None and os.path.isdir(
        os.path.join(_PTO_LIB_PATH, "include")
    )


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------


def run_ptoas(pto_path, cpp_path, *, enable_insert_sync=True):
    """Run ``ptoas`` to assemble *pto_path* → *cpp_path*.

    Raises :class:`subprocess.CalledProcessError` on failure.
    """
    cmd = ["ptoas"]
    if enable_insert_sync:
        cmd.append("--enable-insert-sync")
    cmd += [str(pto_path), "-o", str(cpp_path)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def run_bisheng(caller_cpp, output_so, *, npu_arch="dav-2201", cwd=None):
    """Run ``bisheng`` to compile *caller_cpp* → *output_so*.

    Raises :class:`subprocess.CalledProcessError` on failure.
    """
    pto_isa = os.environ.get("PTO_LIB_PATH", "/sources/pto-isa")
    ascend_home = os.environ.get("ASCEND_TOOLKIT_HOME")
    cmd = [
        "bisheng",
        f"-I{pto_isa}/include",
        f"-I{ascend_home}/include",
        f"-I{ascend_home}/pkg_inc",
        f"-I{ascend_home}/pkg_inc/runtime",
        f"-I{ascend_home}/pkg_inc/profiling",
        "-fPIC",
        "-shared",
        "-D_FORTIFY_SOURCE=2",
        "-O2",
        "-std=c++17",
        "-Wno-macro-redefined",
        "-Wno-ignored-attributes",
        "-fstack-protector-strong",
        "-xcce",
        "-Xhost-start",
        "-Xhost-end",
        "-mllvm",
        "-cce-aicore-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-function-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-record-overflow=true",
        "-mllvm",
        "-cce-aicore-addr-transform",
        "-mllvm",
        "-cce-aicore-dcci-insert-for-scalar=false",
        f"--npu-arch={npu_arch}",
        "-DMEMORY_BASE",
        "-std=gnu++17",
        str(caller_cpp),
        "-o",
        str(output_so),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=cwd)
