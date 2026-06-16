import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class KernelAction:
    kind: str    # "edit", "build", "test", "benchmark"
    payload: dict


@dataclass
class KernelObservation:
    passed_tests: bool
    latency_ms: float | None
    speedup_vs_baseline: float | None
    summary: str


class KernelSearchEnv:
    """
    Optimization environment for finding fast NPU kernels.

    env = KernelSearchEnv(
        repo_path      = "path/to/kernel/dir",
        build_cmd      = ["bash", "compile.sh"],
        test_cmd       = ["python", "run.py"],
        bench_cmd      = ["python", "bench.py"],   # must print "latency_ms=<float>"
        baseline_files = {"builder.py": "<source>"},
    )
    obs = env.reset()
    obs = env.step(KernelAction("edit", {"path": "builder.py", "old": "...", "new": "..."}))
    obs = env.step(KernelAction("test", {}))
    obs = env.step(KernelAction("benchmark", {}))
    """

    def __init__(
        self,
        repo_path: str,
        test_cmd: list[str],
        bench_cmd: list[str] | None = None,
        build_cmd: list[str] | None = None,
        baseline_files: dict[str, str] | None = None,
    ):
        self._root = Path(repo_path)
        self._test_cmd = test_cmd
        self._bench_cmd = bench_cmd
        self._build_cmd = build_cmd
        self._baseline_files: dict[str, str] = baseline_files or {}
        self._baseline_ms: float | None = None
        self._best_ms: float | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> KernelObservation:
        """Restore baseline sources, recompile, and benchmark."""
        self._restore_baseline()
        if self._build_cmd:
            ok, out = self._run_cmd(self._build_cmd)
            if not ok:
                return self._error_obs(f"Baseline build failed:\n{out}")
        latency = self._run_benchmark()
        self._baseline_ms = latency
        self._best_ms = latency
        return KernelObservation(
            passed_tests=True,
            latency_ms=latency,
            speedup_vs_baseline=1.0,
            summary="Baseline ready",
        )

    def step(self, action: KernelAction) -> KernelObservation:
        dispatch = {
            "edit":      self._handle_edit,
            "build":     self._handle_build,
            "test":      self._handle_test,
            "benchmark": self._handle_benchmark,
        }
        handler = dispatch.get(action.kind)
        if handler is None:
            raise ValueError(f"Unknown action kind {action.kind!r}. Valid: {list(dispatch)}")
        return handler(action.payload)

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_edit(self, payload: dict) -> KernelObservation:
        rel = payload.get("path")
        if not rel:
            return self._error_obs("edit payload must include 'path'")

        target = self._root / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        if "content" in payload:
            target.write_text(payload["content"], encoding="utf-8")
        elif "old" in payload and "new" in payload:
            original = target.read_text(encoding="utf-8")
            if payload["old"] not in original:
                return self._error_obs(f"'old' string not found in {rel}; edit not applied.")
            target.write_text(original.replace(payload["old"], payload["new"], 1), encoding="utf-8")
        else:
            return self._error_obs("edit payload must have 'content' or both 'old'+'new'")

        return KernelObservation(
            passed_tests=False,
            latency_ms=None,
            speedup_vs_baseline=None,
            summary=f"Applied edit to {rel}. Run 'test' to verify correctness.",
        )

    def _handle_build(self, payload: dict) -> KernelObservation:
        cmd = payload.get("cmd") or self._build_cmd
        if not cmd:
            return self._error_obs("No build_cmd configured and none provided in payload.")
        ok, output = self._run_cmd(cmd)
        return KernelObservation(
            passed_tests=ok,
            latency_ms=None,
            speedup_vs_baseline=None,
            summary=output if not ok else f"Build succeeded.\n{output}".strip(),
        )

    def _handle_test(self, payload: dict) -> KernelObservation:
        cmd = payload.get("cmd") or self._test_cmd
        ok, output = self._run_cmd(cmd)
        return KernelObservation(
            passed_tests=ok,
            latency_ms=None,
            speedup_vs_baseline=None,
            summary=output if not ok else f"All tests passed.\n{output}".strip(),
        )

    def _handle_benchmark(self, payload: dict) -> KernelObservation:
        cmd = payload.get("cmd") or self._bench_cmd
        ok, output = self._run_cmd(cmd)
        if not ok:
            return KernelObservation(
                passed_tests=False, latency_ms=None, speedup_vs_baseline=None,
                summary=f"Benchmark failed:\n{output}",
            )

        latency = self._parse_latency(output)
        if latency is None:
            return self._error_obs(
                f"Could not parse 'latency_ms=<number>' from bench output:\n{output}"
            )

        speedup = self._baseline_ms / latency if self._baseline_ms else None
        if self._best_ms is None or latency < self._best_ms:
            self._best_ms = latency

        return KernelObservation(
            passed_tests=True,
            latency_ms=latency,
            speedup_vs_baseline=speedup,
            summary=self._bench_summary(latency, speedup),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _restore_baseline(self):
        for rel, content in self._baseline_files.items():
            target = self._root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    def _run_cmd(self, cmd: list[str] | str) -> tuple[bool, str]:
        if isinstance(cmd, str):
            cmd = cmd.split()
        result = subprocess.run(cmd, cwd=str(self._root), capture_output=True, text=True)
        return result.returncode == 0, (result.stdout + result.stderr).strip()

    def _run_benchmark(self) -> float | None:
        ok, output = self._run_cmd(self._bench_cmd)
        return self._parse_latency(output) if ok else None

    @staticmethod
    def _parse_latency(output: str) -> float | None:
        for line in reversed(output.splitlines()):
            if line.strip().startswith("latency_ms="):
                try:
                    return float(line.strip().split("=", 1)[1])
                except ValueError:
                    pass
        return None

    def _bench_summary(self, latency: float, speedup: float | None) -> str:
        parts = [f"latency_ms={latency:.3f}"]
        if speedup is not None:
            parts.append(f"speedup={speedup:.3f}x vs baseline")
        if self._best_ms is not None:
            parts.append(f"best so far: {self._best_ms:.3f} ms")
        return "  ".join(parts)

    @staticmethod
    def _error_obs(msg: str) -> KernelObservation:
        return KernelObservation(passed_tests=False, latency_ms=None, speedup_vs_baseline=None, summary=msg)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def baseline_ms(self) -> float | None:
        return self._baseline_ms

    @property
    def best_ms(self) -> float | None:
        return self._best_ms

    @property
    def best_speedup(self) -> float | None:
        if self._baseline_ms and self._best_ms:
            return self._baseline_ms / self._best_ms
        return None
