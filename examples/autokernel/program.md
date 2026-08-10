# autokernel

This is an experiment to have the LLM improve a NPU kernel for matrix multiplication.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar12`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `compile.sh` — compiles the `kernel.py`, which compiles the kernel and produces files `matmul.cpp`, `matmul.pto`, and the `matmul_kernel.so` that will be run on the NPU, those files are just compiled versions of `kernel.py` and are not important.
   - `kernel.py` — the file you modify. Swizzling techniques, tensor layouts, etc.
   - `benchmark.py` — the script to run and benchmark the kernel giving the TFLOPS
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single NPU. The eval script runs for a **fixed time budget of 30 seconds** (wall clock training time, excluding startup/compilation). You launch it simply as: `bash compile.sh 2>&1 | grep error`.

**What you CAN do:**
- Modify `kernel.py` — this is the only file you edit. Everything is fair game: change the swizzling strategy, pipelining strategy, loops, syncs, double buffering

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies.
- Modify the evaluation harness. The TFLOPS printed from `benchmark.py` is the ground truth metric.

**The goal is simple: get the maximum TFLOPS.** Everything is fair game: change the swizzling strategy, pipelining strategy, loops

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 1 TFLOP improvement that adds 20 lines of hacky code? Probably not worth it. A 0.1 TFLOP improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
(m, n, k)=(4224, 16384, 16384)
TFLOPS: 303.5
execution_time: 33.12321 us  
```

Note that the script is configured to always stop after 30 seconds, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^TFLOPS:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	tflops status  description
```

1. git commit hash (short, 7 chars)
2. TFLOPS achieved (e.g. 201.5) — use 0.0 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	tflops	status	description
a1b2c3d	201.3	keep	baseline
b2c3d4e	150.8	discard	increase tile size of matrix B to 128 from 64
c3d4e5f	344.5	discard	pre-load 8 matrices instead of 4
d4e5f6g	401.1   keep	new swizzling pattern
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Optimize `kernel.py` with an experimental idea by directly hacking the code.
3. git commit
4. Compile the kernel: `bash compile.sh 2>&1 | grep error` (be sure to use the exact command to not flood context) and if any output is generated read the errors and attempt to fix them in `kernel.py`. If you can't get things to work after more than a few attempts, give up.
5. Run and bench the kernel: `pyton benchmark.py` 
6. If it outputs the TFLOPS we know the kernel ran successfully. Otherwise if stack traces or timeouts are printed we now it failed and then you attempt to fix the problem in `kernel.py`.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If TFLOPS improved (higher), you "advance" the branch, keeping the git commit
9. If TFLOPS is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take 30 seconds  total (+ a few seconds for startup and eval overhead). If a run exceeds 30 seconds, treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~1 minutes then you can run approx 12/hour, for a total of about 500 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!