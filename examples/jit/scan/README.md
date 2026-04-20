# Single core prefix sum (scan)

An implementation of prefix sum (scan) algorithm, based on https://arxiv.org/abs/2505.15112v1. Only single core algorithm is implemented (ScanU from the paper).

The implementation follows the one from pto-kernels, however the running sum calculation is stored in a tile to avoid issues with compiler removing it as unused code. Furthermore, due to inability to synchronize on the scalar pipe we use a barrier as a workaround.

Usage:

```bash
python ./run_scan_single_core.py
```
