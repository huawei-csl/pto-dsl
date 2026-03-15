Usage:

```bash
bash ./compile.sh  # generate PTO/CPP and build both auto/manual sync libs
python3 ./run_inverse.py  # test auto-sync lib (default)
python3 ./run_inverse.py --manual-sync  # test manual-sync lib
```

TODO:
- Fix true dynamic-shape behavior for `n < 128` in the DSL kernel (current `run_inverse.py` pads to `128x128` then slices output).
