Try different tile sizes (512 gets highest TFLOPs for long sequence)

```bash
export PTODSL_TEST_DEVICE_ID=7

python ./run.py --tile-s1 512  # default
python ./run.py --tile-s1 256  # slower for long seq
python ./run.py --tile-s1 1024  # wrong result
```

Reference outputs in [./results](./results)
