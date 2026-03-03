Usage:

```bash
bash ./compile.sh  # generate PTO/CPP and build both auto/manual sync libs
python ./run_hadamard.py  # test manual-sync lib (default)
python ./run_hadamard.py --lib ./hadamard_auto_sync_lib.so  # test auto-sync lib
python ./run_hadamard.py --test-both  # test both libs
```
