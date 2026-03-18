
```bash
bash compile.sh  # default for matrix size 64
python run_inverse.py

bash compile.sh 128  # another matrix size
python run_inverse.py --matrix-size 128 --lib-path ./inverse_lib.so
```

This example now also uses persistent-kernel style launch with fixed `blockDim=24`,
and the kernel loops over batch tiles internally.
