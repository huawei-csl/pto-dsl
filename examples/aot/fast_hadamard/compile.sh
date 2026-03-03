# auto sync
python ./hadamard_builder.py > ./hadamard_no_sync.pto
ptoas --enable-insert-sync ./hadamard_no_sync.pto -o ./hadamard_auto_sync.cpp

# manual sync
python ./hadamard_builder.py --manual-sync > ./hadamard_manual_sync.pto
ptoas ./hadamard_manual_sync.pto -o ./hadamard_manual_sync.cpp
