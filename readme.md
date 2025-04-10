# cuDPHIM

## Build Instructions

### Debug Build
To build the project in Debug mode, run the following commands:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -- -j$(nproc)
```

### Release Build
To build the project in Release mode, run the following commands:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j$(nproc)
```

## Dependencies
Ensure the following dependency is installed:
```bash
sudo apt install liburing-dev
```

## Running the Application
After building the project, you can run the application using:
```bash
./build/cuDPEFIM --input accidents_utility_spmf.txt --minutil 15000000 --threads 16 --block_size 512
```

## Generating Documentation
To generate the documentation using Doxygen, execute:
```bash
doxygen Doxyfile
cd docs/latex
make
cd ../..
```

`
cmake --build build -- -j$(nproc) && compute-sanitizer ./build/cuDPEFIM --input accidents_utility_spmf.txt --minutil 15000000 --threads 16 --block_size 512
`
`
sudo /usr/local/cuda-12.6/bin/ncu -o "new_scan_$(date +%Y%m%d_%H%M%S)" --set full -f ./build/cuDPEFIM --input accidents_utility_spmf.txt --minutil 15000000 --threads 16 --block_size 512 --num_blocks 114
`