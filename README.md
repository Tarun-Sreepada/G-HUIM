# cuDPEFIM

**cuDPEFIM**, is a CUDA-based high utility itemset mining program

---

## Project Structure

Below is a snapshot of the project structure (excluding `build/` and `docs/` directories):

```bash
.
├── CMakeLists.txt
├── Doxyfile
├── README.md
├── a.txt
├── include
│   ├── args.hpp
│   ├── data_types
│   ├── file
│   ├── memory
│   ├── mine
│   ├── patterns
│   ├── task
│   └── timer.hpp
└── src
    ├── args.cpp
    ├── file
    ├── main.cu
    ├── memory
    └── mine
```

*(This was generated using: ` tree -L 2 -I 'build|docs' `)*

---

## Build Instructions

### Release Build

To build the project in Release mode, run the following commands:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j$(nproc)
```

---

## Dependencies

---

## Running the Application

After building the project, you can run the application as follows:

- **Display Help:**

  ```bash
  ./build/cuDPEFIM -h
  ```

- **Example Execution Commands:**

  Build and run:
  ```bash
  cmake --build build -- -j$(nproc) && ./build/cuDPEFIM --input '/home/tarun/Documents/cuEFIM/datasets/accidents_utility_spmf.txt' --minutil 15000000 --threads 16 --block_size 512 --total_memory 20G 
  
  --managed true --output /dev/null
  ```

  Launch with NVIDIA Compute Utility (NSight Compute):
  ```bash
  sudo /usr/local/cuda-12.6/bin/ncu -o "new_scan_$(date +%Y%m%d_%H%M%S)" --set full -f ./build/cuDPEFIM --input accidents_utility_spmf.txt --minutil 15000000 --threads 16 --block_size 512 --num_blocks 114
  ```

  Run a sample job:
  ```bash
    cmake --build build -- -j$(nproc) &&  ./build/cuDPEFIM --input '/export/home1/ltarun/cuEFIM/datasets/accidents_utility_spmf.txt' --minutil 8000000 --threads 16 --total_memory 20G
  ```

---
