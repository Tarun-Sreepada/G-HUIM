# cuDPHIM

#### For Debug Build:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -- -j$(nproc)
```

#### For Release Build:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j$(nproc)
```

---

```bash
sudo apt install liburing-dev


cmake --build build -- -j$(nproc) && ./build/cuDPEFIM --input accidents_utility_spmf.txt --minutil 15000000 --threads 16 --managed true


doxygen Doxyfile 
cd docs/latex
make
```