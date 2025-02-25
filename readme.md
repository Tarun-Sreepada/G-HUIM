# cuDPHIM

## Debug
`
cmake -DCMAKE_BUILD_TYPE=Debug .. &&
make -j$(nproc)
`

## Release
`
cmake -DCMAKE_BUILD_TYPE=Release .. &&
make -j$(nproc)
`