
# Introduction

This repository is a pre-generated version of the [spconv](https://github.com/traveller59/spconv) library with the objective to ease its distribution.

The versions used to generate this code are:
 - cumm: 0.5.3
 - spconv: 2.3.8
 - CUDA: 12.8
 - architecture list: 7.5 8.0 8.6 8.7 8.9 9.0 10.0 10.1 12.0

Note: CUDA Toolkit 13.0 renamed the SM101 for Thor GPUs to SM110.

# Compilation and package generation

## Native compilation and installation

```bash
# cumm
mkdir -p cumm/build-amd64 && cd cumm/build-amd64 && cmake .. && make && cpack -G DEB

# The package is generated in cumm/_packages/cumm_0.5.3_amd64.deb
cd ../../ && sudo apt-get install ./cumm/_packages/cumm_0.5.3_amd64.deb

# spconv
mkdir -p spconv/build-amd64 && cd spconv/build-amd64 && cmake .. && make -j $(nproc) && cpack -G DEB

# The package is generated in spconv/_packages/spconv_2.3.8_amd64.deb
cd ../../ && sudo apt-get install ./spconv/_packages/spconv_2.3.8_amd64.deb
```

## Cross compilation for ARM64

### SBSA

```bash
# Install cross compilation tools
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# Install CUDA cross compilation toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/cross-linux-sbsa/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get install cuda-cross-sbsa=12.8.1-1

# cumm
mkdir -p cumm/build-arm64 && cd cumm/build-arm64 && cmake .. -DCMAKE_TOOLCHAIN_FILE=../../extras/arm64-toolchain.cmake && make && cpack -G DEB && cd ../..

# spconv
mkdir -p spconv/build-arm64 && cd spconv/build-arm64 && cmake .. -DCMAKE_TOOLCHAIN_FILE=../../extras/arm64-toolchain.cmake && make -j $(nproc) && cpack -G DEB && cd ../..
```

### Jetson

```bash
# Install cross compilation tools
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# Install CUDA cross compilation toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/cross-linux-aarch64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get install cuda-cross-aarch64=12.8.1-1

# cumm
mkdir -p cumm/build-arm64 && cd cumm/build-arm64 && cmake .. -DCMAKE_TOOLCHAIN_FILE=../../extras/arm64-toolchain.cmake -DARM64_CUDA_TARGET=aarch64-linux && make && cpack -G DEB && cd ../..

# spconv
mkdir -p spconv/build-arm64 && cd spconv/build-arm64 && cmake .. -DCMAKE_TOOLCHAIN_FILE=../../extras/arm64-toolchain.cmake -DARM64_CUDA_TARGET=aarch64-linux && make -j $(nproc) && cpack -G DEB && cd ../..
```

# License

This repository is under the same license as spconv (Apache 2.0)