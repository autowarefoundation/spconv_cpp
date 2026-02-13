
# Introduction

This repository is a pre-generated version of the [spconv](https://github.com/traveller59/spconv) library with the objective to ease its distribution.

The versions used to generate this code are:
 - cumm: 0.5.3
 - spconv: 2.3.8
 - CUDA: 13.0
 - architecture list: 7.5 8.0 8.6 8.7 8.9 9.0 10.0 10.1/11.0 12.0

Note: CUDA Toolkit 13.0 renamed the SM101 for Thor GPUs to SM110.

# Compilation and package generation

Here is a way to use a NVIDIA CUDA Docker image for compilation. Feel free to use your own environment.
```bash
docker pull nvidia/cuda:13.0.2-devel-ubuntu24.04
git clone https://github.com/autowarefoundation/spconv_cpp
cd spconv_cpp
docker run -it --rm --gpus all -v .:/workspace nvidia/cuda:13.0.2-devel-ubuntu24.04 bash
cd /workspace
apt update && apt install -y cmake wget
```

## Native compilation and installation

```bash
# cumm
mkdir -p cumm/build-amd64 && cd cumm/build-amd64 && cmake .. && make && cpack -G DEB

# The package is generated in cumm/_packages/cumm_0.5.3_amd64.deb
cd ../../ && apt install ./cumm/_packages/cumm_0.5.3_amd64.deb

# spconv
mkdir -p spconv/build-amd64 && cd spconv/build-amd64 && cmake .. && make -j $(nproc) && cpack -G DEB

# The package is generated in spconv/_packages/spconv_2.3.8_amd64.deb
cd ../../ && apt install ./spconv/_packages/spconv_2.3.8_amd64.deb
```

## Cross compilation for ARM64

Note: Jetson Thor is based on the SBSA stack, so we can use the SBSA toolchain for cross compilation.

```bash
# Install cross compilation tools
apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# Install CUDA cross compilation toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/cross-linux-sbsa/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update && apt install -y cuda-cross-sbsa=13.0.2-1

# cumm
mkdir -p cumm/build-arm64 && cd cumm/build-arm64 && cmake .. -DCMAKE_TOOLCHAIN_FILE=../../extras/arm64-toolchain.cmake && make && cpack -G DEB && cd ../..

# spconv
mkdir -p spconv/build-arm64 && cd spconv/build-arm64 && cmake .. -DCMAKE_TOOLCHAIN_FILE=../../extras/arm64-toolchain.cmake && make -j $(nproc) && cpack -G DEB && cd ../..
```

# License

This repository is under the same license as spconv (Apache 2.0)
