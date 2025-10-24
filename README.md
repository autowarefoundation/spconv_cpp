
# Introduction

This repository is a pre-generated version of the [spconv](https://github.com/traveller59/spconv) library with the objective to ease its distribution.

The versions used to generate this code are:
 - cumm: 0.5.3
 - spconv: 2.3.8
 - CUDA: 12.8
 - architecture list: 7.5 8.0 8.6 8.7 8.9 9.0 10.0 11.0 12.0

# Compilation and package generation

```bash
# cumm
mkdir -p cumm/build && cd cumm/build && cmake .. && make && cpack -G DEB

# The package is generated in cumm/_packages/cumm_0.5.3_amd64.deb
cd ../../ && sudo apt-get install ./cumm/_packages/cumm_0.5.3_amd64.deb

# spconv
mkdir -p spconv/build && cd spconv/build && cmake .. && make -j $(nproc) && cpack -G DEB

# The package is generated in spconv/_packages/spconv_2.3.8_amd64.deb
cd ../../ && sudo apt-get install ./spconv/_packages/spconv_2.3.8_amd64.deb
```

# License

This repository is under the same license as spconv (Apache 2.0)