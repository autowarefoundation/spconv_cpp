
# How to update this repository

If newer versions of spconv or cuda are needed, this repository may need to be regenerated.
In what follows, we present the basic steps needed:

*Additional changes were needed for the creation of deb packages

Note: CUDA Toolkit 13.0 renamed the SM101 for Thor GPUs to SM110.

```bash
export CUMM_CUDA_VERSION=12.8
export CUMM_DISABLE_JIT=1
export SPCONV_DISABLE_JIT=1
export CUMM_CUDA_ARCH_LIST="7.5 8.0 8.6 8.7 8.9 9.0 10.0 10.1 12.0"

git clone https://github.com/traveller59/spconv.git && git clone https://github.com/FindDefinition/cumm
pip install pccm

cd cumm && git checkout v0.5.3 && python3 setup.py bdist_wheel && pip3 install dist/cumm_cu123-0.5.3-cp310-cp310-linux_x86_64.whl

cmake . -B ./build && \
    cmake --build ./build --config Release --parallel $(nproc)

cd build && cpack -G DEB

cd ../_packages && sudo apt-get install ./cumm_0.5.3_amd64.deb  


cd spconv && mkdir cpp && git checkout v2.3.8 && python3 setup.py bdist_wheel && python3 -m spconv.gencode --include=./cpp/include --src=./cpp/src  --inference_only=True

cd cpp && cp ../example/libspconv/CMakeLists.txt .

mkdir build && cmake . -B ./build && \
    cmake --build ./build --config Release --parallel $(nproc)


cd build && cpack -G DEB
```