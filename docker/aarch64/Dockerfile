FROM ubuntu:latest
WORKDIR /root
RUN apt-get update
# RUN apt-get upgrade -y
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    build-essential clang-18 cmake git libopenblas-dev libboost-exception-dev pkg-config

# Install GTest and GMock
RUN apt install -y libgtest-dev libgmock-dev

# Install Google benchmark
RUN apt install -y libbenchmark-dev

# Install Blaze
RUN git clone https://bitbucket.org/blaze-lib/blaze.git
RUN cd blaze && cmake -DBLAZE_BLAS_MODE=True -DBLAZE_BLAS_USE_MATRIX_MATRIX_MULTIPLICATION=False \
    -DBLAZE_BLAS_USE_MATRIX_VECTOR_MULTIPLICATION=False -DBLAZE_VECTORIZATION=False -DBLAZE_SHARED_MEMORY_PARALLELIZATION=False . && make install

# Install Eigen3
RUN apt install -y libeigen3-dev

# Install blasfeo
RUN apt-get install -y bc
RUN git clone https://github.com/giaf/blasfeo.git
RUN cd blasfeo && git checkout cc90e146ee9089de518f57dbb736e064bd82394e
COPY docker/aarch64/blasfeo/Makefile.rule blasfeo
RUN cd blasfeo && make -j `nproc` static_library && make install_static

# Install xsimd
RUN apt install -y libxsimd-dev

# Install Clang-18
RUN apt install -y clang-18
ENV CC=clang-18
ENV CXX=clang++-18

# Build blast
WORKDIR /blast
COPY bench ./bench
COPY cmake ./cmake
COPY include ./include
COPY test ./test
COPY CMakeLists.txt .
ENV PKG_CONFIG_PATH=/usr/local/lib
RUN cmake -B build -DCMAKE_CXX_FLAGS="-march=native -DXSIMD_DEFAULT_ARCH='neon64'" -DBLAST_WITH_TEST=ON -DBLAST_WITH_BENCHMARK=ON .
RUN cd build && make -j `nproc`
