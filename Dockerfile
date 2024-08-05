FROM ubuntu:kinetic
WORKDIR /root
RUN apt-get update
RUN apt-get upgrade -y
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    build-essential clang-15 cmake git libopenblas-dev libboost-exception-dev pkg-config python3-matplotlib

# Install GTest
RUN git clone https://github.com/google/googletest.git
RUN cd googletest && cmake -DCMAKE_BUILD_TYPE=Release . && make -j `nproc` install

# Install Google benchmark
RUN git clone https://github.com/google/benchmark.git
RUN cd benchmark && cmake -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_ENABLE_GTEST_TESTS=False . && make -j `nproc` install

# Install OpenMP
RUN apt install -y libomp-dev

# Install Blaze
RUN git clone https://bitbucket.org/blaze-lib/blaze.git
RUN cd blaze && cmake -DBLAZE_BLAS_MODE=True -DBLAZE_BLAS_USE_MATRIX_MATRIX_MULTIPLICATION=False \
    -DBLAZE_BLAS_USE_MATRIX_VECTOR_MULTIPLICATION=False -DBLAZE_VECTORIZATION=True -DCMAKE_INSTALL_PREFIX=/usr/local/ . && make install

# Install Eigen3
RUN git clone https://gitlab.com/libeigen/eigen.git
RUN mkdir -p eigen/build && cd eigen/build && cmake -DCMAKE_INSTALL_PREFIX=/usr/local/ .. && make install

# Install blasfeo
RUN apt-get install -y bc
RUN git clone https://github.com/giaf/blasfeo.git
RUN cd blasfeo && git checkout cc90e146ee9089de518f57dbb736e064bd82394e
COPY docker/blasfeo/Makefile.rule blasfeo
RUN cd blasfeo && make -j `nproc` static_library && make install_static

# Install libxsmm
RUN git clone https://github.com/hfp/libxsmm.git
RUN cd libxsmm && make -j `nproc` PREFIX=/usr/local install

# Install MKL
RUN apt-get install -y wget
RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN rm GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN wget https://apt.repos.intel.com/setup/intelproducts.list -O /etc/apt/sources.list.d/intelproducts.list
RUN apt-get update
RUN apt-get install -y intel-mkl-64bit-2020.4-912

# Build blast
COPY bench blast/bench
COPY cmake blast/cmake
COPY include blast/include
COPY test blast/test
COPY CMakeLists.txt blast
COPY Makefile blast/Makefile
ENV PKG_CONFIG_PATH=/usr/local/lib
RUN mkdir -p blast/build && cd blast/build \
    && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_CXX_COMPILER="clang++-15" \
        -DCMAKE_CXX_FLAGS="-march=native -mfma -mavx -mavx2 -msse4 -fno-math-errno -DXSIMD_DEFAULT_ARCH='fma3<avx2>'" \
        -DCMAKE_CXX_FLAGS_RELEASE="-O3 -g -DNDEBUG -ffast-math" .. \
        -DBLAST_WITH_TEST=ON \
        -DBLAST_WITH_BENCHMARK=ON \
        -DBLAST_WITH_BLASFEO=ON \
        -DBLAST_BUILD_BLAST_BENCHMARK=ON \
        -DBLAST_BUILD_LIBXSMM_BENCHMARK=ON \
        -DBLAST_BUILD_BLAS_BENCHMARK=ON \
        -DBLAST_BUILD_BLAZE_BENCHMARK=ON \
        -DBLAST_BUILD_EIGEN_BENCHMARK=ON \
        -DBLAST_BUILD_BLASFEO_BENCHMARK=ON \
    && make -j `nproc` VERBOSE=1

# Run tests
RUN cd blast/build && ctest

# Run benchmarks
ENV MKL_THREADING_LAYER=SEQUENTIAL
ENV OPENBLAS_NUM_THREADS=1
CMD mkdir -p blast/bench_result/data \
    && mkdir -p blast/bench_result/image \
    && cd blast \
    && make -j 1 bench_result/image/dgemm_performance.png bench_result/image/dgemm_performance_ratio.png
