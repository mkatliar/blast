# BLAST
*BLAST* is a high-performance linear algebra library that combines a BLAS-like nterface with modern C++ template metaprogramming.
*BLAST* implementation is single-threaded and intended for matrices of small and medium size (a few hundred rows/columns), which is common for embedded control applications.

The name stands for **BLAS** **T**emplates.

The figures below shows the performance of BLAS *dgemm* routine for different LA implementations on an
*Intel(R) Core(TM) i7-9850H CPU @ 2.60GHz*:
![dgemm_performance](doc/dgemm_performance.png)
![dgemm_performance_ratio](doc/dgemm_performance_ratio.png)

## C++ standard and compilers
*BLAST* uses C++20 standrd, so you need to use a compiler that supports it.

The code performance highly depends on the compiler. We recommend [Clang](https://clang.llvm.org/).

## Installing dependencies
- *CMake 3.24 or higher*.
- *Boost libraries* `sudo apt install libboost-exception-dev`.
- *Blaze 3.9 or higher* https://bitbucket.org/blaze-lib/blaze.
- *BLASFEO* https://github.com/giaf/blasfeo (optional, only if `BLAST_WITH_BLASFEO` is selected). Select a proper target architecture by setting the `TARGET` variable in `Makefile.rule` or in `CMake`. Build and install as usual. The build system searches for BLASFEO in `/opt/blasfeo` by default.
- *Google Test* https://github.com/google/googletest must be installed and findable by the CMake build system (optional, only if `BLAST_WITH_TEST` is selected).

If you want to run benchmarks, you will also need:
- *Google Benchmark* https://github.com/google/benchmark must be installed and findable by the CMake build system (optional, only if `BLAST_WITH_BENCHMARK` is selected).
- *Eigen3 3.3.7 or higher* (optional, if you want to benchmark it).
- For each installed BLAS library such as *MKL*, *OpenBLAS*, etc. a separate benchmark executable will be built, if it is found by *CMake* `FindBLAS`.
- Python3 `sudo apt install python3`
- Matplotlib `sudo apt install python3-matplotlib`


## Building
*BLAST* is a header-only library, so you don't need to build it. You can however build the tests:

1. Install the dependencies.
2. Assuming that you are in the `blast` source root, do

    ```bash
    mkdir build && cd build
    ```
3. Run CMake
```bash
cmake -DBLAST_WITH_BLASFEO=ON ..
```
4. Build
```bash
make -j 10
```
5. Run tests
```bash
ctest
```

## Using
TODO: add examples

## Benchmarks
Specify `BLAST_WITH_BENCHMARK=ON` in *CMake* configure steps if you want to build benchmarks. Benchmarks will be built in `build/bin/bench-*`. The set of benchmarks depends on the options and the installed libraries. It might look like the following:

```
$ls -1 build/bin/bench*
build/bin/bench-blas-Intel10_64_dyn
build/bin/bench-blas-Intel10_64lp_seq
build/bin/bench-blas-OpenBLAS
build/bin/bench-blasfeo
build/bin/bench-blast
build/bin/bench-blaze
build/bin/bench-blazefeo
build/bin/bench-eigen
build/bin/bench-libxsmm
```
Here `bench-blast` is the benchmark for *BLAST* itself, and the others are for other libraries. You can run all *BLAST* benchmarks by simply typing
```bash
build/bin/bench-blast
```
You can select benchmarks for specific functions using `--benchmark_filter=<regex>`.

There are a few targets in the root `Makefile` that run benchmarks and record results in JSON files. The following command runs and records `dgemm` benchmarks:
```bash
make dgemm-benchmarks
```

There are also a few `make` targets for performance plots:
```bash
make bench_result/image/dgemm_performance.png bench_result/image/dgemm_performance_ratio.png
```

You are welcome to contribute by organizing the existing benchmark, writing more benchmarks, and writing scripts to visualize the results.

## Docker image
To automate building and running benchmarks, a [Dockerfile](Dockerfile) is provided. Use the following commands to build a Docker image and run benchmarks in a Docker container:
```bash
cd blast
docker build . --tag blast_bench .
docker run -v `pwd`/bench_result/docker:/root/blast/bench_result blast_bench
```
The benchmark results will be put in `/bench_result/docker`.