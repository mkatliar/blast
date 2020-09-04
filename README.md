# BlazeFEO
**BlazeFEO** is an extension to [Blaze](https://bitbucket.org/blaze-lib/blaze) library that provides a highly efficient implementation of linear algebra routines.
It combines the ideas of another high-performance linear algebra library [BLASFEO](https://github.com/giaf/blasfeo), such as register blocking and efficient usage of cache memory, with the C++ template programming approach.
Unlike the **Blaze** syntax based on expression templates, **BlazeFEO** provides a BLAS-like function-based interface, such that the computational complexity and the overhead of the performed operations can be easily inferred.
The **BlazeFEO** implementation is single-threaded and intended for the matrices of small and medium size (a few hundred rows/columns), which is common for embedded control applications.

## Installing dependencies
- *CMake 3.10 or higher*.
- *Boost* `sudo apt install libboost-dev`.
- *Blaze 3.3 or higher* https://bitbucket.org/blaze-lib/blaze.
- *BLASFEO* https://github.com/giaf/blasfeo (optional, only if `BLAZEFEO_WITH_BLASFEO` is selected). Select a proper target architecture by setting the `TARGET` variable in `Makefile.rule` or in `CMake`. Build and install as usual. The build system searches for BLASFEO in `/opt/blasfeo` by default.
- *Google Test* https://github.com/google/googletest must be installed and findable by the CMake build system (optional, only if `BLAZEFEO_WITH_TEST` is selected).
- *Google Benchmark* https://github.com/google/benchmark must be installed and findable by the CMake build system (optional, only if `BLAZEFEO_WITH_BENCHMARK` is selected).
- *Eigen3 3.3.7 or higher* (optional, only for benchmarks).

## Building
1. Install the dependencies.
2. Assuming that you are in the `blazefeo` source root, do  

    ```bash
    mkdir build && cd build
    ```
3. Run CMake  
```bash
cmake -DBLAZEFEO_WITH_BLASFEO=ON ..
```
4. Build  
```bash
make -j 10
```
5. Run tests  
```bash
ctest
```