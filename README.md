## Installing dependencies
- *CMake 3.9.0 or higher*. Note that Ubuntu 17.04 comes with CMake 3.7, therefore you may need to install latest CMake manually from https://cmake.org/download/
- *Boost* `sudo apt install libboost-dev`.
- *Eigen3 3.3.3 or higher*. Ubuntu 17.04 comes with Eigen3 3.3.2-1, therefore an up-to-date Eigen3 must be downloaded from http://eigen.tuxfamily.org and installed so that CMake can find it.
- *Blaze 3.3 or higher* https://bitbucket.org/blaze-lib/blaze.
- *BLASFEO* https://github.com/giaf/blasfeo (optional, only if `BLAZEFEO_WITH_BLASFEO` is selected). Select a proper target architecture by setting the `TARGET` variable in `Makefile.rule` or in `CMake`. Build and install as usual. The build system searches for BLASFEO in `/opt/blasfeo` by default.
- *Google Test* https://github.com/google/googletest must be installed and findable by the CMake build system (optional, only if `BLAZEFEO_WITH_TEST` is selected).
- *Google Benchmark* https://github.com/google/benchmark must be installed and findable by the CMake build system (optional, only if `BLAZEFEO_WITH_BENCHMARK` is selected).

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