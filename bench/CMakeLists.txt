# Copyright 2020 Mikhail Katliar. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

add_subdirectory("common")

option(BLAST_BUILD_BLAST_BENCHMARK "Build BLAST benchmark")
option(BLAST_BUILD_LIBXSMM_BENCHMARK "Build LIBXSMM benchmark")
option(BLAST_BUILD_BLAS_BENCHMARK "Build benchmarks for all installed BLAS libraries")
option(BLAST_BUILD_BLAZE_BENCHMARK "Build Blaze benchmark")
option(BLAST_BUILD_EIGEN_BENCHMARK "Build Eigen benchmark")
option(BLAST_BUILD_BLASFEO_BENCHMARK "Build BLASFEO benchmark")

if (BLAST_BUILD_BLAST_BENCHMARK)
    add_subdirectory("blast")
endif ()

if (BLAST_BUILD_LIBXSMM_BENCHMARK)
    add_subdirectory("libxsmm")
endif ()

if (BLAST_BUILD_BLAS_BENCHMARK)
    add_subdirectory("blas")
endif ()

if (BLAST_BUILD_BLAZE_BENCHMARK)
    add_subdirectory("blaze")
endif ()

if (BLAST_BUILD_EIGEN_BENCHMARK)
    add_subdirectory("eigen")
endif ()

if (BLAST_BUILD_BLASFEO_BENCHMARK AND BLAST_WITH_BLASFEO)
    add_subdirectory("blasfeo")
endif ()