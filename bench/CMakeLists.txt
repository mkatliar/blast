# Copyright 2020 Mikhail Katliar. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

add_subdirectory("common")
add_subdirectory("blazefeo")
add_subdirectory("libxsmm")
add_subdirectory("blas")
add_subdirectory("blaze")
add_subdirectory("eigen")

if (BLAZEFEO_WITH_BLASFEO)
    add_subdirectory("blasfeo")
endif ()