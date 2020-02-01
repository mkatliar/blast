add_subdirectory("blazefeo")
add_subdirectory("libxsmm")
add_subdirectory("blas")
add_subdirectory("blaze")
add_subdirectory("eigen")

if (BLAZEFEO_WITH_BLASFEO)
    add_subdirectory("blasfeo")
endif ()