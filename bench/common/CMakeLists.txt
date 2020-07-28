find_package(benchmark REQUIRED)

add_library(bench-blazefeo-common STATIC
    Benchmark.cpp
)

target_link_libraries(bench-blazefeo-common
    PUBLIC benchmark
    ${CMAKE_THREAD_LIBS_INIT}
)