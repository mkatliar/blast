#
# test
#
find_package(GTest 1.9.0 REQUIRED)
find_package(Threads REQUIRED)    # required for gtest
find_package(blaze 3.5 REQUIRED)

# ~~~~~~~~~~ Our tests ~~~~~~~~~~~~~~
add_executable(test-blazefeo
    RegisterMatrixTest.cpp
    StaticPanelMatrixTest.cpp
    SubmatrixTest.cpp
    DynamicPanelMatrixTest.cpp
    GemmTest.cpp
    Gemm2Test.cpp
    PotrfTest.cpp

    simd/HsumTest.cpp
)

target_link_libraries(test-blazefeo PRIVATE
    blazefeo
    blaze::blaze
    GTest::GTest
    GTest::Main
    gmock
    openblas
    ${CMAKE_THREAD_LIBS_INIT}    # Needed for gtest
)

target_compile_definitions(test-blazefeo 
    PRIVATE "BLAZE_USER_ASSERTION=1;BLAZE_INTERNAL_ASSERTION=1"
)

gtest_discover_tests(test-blazefeo)

if (BLAZEFEO_WITH_BLASFEO)
    add_subdirectory("blasfeo")
endif ()