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
    DynamicPanelMatrixTest.cpp
    GemmTest.cpp
    PotrfTest.cpp

    simd/HsumTest.cpp

    math/expressions/PMatTransExprTest.cpp
    math/expressions/AssignPanelDenseTest.cpp
    math/expressions/AssignDensePanelTest.cpp

    math/views/RowTest.cpp
    math/views/SubmatrixTest.cpp
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
