add_executable(test-blasfeo
    CustomMatrixTest.cpp
    DynamicMatrixTest.cpp
    SyrkTest.cpp
    SyrkPotrfTest.cpp
    PotrfTest.cpp
    GemmTest.cpp
)

target_link_libraries(test-blasfeo
    blazefeo-blasfeo
    GTest::GTest
    GTest::Main
)

gtest_discover_tests(test-blasfeo)
