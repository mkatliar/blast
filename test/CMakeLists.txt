#
# test
#
find_package(GTest 1.9.0 REQUIRED)
find_package(Threads REQUIRED)    # required for gtest
find_package(blaze 3.5 REQUIRED)

# ~~~~~~~~~~ Our tests ~~~~~~~~~~~~~~
add_executable(test-smoke
    PanelTest.cpp
)

target_link_libraries(test-smoke PRIVATE
    smoke
    blaze::blaze
    GTest::GTest
    GTest::Main
    gmock
    ${CMAKE_THREAD_LIBS_INIT}    # Needed for gtest
)

gtest_discover_tests(test-smoke)
