# Copyright 2020 Mikhail Katliar. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

add_executable(test-blasfeo
    CustomMatrixTest.cpp
    DynamicMatrixTest.cpp
    SyrkTest.cpp
    SyrkPotrfTest.cpp
    PotrfTest.cpp
    GemmTest.cpp
)

target_link_libraries(test-blasfeo
    blast-blasfeo
    GTest::GTest
    GTest::Main
)

gtest_discover_tests(test-blasfeo)
