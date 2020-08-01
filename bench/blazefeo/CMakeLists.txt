find_package(Threads REQUIRED)


#~~~~~~~~~~~~~~~~
add_executable(bench-blazefeo
    Main.cpp

    math/expressions/AssignDensePanel.cpp
    math/expressions/AssignPanelDense.cpp
    
    math/simd/Ger.cpp
    math/simd/Store.cpp
    math/simd/PartialStore.cpp
    math/simd/Potrf.cpp
    math/simd/Trsm.cpp
    math/simd/Trmm.cpp

    math/dense/DynamicSyrk.cpp
    math/dense/StaticSyrk.cpp
    math/dense/DynamicGemm.cpp
    math/dense/StaticGemm.cpp
    math/dense/StaticPotrf.cpp
    math/dense/StaticTrmm.cpp

    math/panel/StaticGemm.cpp
    math/panel/DynamicGemm.cpp    
    math/panel/StaticPotrf.cpp
    math/panel/DynamicPotrf.cpp
)


target_link_libraries(bench-blazefeo
    bench-blazefeo-common
    ${CMAKE_THREAD_LIBS_INIT}
)
