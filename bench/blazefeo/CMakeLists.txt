find_package(Threads REQUIRED)


#~~~~~~~~~~~~~~~~
add_executable(bench-blazefeo
    Main.cpp

    math/expressions/AssignDensePanel.cpp
    math/expressions/AssignPanelDense.cpp
    
    register_matrix/Ger.cpp
    register_matrix/Store.cpp
    register_matrix/PartialStore.cpp
    register_matrix/Potrf.cpp
    register_matrix/Trsm.cpp

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
