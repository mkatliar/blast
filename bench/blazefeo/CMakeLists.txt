find_package(benchmark REQUIRED)
find_package(Threads REQUIRED)


#~~~~~~~~~~~~~~~~
add_executable(bench-blazefeo
    Main.cpp
    
    register_matrix/Ger.cpp
    register_matrix/Store.cpp
    register_matrix/PartialStore.cpp
    register_matrix/Potrf.cpp
    register_matrix/Trsm.cpp

    math/dense/StaticSyrk.cpp
    math/dense/DynamicGemm.cpp
    math/dense/StaticGemm.cpp

    math/panel/StaticPanelGemm.cpp
    math/panel/DynamicPanelGemm.cpp    
    math/panel/StaticPanelPotrf.cpp
    math/panel/DynamicPanelPotrf.cpp
)


target_link_libraries(bench-blazefeo
    blazefeo
    benchmark
    ${CMAKE_THREAD_LIBS_INIT}
)
