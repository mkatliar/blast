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

    GemmStatic.cpp
    GemmDynamic.cpp
    PotrfStatic.cpp
    PotrfDynamic.cpp
)


target_link_libraries(bench-blazefeo
    blazefeo
    benchmark
    ${CMAKE_THREAD_LIBS_INIT}
)
