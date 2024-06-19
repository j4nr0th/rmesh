cmake_minimum_required(VERSION 3.28)
project(mesh C)

set(CMAKE_C_STANDARD 99)

list(APPEND MESH_FILES geometry.h
        mesh.c
        mesh.h
        err.c
        err.h
        io.c
        io.h)

add_executable(mesh first_test.c ${MESH_FILES})
add_subdirectory(jmtx)
target_compile_options(mesh PRIVATE -Wall -Werror -Wextra)
target_link_libraries(mesh PRIVATE jmtx)
target_include_directories(mesh PRIVATE jmtx/include)

add_executable(bigger_test bigger_test.c ${MESH_FILES})
target_compile_options(bigger_test PRIVATE -Wall -Werror -Wextra)
target_link_libraries(bigger_test PRIVATE jmtx)
target_include_directories(bigger_test PRIVATE jmtx/include)
