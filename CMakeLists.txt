cmake_minimum_required(VERSION 3.28)
project(mesh C)

set(CMAKE_C_STANDARD 99)

add_executable(mesh main.c
        geometry.h
        mesh.c
        mesh.h
        err.c
        err.h
        io.c
        io.h
)
add_subdirectory(jmtx)
#target_compile_options(mesh PRIVATE -Wall -Werror -Wextra)
target_link_libraries(mesh PRIVATE jmtx)
target_include_directories(mesh PRIVATE jmtx/include)