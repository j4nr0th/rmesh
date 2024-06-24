cmake_minimum_required(VERSION 3.28)
# project(mesh C)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C)

set(CMAKE_C_STANDARD 99)

#   Load Python package
find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
set(Python_NumPy_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/venv/lib/python3.12/site-packages/numpy/_core/include")

list(APPEND MESH_FILES geometry.h
        mesh2d.c
        mesh2d.h
        err.c
        err.h
        io.c
        io.h)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(jmtx)
set_property(TARGET jmtx PROPERTY POSITION_INDEPENDENT_CODE ON)
#add_executable(mesh first_test.c ${MESH_FILES})
#target_compile_options(mesh PRIVATE -Wall -Werror -Wextra)
#target_link_libraries(mesh PRIVATE jmtx)
#target_include_directories(mesh PRIVATE jmtx/include)
#
#add_executable(bigger_test bigger_test.c ${MESH_FILES})
#target_compile_options(bigger_test PRIVATE -Wall -Werror -Wextra)
#target_link_libraries(bigger_test PRIVATE jmtx)
#target_include_directories(bigger_test PRIVATE jmtx/include)

link_directories("cmake-build-release/jmtx")
Python_add_library(_rmsh MODULE ${MESH_FILES} rmsh/core.c WITH_SOABI)
target_include_directories(_rmsh PRIVATE jmtx/include ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
target_link_libraries(_rmsh PRIVATE jmtx)

install(TARGETS _rmsh DESTINATION rmsh)
