cmake_minimum_required(VERSION 3.28)
# project(mesh C)
if (DEFINED SKBUILD)
    project(${SKBUILD_PROJECT_NAME} LANGUAGES C)
else()
    project(mesh LANGUAGES C)
endif()
set(CMAKE_C_STANDARD 99)

#   Load Python package
find_package(Python 3.10 COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
#set(Python_NumPy_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/venv/lib/python3.12/site-packages/numpy/_core/include")
execute_process(
        COMMAND "${Python3_EXECUTABLE}"
        -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NumPy_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND MESH_FILES
        src/geometry.h
        src/geometry.c
        src/mesh2d.c
        src/mesh2d.h
        src/err.c
        src/err.h
        src/io.c
        src/io.h
        src/defines.h)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(jmtx)
set_property(TARGET jmtx PROPERTY POSITION_INDEPENDENT_CODE ON)

if (NOT DEFINED SKBUILD)
    enable_testing()
endif ()

add_executable(mesh tests/first_test.c ${MESH_FILES})
target_compile_options(mesh PRIVATE -Wall -Werror -Wextra)
target_link_libraries(mesh PRIVATE jmtx)
target_include_directories(mesh PRIVATE jmtx/include)

add_executable(bigger_test tests/bigger_test.c ${MESH_FILES})
target_compile_options(bigger_test PRIVATE -Wall -Werror -Wextra)
target_link_libraries(bigger_test PRIVATE jmtx)
target_include_directories(bigger_test PRIVATE jmtx/include)

add_executable(adjacent_test tests/adjacent_test.c ${MESH_FILES})
target_compile_options(adjacent_test PRIVATE -Wall -Werror -Wextra)
target_link_libraries(adjacent_test PRIVATE jmtx)
target_include_directories(adjacent_test PRIVATE jmtx/include)

add_executable(adjacent_test2 tests/adjacent_test2.c ${MESH_FILES})
target_compile_options(adjacent_test2 PRIVATE -Wall -Werror -Wextra)
target_link_libraries(adjacent_test2 PRIVATE jmtx)
target_include_directories(adjacent_test2 PRIVATE jmtx/include)

python_add_library(_rmsh MODULE ${MESH_FILES} src/core.c WITH_SOABI USE_SABI 3.10)
target_include_directories(_rmsh PRIVATE jmtx/include ${Python_INCLUDE_DIRS} ${NumPy_INCLUDE_DIRS})
target_link_libraries(_rmsh PRIVATE jmtx)

if (DEFINED SKBUILD)
    install(TARGETS _rmsh
            DESTINATION ${SKBUILD_PROJECT_NAME}
            COMPONENT PythonModule)
endif()
