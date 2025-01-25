cmake_minimum_required(VERSION 3.28)
# project(mesh C)
if (DEFINED SKBUILD)
    project(${SKBUILD_PROJECT_NAME} LANGUAGES C)
else()
    project(mesh LANGUAGES C)
endif()
set(CMAKE_C_STANDARD 99)

#   Load Python package
find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
#set(Python_NumPy_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/venv/lib/python3.12/site-packages/numpy/_core/include")

execute_process(
        COMMAND "${Python_EXECUTABLE}"
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
        src/defines.h)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(jmtx)
set_property(TARGET jmtx PROPERTY POSITION_INDEPENDENT_CODE ON)

if (NOT DEFINED SKBUILD)
    enable_testing()
    list(APPEND TEST_FILES
        src/io.c
        src/io.h
    )

    add_executable(mesh tests/first_test.c ${MESH_FILES} ${TEST_FILES})
    target_compile_options(mesh PRIVATE -Wall -Werror -Wextra -Wno-unused-result)
    target_link_libraries(mesh PRIVATE jmtx)
    target_include_directories(mesh PRIVATE jmtx/include)
    add_test(NAME mesh COMMAND mesh)

    add_executable(bigger_test tests/bigger_test.c ${MESH_FILES} ${TEST_FILES})
    target_compile_options(bigger_test PRIVATE -Wall -Werror -Wextra -Wno-unused-result)
    target_link_libraries(bigger_test PRIVATE jmtx)
    target_include_directories(bigger_test PRIVATE jmtx/include)
    add_test(NAME bigger_test COMMAND bigger_test)

    add_executable(adjacent_test tests/adjacent_test.c ${MESH_FILES} ${TEST_FILES})
    target_compile_options(adjacent_test PRIVATE -Wall -Werror -Wextra -Wno-unused-result)
    target_link_libraries(adjacent_test PRIVATE jmtx)
    target_include_directories(adjacent_test PRIVATE jmtx/include)
    add_test(NAME adjacent_test COMMAND adjacent_test)

    add_executable(adjacent_test2 tests/adjacent_test2.c ${MESH_FILES} ${TEST_FILES})
    target_compile_options(adjacent_test2 PRIVATE -Wall -Werror -Wextra -Wno-unused-result)
    target_link_libraries(adjacent_test2 PRIVATE jmtx)
    target_include_directories(adjacent_test2 PRIVATE jmtx/include)
    add_test(NAME adjacent_test2 COMMAND adjacent_test2)
endif ()


python_add_library(_rmsh MODULE ${MESH_FILES} src/core.c WITH_SOABI)
target_include_directories(_rmsh PRIVATE jmtx/include ${Python_INCLUDE_DIRS} ${NumPy_INCLUDE_DIRS})
target_link_libraries(_rmsh PRIVATE jmtx)

if (DEFINED SKBUILD)
    install(TARGETS _rmsh
            DESTINATION ${SKBUILD_PROJECT_NAME}
            COMPONENT PythonModule)
endif()

if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    # When using asan, make sure to use "export LD_PRELOAD=$(gcc -print-file-name=libasan.so)"
    target_compile_options(_rmsh PRIVATE -Wall -Wextra -Werror -fno-omit-frame-pointer -g)
endif ()
