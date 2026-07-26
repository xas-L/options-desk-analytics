cmake_minimum_required(VERSION 3.15)
project(odx_cpp_pricer)

# Requires Python3 and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Use pybind11 cmake config path provided by pip installation
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR "${pybind11_DIR}" CACHE PATH "Path to pybind11 cmake module")

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(bs_pricer_cpp bs.cpp)

# Statically link MinGW runtime so the .pyd doesn't need libstdc++/libgcc/libwinpthread on PATH
if (MINGW)
    target_link_options(bs_pricer_cpp PRIVATE -static-libgcc -static-libstdc++ -static -lwinpthread)
endif()

# Define _USE_MATH_DEFINES for MSVC to ensure math constants (like M_PI) are available
target_compile_definitions(bs_pricer_cpp PRIVATE _USE_MATH_DEFINES)
target_compile_features(bs_pricer_cpp PRIVATE cxx_std_17)

# Direct the compiled library to the python src folder
set_target_properties(bs_pricer_cpp PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../src/odx/pricers"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../src/odx/pricers"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/../src/odx/pricers"
    PREFIX ""
)
