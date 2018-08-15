cmake_minimum_required(VERSION 3.5)
project(ceres_reprojection_pylib)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PYBIND11_CPP_STANDARD -std=c++14)
add_subdirectory(pybind11)
pybind11_add_module(ceres_reprojection ceres_reprojection.cpp)
target_link_libraries(ceres_reprojection PRIVATE ceres)


# Eigen
find_package(Eigen3 REQUIRED)
include_directories( ${EIGEN3_INCLUDE_DIR} )

# # Build shared library for ceres-solver
set(BUILD_SHARED_LIBS ON)
add_subdirectory(lib/external/ceres-solver)
include_directories(lib/external/ceres-solver/include)
