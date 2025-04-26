cmake_minimum_required(VERSION 3.18)
project(cyclic_ca LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 75)

# set(CUDA_NVCC_FLAGS "-arch=sm_86")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

include_directories(
  include
)

add_library(${PROJECT_NAME}_utils
  src/utils.cpp)

add_library(${PROJECT_NAME}_kernels
  src/kernels.cu
  src/kernels/wireworld.cu
)
target_link_libraries(${PROJECT_NAME}_kernels
  ${PROJECT_NAME}_utils
)

add_library(${PROJECT_NAME}_gol
  src/cyclic_ca.cpp)
target_link_libraries(${PROJECT_NAME}_gol
  ${PROJECT_NAME}_kernels
  ${PROJECT_NAME}_utils
)

add_library(${PROJECT_NAME}_ww
  src/wireworld.cpp)
target_link_libraries(${PROJECT_NAME}_ww
  ${PROJECT_NAME}_kernels
  ${PROJECT_NAME}_utils
)

add_executable(${PROJECT_NAME}.o
  src/test.cpp)
target_link_libraries(${PROJECT_NAME}.o
  ${PROJECT_NAME}_gol
  ${PROJECT_NAME}_ww
)
