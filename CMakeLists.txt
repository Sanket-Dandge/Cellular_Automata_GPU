cmake_minimum_required(VERSION 3.18)
project(game_of_life LANGUAGES CXX CUDA)

set(CUDA_NVCC_FLAGS "-arch=sm_86")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

include_directories(
  include
)

add_library(${PROJECT_NAME}_utils
  src/utils.cpp)

add_library(${PROJECT_NAME}_kernels
  src/kernels.cu)
target_link_libraries(${PROJECT_NAME}_kernels
  ${PROJECT_NAME}_utils
)

add_library(${PROJECT_NAME}_gol
  src/game_of_life.cpp)
target_link_libraries(${PROJECT_NAME}_gol
  ${PROJECT_NAME}_kernels
  ${PROJECT_NAME}_utils
)

add_executable(${PROJECT_NAME}.o
  src/test.cpp)
target_link_libraries(${PROJECT_NAME}.o
  ${PROJECT_NAME}_cuda
  ${PROJECT_NAME}_cuda
)
