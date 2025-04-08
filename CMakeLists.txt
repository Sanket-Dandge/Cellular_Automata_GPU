cmake_minimum_required(VERSION 3.16)
project(game_of_life)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "-arch=sm_86")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

include_directories(
  include
)

add_library(${PROJECT_NAME}_utils
  src/utils.cpp)
# target_link_libraries(${PROJECT_NAME}_utils
#   ${OPENGL_LIBRARIES}
# )

cuda_add_library(${PROJECT_NAME}_kernels
  src/kernels.cu)
target_link_libraries(${PROJECT_NAME}_kernels
  ${PROJECT_NAME}_utils
)

cuda_add_library(${PROJECT_NAME}_cuda
  src/game_of_life.cu)
target_link_libraries(${PROJECT_NAME}_cuda
  ${PROJECT_NAME}_kernels
  ${PROJECT_NAME}_utils
)

cuda_add_executable(${PROJECT_NAME}.o
  src/main.cpp)
target_link_libraries(${PROJECT_NAME}.o
  ${PROJECT_NAME}_cuda
)
