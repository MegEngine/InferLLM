cmake_minimum_required(VERSION 3.15.2)
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE INTERNAL "")
project(InferLLM LANGUAGES C CXX)

option(ENABLE_ASAN "Build with asan." OFF)
option(ENABLE_PROFILE "Build with profile information." OFF)
option(ENABLE_DOT "Build with Arm dotprod." OFF)
option(ENABLE_FP16 "Build with Arm FP16." OFF)
option(ENABLE_GPU "Build with GPU." OFF)
option(ENABLE_TEST "Build with TEST." OFF)
set(INFER_ARCH x86_64 CACHE STRING "Architecture to be built.")



if(CMAKE_TOOLCHAIN_FILE)
  message(STATUS "We are cross compiling for android.")
  add_definitions(-DINFER_ARM=1)
else()
  if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64" OR ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "AMD64") 
    message(STATUS "current platform: x86_64")
    set(CMAKE_CXX_FLAGS " -march=native ${CMAKE_CXX_FLAGS}")
    add_definitions(-DINFER_X86=1)
  elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64" OR ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64")
    message(STATUS "current platform: aarch64")
    add_definitions(-DINFER_ARM=1)
  endif()
endif()

if(ENABLE_PROFILE)
  message(STATUS "Enable profile information.")
  add_definitions(-DINFER_PROFILE) 
endif()

if(ENABLE_DOT)
    message(STATUS "Enable dotprod feature in armv8.2-a")
    set(CMAKE_CXX_FLAGS " -march=armv8.2-a+dotprod ${CMAKE_CXX_FLAGS}")
endif()

if(ENABLE_FP16)
    message(STATUS "Enable fp16 feature in armv8.2-a")
    set(CMAKE_CXX_FLAGS " -march=armv8.2-a+fp16 ${CMAKE_CXX_FLAGS}")
endif()

file(GLOB_RECURSE SRC src/*/*.cpp src/*/*.h src/*.cpp)

set(CMAKE_CXX_FLAGS " -std=c++11 -pthread -Wno-multichar -fPIC ${CMAKE_CXX_FLAGS}")

if(ENABLE_GPU)
  message(STATUS "Build with GPU.")
  add_definitions(-DENABLE_GPU=1)
  find_package(CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})

  file(GLOB_RECURSE GPU_SRC src/kern/gpu/*.cu src/kern/gpu/*.h)
  cuda_add_library(InferLLMGPU STATIC ${GPU_SRC})
  target_include_directories(InferLLMGPU PUBLIC include src)
  target_link_libraries(InferLLMGPU ${CUDA_LIBRARIES})
endif()
add_library(InferLLM STATIC ${SRC})

add_library(InferLLMShared SHARED  ${SRC})
set_target_properties(InferLLMShared PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/include/model.h)

target_include_directories(InferLLM PUBLIC include src)
target_include_directories(InferLLMShared PUBLIC include src)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(CMAKE_CXX_FLAGS " -g -O0 ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS " -O3 -g ${CMAKE_CXX_FLAGS}")
endif()

if(ENABLE_ASAN)
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS
        " -fsanitize=address -g -O0 -static-libsan ${CMAKE_CXX_FLAGS}"
    )
    set(CMAKE_LINK_FLAGS "-fsanitize=address -static-libsan ${CMAKE_LINK_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS
        " -fsanitize=address -g -O0 -static-libasan ${CMAKE_CXX_FLAGS}"
    )
    set(CMAKE_LINK_FLAGS "-fsanitize=address -static-libasan ${CMAKE_LINK_FLAGS}")
  endif()
  add_definitions(-DENABLE_ASAN)
endif()

add_executable(alpaca application/alpaca/alpaca.cpp)
target_link_libraries(alpaca InferLLM)

add_executable(chatglm application/chatglm/chatglm.cpp)
target_link_libraries(chatglm InferLLM)

add_executable(chat application/chat.cpp)
target_link_libraries(chat InferLLM)

add_executable(quantizer application/quantizer.cpp)
target_link_libraries(quantizer InferLLM)

if(ENABLE_GPU)
  target_link_libraries(InferLLM InferLLMGPU)
  target_link_libraries(InferLLMShared InferLLMGPU)
  target_link_libraries(alpaca InferLLMGPU)
  target_link_libraries(chatglm InferLLMGPU)
  target_link_libraries(quantizer InferLLMGPU)
  target_link_libraries(chat InferLLMGPU)
endif()

if(ENABLE_TEST)
  message(STATUS "Build with TEST.")
  add_subdirectory(third-party/googletest ${CMAKE_CURRENT_BINARY_DIR}/googletest EXCLUDE_FROM_ALL)

  file(GLOB TEST_SRC test/*.cpp test/CPU/*.cpp test/CPU/*.h)
  if(ENABLE_GPU)
    file(GLOB_RECURSE GPU_TEST_SRC test/GPU/*.cpp test/GPU/*.h)
    list(APPEND TEST_SRC ${GPU_TEST_SRC})
  endif()
  add_executable(InferLLMTest ${TEST_SRC})
  target_include_directories(InferLLMTest PUBLIC test inlude src)
  target_link_libraries(InferLLMTest InferLLM gtest)
endif()
