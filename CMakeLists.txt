cmake_minimum_required(VERSION 3.15.2)
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE INTERNAL "")
project(InferLLM LANGUAGES C CXX CUDA)



option(ENABLE_ASAN "Build with asan." OFF)
option(ENABLE_PROFILE "Build with profile information." OFF)
option(ENABLE_DOT "Build with Arm dotprod." OFF)
option(ENABLE_FP16 "Build with Arm FP16." OFF)
option(ENABLE_GPU "Build with GPU." ON)
set(INFER_ARCH x86_64 CACHE STRING "Architecture to be built.")


if(ENABLE_GPU)
  add_definitions(-DENABLE_GPU)
  find_package(CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
endif()



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

file(GLOB_RECURSE SRC src/*/*.cpp src/*/*.h src/*.cpp src/*/*.cu)
add_library(InferLLM STATIC ${SRC})

add_library(InferLLMShared SHARED  ${SRC})
set_target_properties(InferLLMShared PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/include/model.h)

target_include_directories(InferLLM PUBLIC include src)
target_include_directories(InferLLMShared PUBLIC include src)

set(CMAKE_CXX_FLAGS " -std=c++11 -pthread -Wno-multichar ${CMAKE_CXX_FLAGS}")

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

add_executable(chatglm_quantizer application/chatglm/quantizer.cpp)
target_link_libraries(chatglm_quantizer InferLLM)