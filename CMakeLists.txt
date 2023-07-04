cmake_minimum_required(VERSION 3.15.2)
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE INTERNAL "")
project(InferLLM LANGUAGES C CXX ASM)

option(ENABLE_ASAN "Build with asan." OFF)
option(ENABLE_PROFILE "Build with profile information." OFF)
option(ENABLE_GPROF "Build with gprof enabled." OFF)
option(ENABLE_DOT "Build with Arm dotprod." OFF)
option(ENABLE_FP16 "Build with Arm FP16." OFF)
option(ENABLE_GPU "Build with GPU." OFF)
option(ENABLE_TEST "Build with TEST." OFF)
set(INFER_ARCH "auto" CACHE STRING "Build with specific ISA (x86/arm).")



if(CMAKE_TOOLCHAIN_FILE)
  message(STATUS "We are cross compiling for android.")
  add_definitions(-DINFER_ARM=1)
else()
  if(${INFER_ARCH} STREQUAL "auto")
    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64" OR ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "AMD64") 
      set(INFER_ARCH "x86")
    elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64" OR ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64")
      set(INFER_ARCH "arm")
    endif()
  endif()

  if(${INFER_ARCH} STREQUAL "x86")
    message(STATUS "current platform: x86")
    set(INFER_ARCH "x86")
    add_definitions(-DINFER_X86=1)
  elseif(${INFER_ARCH} STREQUAL "arm")
    message(STATUS "current platform: arm")
    set(INFER_ARCH "arm")
    add_definitions(-DINFER_ARM=1)
    if(ENABLE_DOT)
        message(STATUS "Enable dotprod feature in armv8.2-a")
        set(CMAKE_CXX_FLAGS " -march=armv8.2-a+dotprod ${CMAKE_CXX_FLAGS}")
    endif()
    if(ENABLE_FP16)
        message(STATUS "Enable fp16 feature in armv8.2-a")
        set(CMAKE_CXX_FLAGS " -march=armv8.2-a+fp16 ${CMAKE_CXX_FLAGS}")
    endif()
  else()
    message(STATUS "current platform: NONE")
    set(INFER_ARCH "")
  endif()
endif()

if(ENABLE_PROFILE)
  message(STATUS "Enable profile information.")
  add_definitions(-DINFER_PROFILE) 
endif()

file(GLOB_RECURSE SRC src/*.cpp src/*.h)

list(FILTER SRC EXCLUDE REGEX "src/kern/optimized/.*")
list(FILTER SRC EXCLUDE REGEX "src/kern/gpu/.*")
if (NOT ${INFER_ARCH} STREQUAL "")
  file(GLOB_RECURSE PLTSRC src/kern/optimized/${INFER_ARCH}/*)
endif()

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
add_library(InferLLM STATIC ${SRC} ${PLTSRC})

add_library(InferLLMShared SHARED  ${SRC} ${PLTSRC})
set_target_properties(InferLLMShared PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/include/model.h)

target_include_directories(InferLLM PUBLIC include src)
target_include_directories(InferLLMShared PUBLIC include src)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(CMAKE_CXX_FLAGS " -g -O0 ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS " -O3 -g ${CMAKE_CXX_FLAGS}")
endif()

if(ENABLE_GPROF)
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-pg" HAS_GPROF)
  message(WARNING "Build with gprof: ${HAS_GPROF}. Gprof may not work on darwin.")
  if(HAS_GPROF)
    set(CMAKE_CXX_FLAGS " -pg ${CMAKE_CXX_FLAGS}")
  endif()
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
