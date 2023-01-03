cmake_minimum_required(VERSION 3.14)
project(ppl.kernel.cpu)

# --------------------------------------------------------------------------- #

# options

option(PPLNN_INSTALL "install ppl headers and libs" ON)

option(PPLNN_HOLD_DEPS "don't update dependencies" OFF)

if(MSVC)
    option(PPLNN_USE_MSVC_STATIC_RUNTIME "" ON)
endif()

option(PPLNN_USE_X86_64 "" OFF)
option(PPLNN_USE_AARCH64 "" OFF)
option(PPLNN_USE_ARMV7 "" OFF)
option(PPLNN_USE_RISCV64 "" OFF)
# --------------------------------------------------------------------------- #

if(APPLE)
    message(FATAL_ERROR "MacOS is not supported currently.")
endif()

# --------------------------------------------------------------------------- #

# dependencies

include(cmake/deps.cmake)
hpcc_populate_dep(pplcommon)

# --------------------------------------------------------------------------- #

# compiler related

if(PPLNN_USE_MSVC_STATIC_RUNTIME)
    hpcc_use_msvc_static_runtime()
else()
    hpcc_use_msvc_dynamic_runtime()
endif()

# --------------------------------------------------------------------------- #

# kernels

if(PPLNN_USE_X86_64)
    include(cmake/x86.cmake)
endif()

if(PPLNN_USE_RISCV64)
    include(cmake/riscv.cmake)
endif()

if(PPLNN_USE_AARCH64 OR PPLNN_USE_ARMV7)
    include(cmake/arm.cmake)
endif()

# installations

if(PPLNN_INSTALL)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION include)
endif()

# --------------------------------------------------------------------------- #
