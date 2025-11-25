cmake_minimum_required(VERSION 3.5)
project(fast_cd_pyb
  DESCRIPTION
  "Fast CD Python Bindings"
)

# Set architecture based on current platform
# This allows the extension to be built for the native architecture (arm64 for Apple Silicon, x86_64 for Intel Macs)
if(APPLE)
    # Only set if not already specified (allows override from setup.py)
    if(NOT CMAKE_OSX_ARCHITECTURES)
        # Detect architecture from CMAKE_SYSTEM_PROCESSOR or default to native
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
            set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE STRING "" FORCE)
        else()
            # Default to native architecture if unknown
            execute_process(
                COMMAND uname -m
                OUTPUT_VARIABLE ARCH
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            set(CMAKE_OSX_ARCHITECTURES "${ARCH}" CACHE STRING "" FORCE)
        endif()
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE "Release")
set(EXT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#get fast_cd library... should really put this in deps and separate the official python bindings from this repo

add_subdirectory(./deps/fast_cd)
add_subdirectory(./deps/pybind11/)

pybind11_add_module(fast_cd_pyb
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/core.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/viewer.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/igl.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/sim.cpp"
)



target_include_directories(fast_cd_pyb PUBLIC ./deps/fast_cd/src/include/)
target_include_directories(fast_cd_pyb PUBLIC ./src/include/)
target_include_directories(fast_cd_pyb PUBLIC ./src/cpp/)

target_link_libraries(fast_cd_pyb PUBLIC fast_cd)

set_target_properties(fast_cd_pyb PROPERTIES LINKER_LANGUAGE CXX)
set_property(TARGET fast_cd_pyb PROPERTY CXX_STANDARD 17)
