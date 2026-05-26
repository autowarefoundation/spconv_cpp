cmake_minimum_required(VERSION 3.20)

project(thrust_replacement LANGUAGES CXX)

add_library(thrust_replacement INTERFACE)
add_library(thrust_replacement::thrust_replacement ALIAS thrust_replacement)

target_include_directories(
  thrust_replacement
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)

target_compile_features(thrust_replacement INTERFACE cxx_std_14)
