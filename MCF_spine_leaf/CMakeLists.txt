cmake_minimum_required(VERSION 3.26)
project(MCF_spine_leaf)

set(CMAKE_CXX_STANDARD 23)
add_executable(MCF_spine_leaf main.cpp fort.c)
include_directories("graaf/include")
