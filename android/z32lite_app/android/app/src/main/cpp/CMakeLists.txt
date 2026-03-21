# CMakeLists.txt for Z32LITE Native Layer

cmake_minimum_required(VERSION 3.22.1)

project("z32lite")

# Link Android logging libraries
find_library(log-lib log)

# Define the shared library target
add_library(
        z32lite
        SHARED
        LlamaAndroid.cpp
)

# In a real project, you would add llama.cpp here:
# include_directories(${LLAMA_CPP_DIR})
# add_subdirectory(${LLAMA_CPP_DIR} llama.cpp)

target_link_libraries(
        z32lite
        ${log-lib}
)
