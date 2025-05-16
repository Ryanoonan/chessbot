cmake_minimum_required(VERSION 3.5)
project(chess_agent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add OpenMP paths based on Homebrew installation
include_directories(/opt/homebrew/opt/libomp/include)
link_directories(/opt/homebrew/opt/libomp/lib)

# Add compiler flags for OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/opt/homebrew/opt/libomp/include")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/opt/homebrew/opt/libomp/lib -lomp")

find_package(Torch REQUIRED)

# Define the source files for the chess agent
add_executable(chess_agent 
    chess_agent.cpp
)

# Include the header files
target_include_directories(chess_agent PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(chess_agent "${TORCH_LIBRARIES}")

# Set C++ standard specifically for the targets
set_property(TARGET chess_agent PROPERTY CXX_STANDARD 17)