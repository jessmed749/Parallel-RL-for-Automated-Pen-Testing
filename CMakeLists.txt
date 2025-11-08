cmake_minimum_required(VERSION 3.10)
project(ParallelPentestRL CXX)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pthread")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
endif()

# Optional: ThreadSanitizer for detecting data races
# Uncomment to enable:
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -g")

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/src/CPU/environment)
include_directories(${PROJECT_SOURCE_DIR}/src/CPU/queues)

# Environment library
add_library(environment
    src/CPU/environment/pentest_env.cpp
)

target_include_directories(environment PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CPU/environment
)

# Queue library
add_library(queues
    src/CPU/queues/request_queue.cpp
    src/CPU/queues/response_queue.cpp
)

target_include_directories(queues PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CPU/queues
)

# Test executable for environment
add_executable(test_env
    tests/test_env.cpp
)

target_link_libraries(test_env environment)

# Test executable for queues
add_executable(test_queues
    tests/test_queues.cpp
)

target_link_libraries(test_queues queues)

# Enable testing
enable_testing()
add_test(NAME EnvironmentTests COMMAND test_env)
add_test(NAME QueueTests COMMAND test_queues)

# Print build configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")