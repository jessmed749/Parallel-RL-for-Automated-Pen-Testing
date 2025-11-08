cmake_minimum_required(VERSION 3.10)
project(ParallelPentestRL)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Environment library
add_library(environment
    src/CPU/environment/pentest_env.cpp
)

target_include_directories(environment PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CPU/environment
)

# Test executable
add_executable(test_env
    tests/test_environment.cpp
)

target_link_libraries(test_env environment)

# Enable testing
enable_testing()
add_test(NAME EnvironmentTests COMMAND test_env)