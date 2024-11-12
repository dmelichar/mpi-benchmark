cmake_minimum_required(VERSION 3.25)
project(benchmark)
set(CMAKE_CXX_STANDARD 20)

find_package(MPI REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(nlohmann_json_schema_validator REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

add_executable(main main.cpp)

target_link_libraries(main PRIVATE ${MPI_LIBRARIES})
target_link_libraries(main PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(main PRIVATE nlohmann_json_schema_validator)

add_custom_target(format COMMAND clang-format -i ${SOURCE_FILES})

