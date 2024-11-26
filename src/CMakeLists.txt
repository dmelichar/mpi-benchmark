find_package(MPI REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(nlohmann_json_schema_validator REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

add_executable(main main.cpp)
add_executable(json json.cpp)

target_link_libraries(main PRIVATE ${MPI_LIBRARIES})
target_link_libraries(main PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(main PRIVATE nlohmann_json_schema_validator)

target_link_libraries(json PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(json PRIVATE nlohmann_json_schema_validator)

add_custom_target(format COMMAND clang-format -i ${SOURCE_FILES})

configure_file(schema.json ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

