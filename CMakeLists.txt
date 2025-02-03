cmake_minimum_required(VERSION 3.25)
project(benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)

find_package(MPI REQUIRED)
find_package(Python3 REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

add_executable(bcast src/bcast.cpp)
add_executable(allgather src/bcast.cpp)
add_executable(ibcast src/bcast.cpp)
add_executable(scatterv src/bcast.cpp)

target_link_libraries(bcast PRIVATE ${MPI_LIBRARIES})
target_link_libraries(allgather PRIVATE ${MPI_LIBRARIES})
target_link_libraries(ibcast PRIVATE ${MPI_LIBRARIES})
target_link_libraries(scatterv PRIVATE ${MPI_LIBRARIES})

enable_testing()
file(COPY ${CMAKE_SOURCE_DIR}/test/ DESTINATION ${CMAKE_BINARY_DIR})
add_test(NAME Alll COMMAND ${Python3_EXECUTABLE} ${CMAKE_BINARY_DIR}/suite.py ${CMAKE_BINARY_DIR}/custom.csv )
