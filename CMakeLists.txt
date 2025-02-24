cmake_minimum_required(VERSION 3.18.4)
project(benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)

find_package(MPI REQUIRED)
find_package(Python3 REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

add_executable(bcast src/bcast.cpp)
add_executable(allgatherv src/allgatherv.cpp)
add_executable(scatterv src/scatterv.cpp)
add_executable(gatherv src/gatherv.cpp)
add_executable(alltoallw src/alltoallw.cpp)

target_link_libraries(bcast PRIVATE ${MPI_LIBRARIES})
target_link_libraries(allgatherv PRIVATE ${MPI_LIBRARIES})
target_link_libraries(gatherv PRIVATE ${MPI_LIBRARIES})
target_link_libraries(scatterv PRIVATE ${MPI_LIBRARIES})
target_link_libraries(alltoallw PRIVATE ${MPI_LIBRARIES})

enable_testing()
file(COPY ${CMAKE_SOURCE_DIR}/test/ DESTINATION ${CMAKE_BINARY_DIR})
add_test(NAME Alll COMMAND ${Python3_EXECUTABLE} ${CMAKE_BINARY_DIR}/suite.py ${CMAKE_BINARY_DIR}/example.json)
