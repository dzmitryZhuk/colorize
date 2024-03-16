cmake_minimum_required(VERSION 3.1...3.27 FATAL_ERROR)

project(Colorize)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB SOURCES "src/*.cpp")
file(GLOB HEADERS "include/*.h")

include_directories(${DIR_SOURCES} "include/")

add_subdirectory(src)
add_subdirectory(include)


add_executable(${PROJECT_NAME} 
              ${SOURCES}
              ${HEADER}
)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")



# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)