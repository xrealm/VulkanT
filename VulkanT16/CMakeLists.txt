cmake_minimum_required(VERSION 3.15)
project(VulkanT16)

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES
        main.cpp
        vshader.h)

add_executable(VulkanT16 ${SOURCE_FILES})

find_package(PkgConfig REQUIRED)

pkg_search_module(GLFW3 3.3 REQUIRED glfw3)
find_package(GLFW3 3.3 REQUIRED glfw3)
if(GLFW3_FOUND)
    message(STATUS "Found GLFW, Including and Linking now")
    include_directories(${GLFW3_INCLUDE_DIRS})
    target_link_libraries(VulkanT16 ${GLFW3_STATIC_LIBRARIES})
    message(STATUS "glfw Include dirs: ${GLFW3_INCLUDE_DIRS} and libs: ${GLFW3_STATIC_LIBRARIES}")
endif(GLFW3_FOUND)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

find_package(GLM REQUIRED)
if (GLM_FOUND)
    include_directories(${GLM_INCLUDE_DIRS})
endif (GLM_FOUND)

find_package(Vulkan REQUIRED)
if (Vulkan_FOUND)
    message(STATUS "Found Vulkan, Including and Linking now")
    include_directories(${Vulkan_INCLUDE_DIRS})
    target_link_libraries (VulkanT16 ${Vulkan_LIBRARIES})
    message(STATUS "Vulkan Include dirs: ${Vulkan_INCLUDE_DIRS} and libs: ${Vulkan_LIBRARIES}")
endif (Vulkan_FOUND)