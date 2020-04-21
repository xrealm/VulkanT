cmake_minimum_required(VERSION 3.15)
project(VulkanTesting)

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES
        main.cpp)

add_executable(VulkanTesting ${SOURCE_FILES})

set(BASEPATH "${CMAKE_SOURCE_DIR}/src")
include_directories("${BASEPATH}")

find_package(PkgConfig REQUIRED)

find_package(GLFW3 3.3 REQUIRED glfw3)
#pkg_search_module(GLFW3 3.3 REQUIRED glfw3)
if(GLFW3_FOUND)
    message(STATUS "Found GLFW, Including and Linking now")
    include_directories(${GLFW3_INCLUDE_DIRS})
    target_link_libraries(VulkanTesting ${GLFW3_STATIC_LIBRARIES})
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
    target_link_libraries (VulkanTesting ${Vulkan_LIBRARIES})
    message(STATUS "Vulkan Include dirs: ${Vulkan_INCLUDE_DIRS} and libs: ${Vulkan_LIBRARIES}")
endif (Vulkan_FOUND)

#set(GLM_INCLUDE_DIRS /usr/local/Cellar/glm/0.9.9.5/)
#include_directories(${GLM_INCLUDE_DIRS})
#
#set(GLFW_INCLUDE_DIRS /usr/localCellar/glfw/HEAD-e65de29/include/)
#set(GLFW_LIBRARIES /usr/localCellar/glfw/HEAD-e65de29/lib/)
#include_directories(${GLFW_INCLUDE_DIRS})
#target_link_libraries(VulkanTesting ${GLFW_LIBRARIES})

#find_package(Vulkan REQUIRED)
#target_include_directories(${PROJECT_NAME} PUBLIC ${Vulkan_INCLUDE_DIRS})
#target_link_libraries(${PROJECT_NAME} Vulkan::Vulkan)
#
#find_package(PkgConfig REQUIRED)
#pkg_search_module(GLM REQUIRED glm)
#include_directories(${GLM_INCLUDE_DIRS})
#target_link_libraries(${PROJECT_NAME} ${GLM_LIBRARY_DIRS})
#
#find_package(glfw3 3.2 REQUIRED)
#include_directories(${GLFW_INCLUDE_DIRS})
#target_link_libraries(${PROJECT_NAME} ${GLFW_LIBRARIES})