cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(o-eFMT)
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
find_library(FFTW_LIBRARY
     NAMES fftw3 fftw)
set(FFTW_LIBRARIES "${FFTW_LIBRARY}")

find_library(FFTW_libm_LIBRARY
         NAMES m)
list(APPEND FFTW_LIBRARIES "${FFTW_libm_LIBRARY}")

find_package(OpenCV REQUIRED)
MESSAGE( STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}.")
find_package(Ceres REQUIRED)

include_directories(
    include/
    ${OpenCV_INCLUDE_DIRS}
    ${fftw_path}
    /usr/include/eigen3
    ${CERES_INCLUDE_DIRS}
)

add_library(o-eFMT
    src/utility.cpp
    src/image_dft.cpp
    src/image_transforms.cpp
    src/image_registration.cpp
    src/optimizer.cpp)

target_link_libraries(o-eFMT
    ${OpenCV_LIBS}
    ${FFTW_LIBRARIES}
    ${CERES_LIBRARIES}
)

add_executable(image_main src/image_main.cpp)
target_link_libraries(image_main o-eFMT)
