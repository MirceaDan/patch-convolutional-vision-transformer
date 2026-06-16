cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
project(patchcvt_inference LANGUAGES CXX)

# =========================
# Compiler settings
# =========================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

SET(TEST_MODE False)
if (TEST_MODE)
    add_compile_definitions(TEST_MODE=1)
else()
    add_compile_definitions(TEST_MODE=0)
endif()

# =========================
# OpenCV
# =========================
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")

# =========================
# JsonCpp
# =========================
find_package(jsoncpp REQUIRED)

# =========================
# LibTorch
# =========================set(Torch_DIR
set(Torch_DIR ${CMAKE_SOURCE_DIR}/thirdparty/libtorch/share/cmake/Torch)    
find_package(Torch REQUIRED)
message(STATUS "Found Torch")

# =========================
# LibCamera
# =========================
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCAMERA REQUIRED libcamera)
message(STATUS "Found LibCamera")

include_directories(
    ${LIBCAMERA_INCLUDE_DIRS}
)

link_directories(
    ${LIBCAMERA_LIBRARY_DIRS}
)

# =========================
# Executable
# =========================
if (TEST_MODE)    
    add_executable(patchcvt_test
        src/main.cpp
        inference/inference.cpp
        test/test.cpp
    )

    # =========================
    # Include paths
    # =========================
    target_include_directories(patchcvt_test PRIVATE
        ${OpenCV_INCLUDE_DIRS}
    )

    # =========================
    # Link libraries
    # =========================
    target_link_libraries(patchcvt_test
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
        jsoncpp
    )

    # =========================
    # Torch compile flags
    # =========================
    set_property(TARGET patchcvt_test PROPERTY CXX_STANDARD 17)

    # Fix for Raspberry Pi + GCC
    target_compile_definitions(patchcvt_test PRIVATE
        -D_GLIBCXX_USE_CXX11_ABI=1
    )

    # =========================
    # RPATH (important on Pi)
    # =========================
    set_target_properties(patchcvt_test PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
    )
else()
    add_executable(patchcvt
        src/main.cpp
        camera/camera.cpp
        inference/inference.cpp
    )

    # =========================
    # Include paths
    # =========================
    target_include_directories(patchcvt PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        camera
    )

    # =========================
    # Link libraries
    # =========================
    target_link_libraries(patchcvt
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
        ${LIBCAMERA_LDFLAGS}
        jsoncpp
    )

    # =========================
    # Torch compile flags
    # =========================
    set_property(TARGET patchcvt PROPERTY CXX_STANDARD 17)

    # Fix for Raspberry Pi + GCC
    target_compile_definitions(patchcvt PRIVATE
        -D_GLIBCXX_USE_CXX11_ABI=1
    )

    # =========================
    # RPATH (important on Pi)
    # =========================
    set_target_properties(patchcvt PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
    )
endif()


