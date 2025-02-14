cmake_minimum_required(VERSION 3.20)

project(spconv LANGUAGES CXX CUDA VERSION 2.3.8)

# Set installation directories
include(GNUInstallDirs)

# Create a version file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/spconvConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Configure the package config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/spconvConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/spconvConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/spconv
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(src)
# tell parent spconv2 include path.
set(SPCONV2_INCLUDE_PATH ${${PROJECT_NAME}_SOURCE_DIR}/include PARENT_SCOPE)

# Install the shared library
install(TARGETS spconv
    EXPORT spconvTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}  # If you also have a static library
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}  # If you have executables
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install the header files
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export the targets
install(EXPORT spconvTargets
    FILE spconvTargets.cmake
    NAMESPACE spconv::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/spconv
)

# Install the config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/spconvConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/spconvConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/spconv
)

include(cmake/Packing.cmake)