cmake_minimum_required(VERSION 3.25)
project(project)

set(CMAKE_CXX_STANDARD 17)

# Create a target for the installation of Python requirements
add_custom_target(install_python_requirements
        COMMAND ${CMAKE_COMMAND} -E echo "Installing Python requirements..."
        COMMAND python -m pip install -r ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
        COMMENT "Installing Python requirements")

add_executable(project main.cpp)

# Add the install_python_requirements target as a dependency to your main target
add_dependencies(project install_python_requirements)

# Create a target for running the Python plot script
add_custom_target(plot
        COMMAND ${CMAKE_COMMAND} -E echo "Running Python plot script..."
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/plot.py
        COMMENT "Running Python plot script")

# Create a target for clean-all that also removes CMake generated files
add_custom_target(clean-all
        COMMAND ${CMAKE_BUILD_TOOL} clean
        COMMAND ${CMAKE_COMMAND} -E remove_directory CMakeFiles
        COMMAND ${CMAKE_COMMAND} -E remove CMakeCache.txt cmake_install.cmake Makefile
        COMMAND ${CMAKE_COMMAND} -E remove results/dsgd_results.csv results/fsgd_results.csv results/dsgd_results_hot.csv
        COMMENT "Removing all build files and CMake generated files")
