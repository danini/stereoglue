# Find Pybind11
find_package(pybind11 REQUIRED)

# Add Python module
pybind11_add_module(pystereoglue MODULE pybind_module.cpp pybind_functions.cpp)

# Link the C++ library to the Python module
target_link_libraries(pystereoglue PRIVATE StereoGlue)


