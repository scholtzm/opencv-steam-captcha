# Setup
cmake_minimum_required( VERSION 2.8 )
project( "Steam Captcha Solver" )

# Find OpenCV
find_package( OpenCV REQUIRED )

# Find Boost
find_package( Boost 1.39 COMPONENTS filesystem system REQUIRED )

# Create exec
add_executable( captcha src/classify.cpp src/descriptors.cpp src/image.cpp src/main.cpp src/misc.cpp src/segments.cpp )

# Linker
target_link_libraries( captcha ${OpenCV_LIBS} )
target_link_libraries( captcha ${Boost_LIBRARIES} )
