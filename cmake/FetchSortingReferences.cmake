include(FetchContent)

# IPS²Ra (In-place Parallel Super Scalar Radix Sort) and IPS⁴o (In-place
# Parallel Super Scalar Samplesort) are header-only libraries. Only their
# headers are used: SOURCE_SUBDIR points to a directory without CMakeLists.txt
# so that their own CMake configuration (which adds -march=native to the global
# compiler flags) is not processed.
FetchContent_Declare(
    ips2ra
    GIT_REPOSITORY https://github.com/ips4o/ips2ra.git
    GIT_TAG 83f6b76ca308b277b2acc3a36426a3baa3c9cb73
    GIT_SUBMODULES ""
    SOURCE_SUBDIR do-not-configure
    SYSTEM
    EXCLUDE_FROM_ALL
)
FetchContent_Declare(
    ips4o
    GIT_REPOSITORY https://github.com/ips4o/ips4o.git
    GIT_TAG 08a5b926ee65cef19139057c6bde02bb5542c1cb
    SOURCE_SUBDIR do-not-configure
    SYSTEM
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(ips2ra ips4o)

# Set variables for use in the sorting benchmark
set(SORTING_REFERENCES_INCLUDE_DIRS "${ips2ra_SOURCE_DIR}/include" "${ips4o_SOURCE_DIR}/include")
set(SORTING_REFERENCES_FOUND TRUE)

message(STATUS "Sorting reference libraries (ips2ra, ips4o) downloaded to: ${ips2ra_SOURCE_DIR}, ${ips4o_SOURCE_DIR}")
