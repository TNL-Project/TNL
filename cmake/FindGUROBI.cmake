# FindGUROBI.cmake

# Locate Gurobi libraries and headers
# Once done, this will define:
#
# GUROBI_FOUND        - system has Gurobi
# GUROBI_INCLUDE_DIRS - the Gurobi include directories
# GUROBI_LIBRARIES    - link these to use Gurobi

if(NOT GUROBI_FOUND)
    # Search paths for headers and libraries
    set(SEARCH_PATHS
        "$ENV{GUROBI_HOME}"
        "/opt/gurobi"
        "/usr/local/gurobi"
        "C:/gurobi"
    )

    # Find the Gurobi include directory
    find_path(GUROBI_INCLUDE_DIR
        NAMES gurobi_c++.h
        HINTS ${SEARCH_PATHS}
        PATH_SUFFIXES include
    )

    # Find the Gurobi C library
    find_library(GUROBI_C_LIBRARY
        NAMES gurobi gurobi100 gurobi95 gurobi90 gurobi81
        HINTS ${SEARCH_PATHS}
        PATH_SUFFIXES lib
    )

    # Find the Gurobi C++ library
    find_library(GUROBI_CXX_LIBRARY
        NAMES gurobi_c++
        HINTS ${SEARCH_PATHS}
        PATH_SUFFIXES lib
    )

    # Set the include directories and libraries
    set(GUROBI_INCLUDE_DIRS ${GUROBI_INCLUDE_DIR})
    set(GUROBI_LIBRARIES ${GUROBI_CXX_LIBRARY} ${GUROBI_C_LIBRARY})

    # Determine if Gurobi was found
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(GUROBI DEFAULT_MSG
        GUROBI_INCLUDE_DIR GUROBI_C_LIBRARY GUROBI_CXX_LIBRARY)
endif()

# Mark variables as advanced to hide them in GUI
mark_as_advanced(
    GUROBI_INCLUDE_DIR
    GUROBI_C_LIBRARY
    GUROBI_CXX_LIBRARY
)
