# remove existing files under the install prefix
install( CODE "MESSAGE( \"-- Removing existing include directory: ${CMAKE_INSTALL_FULL_INCLUDEDIR}/TNL/\")"
         COMPONENT headers )
install( CODE "file( REMOVE_RECURSE \"${CMAKE_INSTALL_FULL_INCLUDEDIR}/TNL/\" )"
         COMPONENT headers )

# install the header files
install( CODE "MESSAGE( \"-- Installing include directory: ${CMAKE_INSTALL_FULL_INCLUDEDIR}/TNL/\")"
         COMPONENT headers )
install( DIRECTORY TNL TYPE INCLUDE COMPONENT headers
         MESSAGE_NEVER
         FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" )

# add all subdirectories
add_subdirectory( Python )
add_subdirectory( Benchmarks )
add_subdirectory( Examples )
add_subdirectory( Tools )
add_subdirectory( UnitTests )

# collect targets defined in each subdirectory
include( get_all_targets )
get_all_targets( TNL_BENCHMARKS_TARGETS ${CMAKE_CURRENT_LIST_DIR}/Benchmarks )
get_all_targets( TNL_EXAMPLES_TARGETS ${CMAKE_CURRENT_LIST_DIR}/Examples )
get_all_targets( TNL_TOOLS_TARGETS ${CMAKE_CURRENT_LIST_DIR}/Tools )
get_all_targets( TNL_TESTS_TARGETS ${CMAKE_CURRENT_LIST_DIR}/UnitTests )

# add utility targets for each subdirectory
add_custom_target( benchmarks )
if( TNL_BENCHMARKS_TARGETS )
   add_dependencies( benchmarks ${TNL_BENCHMARKS_TARGETS} )
endif()

add_custom_target( examples )
if( TNL_EXAMPLES_TARGETS )
   add_dependencies( examples ${TNL_EXAMPLES_TARGETS} )
endif()

add_custom_target( tools )
if( TNL_TOOLS_TARGETS )
   add_dependencies( tools ${TNL_TOOLS_TARGETS} )
endif()

add_custom_target( tests )
if( TNL_TESTS_TARGETS )
   add_dependencies( tests ${TNL_TESTS_TARGETS} )
endif()

# add utility targets for large groups of unit tests
get_all_targets( TNL_MATRIX_TESTS_TARGETS ${CMAKE_CURRENT_LIST_DIR}/UnitTests/Matrices )
set( TNL_NON_MATRIX_TESTS_TARGETS ${TNL_TESTS_TARGETS} )
list( REMOVE_ITEM TNL_NON_MATRIX_TESTS_TARGETS ${TNL_MATRIX_TESTS_TARGETS} )

add_custom_target( matrix-tests )
add_custom_target( non-matrix-tests )
if( TNL_MATRIX_TESTS_TARGETS )
   add_dependencies( matrix-tests ${TNL_MATRIX_TESTS_TARGETS} )
endif()
if( TNL_NON_MATRIX_TESTS_TARGETS )
   add_dependencies( non-matrix-tests ${TNL_NON_MATRIX_TESTS_TARGETS} )
endif()
