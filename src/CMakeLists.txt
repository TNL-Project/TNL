if( TNL_INSTALL_HEADERS )
   install( DIRECTORY TNL TYPE INCLUDE
            MESSAGE_NEVER
            FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" )

   add_subdirectory( Python )
endif()

if( TNL_BUILD_BENCHMARKS )
   add_subdirectory( Benchmarks )
endif()

if( TNL_BUILD_EXAMPLES )
   add_subdirectory( Examples )
endif()

if( TNL_BUILD_TOOLS )
   add_subdirectory( Tools )
endif()

if( TNL_BUILD_TESTS )
   add_subdirectory( UnitTests )
endif()
if( TNL_BUILD_MATRIX_TESTS )
   add_subdirectory( UnitTests/Matrices )
endif()
