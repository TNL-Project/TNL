set( CPP_EXAMPLES
   ReadMeshExample
   MeshConfigurationExample
   MeshIterationExample
   GameOfLife
)
set( CUDA_EXAMPLES
   ParallelIterationCuda
   GameOfLifeCuda
)
set( ALL_EXAMPLES ${CPP_EXAMPLES} )

foreach( target IN ITEMS ${CPP_EXAMPLES} )
   add_executable( ${target} ${target}.cpp )
   target_link_libraries( ${target} PUBLIC TNL::TNL_CXX )
   add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                       OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                       DEPENDS ${target} )
   set( DOC_OUTPUTS ${DOC_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
endforeach()

if( TNL_BUILD_CUDA )
   foreach( target IN ITEMS ${CUDA_EXAMPLES} )
      add_executable( ${target} ${target}.cu )
      target_link_libraries( ${target} PUBLIC TNL::TNL_CUDA )
      add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                          OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                          DEPENDS ${target} )
      set( DOC_OUTPUTS ${DOC_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
   endforeach()
   set( ALL_EXAMPLES ${ALL_EXAMPLES} ${CUDA_EXAMPLES} )
endif()

find_package( ZLIB )
if( ZLIB_FOUND )
   foreach( target IN ITEMS ${ALL_EXAMPLES} )
      target_compile_definitions(${target} PUBLIC "-DHAVE_ZLIB")
      target_include_directories(${target} PUBLIC ${ZLIB_INCLUDE_DIRS})
      target_link_libraries(${target} PUBLIC ${ZLIB_LIBRARIES})
   endforeach()
endif()

find_package( tinyxml2 QUIET )
if( tinyxml2_FOUND )
   foreach( target IN ITEMS ${ALL_EXAMPLES} )
      target_compile_definitions(${target} PUBLIC "-DHAVE_TINYXML2")
      target_link_libraries(${target} PUBLIC tinyxml2::tinyxml2)
   endforeach()
endif()

add_custom_target( RunTutorialsMeshesExamples ALL DEPENDS ${DOC_OUTPUTS} )

# add the dependency to the main target
add_dependencies( run-doc-examples RunTutorialsMeshesExamples )


set( DATA_FILES
   example-triangles.vtu
   grid-100x100.vtu
)

foreach( file IN ITEMS ${DATA_FILES} )
   configure_file(${file} ${CMAKE_CURRENT_BINARY_DIR}/${file} COPYONLY)
endforeach()
