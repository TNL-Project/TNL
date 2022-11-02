ADD_SUBDIRECTORY( Algorithms )
ADD_SUBDIRECTORY( Containers )
ADD_SUBDIRECTORY( Pointers )
ADD_SUBDIRECTORY( Matrices )
ADD_SUBDIRECTORY( Meshes )
ADD_SUBDIRECTORY( Solvers )

set( COMMON_EXAMPLES )

set( CUDA_EXAMPLES
   FileExampleCuda
)

set( HOST_EXAMPLES
   FileExample
   FileExampleSaveAndLoad
   FileNameExample
   FileNameExampleDistributedSystemNodeCoordinates
   FileNameExampleDistributedSystemNodeId
   ObjectExample_getType
   ParseObjectTypeExample
   StringExample
   StringExampleGetAllocatedSize
   StringExampleReplace
   StringExampleSplit
   StringExampleSetSize
   StringExampleStrip
   TimerExample
   TimerExampleLogger )

if( TNL_BUILD_CUDA )
   foreach( target IN ITEMS ${COMMON_EXAMPLES} ${CUDA_EXAMPLES} )
      add_executable( ${target} ${target}.cu )
      target_link_libraries( ${target} PUBLIC TNL::TNL_CUDA )
      add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                          OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                          DEPENDS ${target} )
      set( DOC_OUTPUTS ${DOC_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
   endforeach()
else()
   foreach( target IN ITEMS ${COMMON_EXAMPLES} )
      add_executable( ${target} ${target}.cpp )
      target_link_libraries( ${target} PUBLIC TNL::TNL_CXX )
      add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                          OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                          DEPENDS ${target} )
      set( DOC_OUTPUTS ${DOC_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
   endforeach()
endif()

foreach( target IN ITEMS ${HOST_EXAMPLES} )
   add_executable( ${target} ${target}.cpp )
   target_link_libraries( ${target} PUBLIC TNL::TNL_CXX )
   add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                       OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                       DEPENDS ${target} )
   set( DOC_OUTPUTS ${DOC_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
endforeach()

add_custom_target( RunExamples ALL DEPENDS ${DOC_OUTPUTS} )

# add the dependency to the main target
add_dependencies( run-doc-examples RunExamples )
