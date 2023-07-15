set( COMMON_EXAMPLES
   printSegmentsExample-1
   printSegmentsExample-2
   SegmentsExample_CSR_constructor_1
   SegmentsExample_CSR_constructor_2
   SegmentsExample_CSR_getSerializationType
   SegmentsExample_CSR_getSegmentsType
   SegmentsExample_CSR_setSegmentsSizes
   SegmentsExample_CSR_getSegmentView
   SegmentsExample_CSR_forElements
   SegmentsExample_CSR_forSegments
   SegmentsExample_CSR_sequentialForSegments
   SegmentsExample_CSR_reduceSegments
   SegmentsExample_forElements
   SegmentsExample_forSegments-1
   SegmentsExample_forSegments-2
   SegmentsExample_reduceSegments
)

if( TNL_BUILD_CUDA )
   foreach( target IN ITEMS ${COMMON_EXAMPLES} )
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

add_custom_target( RunSegmentsExamples ALL DEPENDS ${DOC_OUTPUTS} )

# add the dependency to the main target
add_dependencies( run-doc-examples RunSegmentsExamples )
