set( COMMON_EXAMPLES
     SumExample
     SumExampleWithFunctional
     ProductExample
     ScalarProductExample
     ScalarProductWithFunctionalExample
     MaximumNormExample
     ComparisonExample
     UpdateAndResidueExample
     MapReduceExample-1
     MapReduceExample-2
     MapReduceExample-3
     ReductionWithArgument
     ReductionWithArgumentWithFunctional
     inclusiveScanExample
     exclusiveScanExample
     inplaceInclusiveScanExample
     inplaceExclusiveScanExample
     SegmentedScanExample
)

if( BUILD_CUDA )
   foreach( target IN ITEMS ${COMMON_EXAMPLES} )
      cuda_add_executable( ${target} ${target}.cu )
      add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                          OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                          DEPENDS ${target} )
      set( CUDA_OUTPUTS ${CUDA_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
   endforeach()
else()
   foreach( target IN ITEMS ${COMMON_EXAMPLES} )
      add_executable( ${target} ${target}.cpp )
      add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                          OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                          DEPENDS ${target} )
      set( HOST_OUTPUTS ${HOST_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
   endforeach()
endif()


IF( BUILD_CUDA )
   ADD_CUSTOM_TARGET( RunTutorialsReductionAndScanExamples ALL DEPENDS ${CUDA_OUTPUTS} )
ELSE()
   ADD_CUSTOM_TARGET( RunTutorialsReductionAndScanExamples ALL DEPENDS ${HOST_OUTPUTS} )
ENDIF()
