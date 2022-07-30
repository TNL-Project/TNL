set( CPP_EXAMPLES
   NDArrayExample_RowMajorArray
   NDArrayExample_3Dpermutations
   NDArrayExample_HalfStaticArray
   NDArrayExample_SlicedArray
)

foreach( target IN ITEMS ${CPP_EXAMPLES} )
   add_executable( ${target} ${target}.cpp )
   add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                       OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                       DEPENDS ${target} )
   set( CPP_OUTPUTS ${CPP_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
endforeach()
add_custom_target( RunTutorialsNDArraysExamples ALL DEPENDS ${CPP_OUTPUTS} )
