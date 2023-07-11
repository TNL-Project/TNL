set( CPP_EXAMPLES
   NDArrayExample_RowMajorArray
   NDArrayExample_3Dpermutations
   NDArrayExample_HalfStaticArray
   NDArrayExample_SlicedArray
)

foreach( target IN ITEMS ${CPP_EXAMPLES} )
   add_executable( ${target} ${target}.cpp )
   target_link_libraries( ${target} PUBLIC TNL::TNL_CXX )
   add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                       OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                       DEPENDS ${target} )
   set( DOC_OUTPUTS ${DOC_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
endforeach()

add_custom_target( RunUGNDArraysExamples ALL DEPENDS ${DOC_OUTPUTS} )

# add the dependency to the main target
add_dependencies( run-doc-examples RunUGNDArraysExamples )
