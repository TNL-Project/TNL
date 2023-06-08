set( COMMON_EXAMPLES
   SparseMatrixExample_Constructor_init_list_1
   SparseMatrixExample_Constructor_init_list_2
   SparseMatrixExample_Constructor_rowCapacities_vector
   SparseMatrixExample_Constructor_std_map
   SparseMatrixExample_getSerializationType
   SparseMatrixExample_setRowCapacities
   SparseMatrixExample_setElements
   SparseMatrixExample_setElements_map
   SparseMatrixExample_getCompressedRowLengths
   SparseMatrixExample_addElement
   SparseMatrixExample_getElement
   SparseMatrixExample_reduceRows
   SparseMatrixExample_reduceAllRows
   SparseMatrixExample_forElements
   SparseMatrixExample_forAllElements
   SparseMatrixExample_forRows
   SparseMatrixViewExample_getConstRow
   SparseMatrixViewExample_getRow
   SparseMatrixViewExample_setElement
   SparseMatrixViewExample_wrapCSR
   SparseMatrixViewExample_wrapEllpack
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

add_custom_target( RunSparseMatricesExamples ALL DEPENDS ${DOC_OUTPUTS} )

# add the dependency to the main target
add_dependencies( run-doc-examples RunSparseMatricesExamples )
