set( COMMON_EXAMPLES
    DenseMatrixExample_Constructor_init_list
    DenseMatrixExample_setElements
    DenseMatrixExample_getCompressedRowLengths
    DenseMatrixExample_getElementsCount
    DenseMatrixExample_getConstRow
    DenseMatrixExample_getRow
    DenseMatrixExample_setElement
    DenseMatrixExample_addElement
    DenseMatrixExample_getElement
    DenseMatrixExample_reduceRows
    DenseMatrixExample_reduceAllRows
    DenseMatrixExample_forElements
    DenseMatrixExample_forAllElements
    DenseMatrixExample_forRows
    DenseMatrixViewExample_constructor
    DenseMatrixViewExample_getCompressedRowLengths
    DenseMatrixViewExample_getElementsCount
    DenseMatrixViewExample_getConstRow
    DenseMatrixViewExample_getRow
    DenseMatrixViewExample_setElement
    DenseMatrixViewExample_addElement
    DenseMatrixViewExample_getElement
    DenseMatrixViewExample_reduceRows
    DenseMatrixViewExample_reduceAllRows
    DenseMatrixViewExample_forElements
    DenseMatrixViewExample_forRows
    DenseMatrixViewExample_forAllElements
    DenseMatrixViewExample_wrap
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

add_custom_target( RunDenseMatricesExamples ALL DEPENDS ${DOC_OUTPUTS} )

# add the dependency to the main target
add_dependencies( run-doc-examples RunDenseMatricesExamples )
