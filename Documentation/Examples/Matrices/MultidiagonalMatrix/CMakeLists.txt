set( COMMON_EXAMPLES
   MultidiagonalMatrixExample_Constructor
    MultidiagonalMatrixExample_Constructor_init_list_1
    MultidiagonalMatrixExample_Constructor_init_list_2
    MultidiagonalMatrixExample_getSerializationType
    MultidiagonalMatrixExample_setElements
    MultidiagonalMatrixExample_getCompressedRowLengths
#    MultidiagonalMatrixExample_getElementsCount
    MultidiagonalMatrixExample_getConstRow
    MultidiagonalMatrixExample_getRow
    MultidiagonalMatrixExample_setElement
    MultidiagonalMatrixExample_addElement
    MultidiagonalMatrixExample_getElement
    MultidiagonalMatrixExample_reduceRows
    MultidiagonalMatrixExample_reduceAllRows
    MultidiagonalMatrixExample_forElements
    MultidiagonalMatrixExample_forAllElements
    MultidiagonalMatrixExample_forRows
#    MultidiagonalMatrixViewExample_constructor
    MultidiagonalMatrixViewExample_getCompressedRowLengths
#    MultidiagonalMatrixViewExample_getElementsCount
    MultidiagonalMatrixViewExample_getConstRow
    MultidiagonalMatrixViewExample_getRow
    MultidiagonalMatrixViewExample_setElement
    MultidiagonalMatrixViewExample_addElement
    MultidiagonalMatrixViewExample_getElement
    MultidiagonalMatrixViewExample_reduceRows
    MultidiagonalMatrixViewExample_reduceAllRows
    MultidiagonalMatrixViewExample_forElements
    MultidiagonalMatrixViewExample_forAllElements
    MultidiagonalMatrixViewExample_forRows
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

add_custom_target( RunMultidiagonalMatricesExamples ALL DEPENDS ${DOC_OUTPUTS} )

# add the dependency to the main target
add_dependencies( run-doc-examples RunMultidiagonalMatricesExamples )
