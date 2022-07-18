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
   ADD_CUSTOM_TARGET( RunMultidiagonalMatricesExamples ALL DEPENDS ${CUDA_OUTPUTS} )
ELSE()
   ADD_CUSTOM_TARGET( RunMultidiagonalMatricesExamples ALL DEPENDS ${HOST_OUTPUTS} )
ENDIF()
