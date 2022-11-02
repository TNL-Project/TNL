ADD_SUBDIRECTORY( DenseMatrix )
ADD_SUBDIRECTORY( LambdaMatrix )
ADD_SUBDIRECTORY( MultidiagonalMatrix )
ADD_SUBDIRECTORY( SparseMatrix )
ADD_SUBDIRECTORY( TridiagonalMatrix )

IF( TNL_BUILD_CUDA )
   ADD_EXECUTABLE( MatrixWriterReaderExample MatrixWriterReaderExample.cu )
   target_link_libraries( MatrixWriterReaderExample PUBLIC TNL::TNL_CUDA )
   ADD_CUSTOM_COMMAND( COMMAND MatrixWriterReaderExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MatrixWriterReaderExample.out
                       OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MatrixWriterReaderExample.out
                       DEPENDS MatrixWriterReaderExample )
ELSE()
   ADD_EXECUTABLE( MatrixWriterReaderExample MatrixWriterReaderExample.cpp )
   target_link_libraries( MatrixWriterReaderExample PUBLIC TNL::TNL_CXX )
   ADD_CUSTOM_COMMAND( COMMAND MatrixWriterReaderExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MatrixWriterReaderExample.out
                       OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MatrixWriterReaderExample.out
                       DEPENDS MatrixWriterReaderExample )
ENDIF()
set( DOC_OUTPUTS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MatrixWriterReaderExample.out )

add_custom_target( RunMatricesExamples ALL DEPENDS ${DOC_OUTPUTS} )

# add the dependency to the main target
add_dependencies( run-doc-examples RunMatricesExamples )
