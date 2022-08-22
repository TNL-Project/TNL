ADD_SUBDIRECTORY( DenseMatrix )
ADD_SUBDIRECTORY( LambdaMatrix )
ADD_SUBDIRECTORY( MultidiagonalMatrix )
ADD_SUBDIRECTORY( SparseMatrix )
ADD_SUBDIRECTORY( TridiagonalMatrix )

IF( TNL_BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( MatrixWriterReaderExample MatrixWriterReaderExample.cu )
   ADD_CUSTOM_COMMAND( COMMAND MatrixWriterReaderExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MatrixWriterReaderExample.out
                       OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MatrixWriterReaderExample.out
                       DEPENDS MatrixWriterReaderExample )
ELSE()
   ADD_EXECUTABLE( MatrixWriterReaderExample MatrixWriterReaderExample.cpp )
   ADD_CUSTOM_COMMAND( COMMAND MatrixWriterReaderExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MatrixWriterReaderExample.out
                       OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MatrixWriterReaderExample.out
                       DEPENDS MatrixWriterReaderExample )
ENDIF()

ADD_CUSTOM_TARGET( RunMatricesExamples ALL DEPENDS
   ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MatrixWriterReaderExample.out
)
