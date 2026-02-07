add_subdirectory(DenseMatrix)
add_subdirectory(LambdaMatrix)
add_subdirectory(MultidiagonalMatrix)
add_subdirectory(SparseMatrix)
add_subdirectory(TridiagonalMatrix)

if(TNL_BUILD_CUDA)
    add_executable(MatrixWriterReaderExample MatrixWriterReaderExample.cu)
    target_link_libraries(MatrixWriterReaderExample PUBLIC TNL::TNL)
    add_custom_command(
        COMMAND
        MatrixWriterReaderExample
        >
        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MatrixWriterReaderExample.out
        OUTPUT
        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MatrixWriterReaderExample.out
        DEPENDS
        MatrixWriterReaderExample
    )
else()
    add_executable(MatrixWriterReaderExample MatrixWriterReaderExample.cpp)
    target_link_libraries(MatrixWriterReaderExample PUBLIC TNL::TNL)
    add_custom_command(
        COMMAND
        MatrixWriterReaderExample
        >
        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MatrixWriterReaderExample.out
        OUTPUT
        ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MatrixWriterReaderExample.out
        DEPENDS
        MatrixWriterReaderExample
    )
endif()
set(DOC_OUTPUTS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/MatrixWriterReaderExample.out)

add_custom_target(RunMatricesExamples ALL DEPENDS ${DOC_OUTPUTS})

# add the dependency to the main target
add_dependencies(run-doc-examples RunMatricesExamples)
