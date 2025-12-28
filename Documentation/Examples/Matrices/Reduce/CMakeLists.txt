set(COMMON_EXAMPLES
    MatrixExample_reduceAllRows
    MatrixExample_reduceRows
    MatrixExample_reduceAllRowsWithArgument
    MatrixExample_reduceRowsWithArgument
    MatrixExample_reduceAllRowsIf
    MatrixExample_reduceRowsIf
    MatrixExample_reduceAllRowsWithArgumentIf
    MatrixExample_reduceRowsWithArgumentIf
)

if(TNL_BUILD_CUDA)
    foreach(target IN ITEMS ${COMMON_EXAMPLES})
        add_executable(${target} ${target}.cu)
        target_link_libraries(${target} PUBLIC TNL::TNL)
        add_custom_command(
            COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
            OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out DEPENDS ${target}
        )
        set(DOC_OUTPUTS ${DOC_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out)
    endforeach()
else()
    foreach(target IN ITEMS ${COMMON_EXAMPLES})
        add_executable(${target} ${target}.cpp)
        target_link_libraries(${target} PUBLIC TNL::TNL)
        add_custom_command(
            COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
            OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out DEPENDS ${target}
        )
        set(DOC_OUTPUTS ${DOC_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out)
    endforeach()
endif()

add_custom_target(RunMatricesReduceExamples ALL DEPENDS ${DOC_OUTPUTS})
