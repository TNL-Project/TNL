set(COMMON_EXAMPLES
    MatrixExample_forElements
    MatrixExample_forAllElements
    MatrixExample_forElementsWithIndexes
    MatrixExample_forElementsIf
    MatrixExample_forAllElementsIf
    MatrixExample_forRows-1
    MatrixExample_forRows-2
    MatrixExample_forAllRows
    MatrixExample_forRowsWithIndexes
    MatrixExample_forRowsIf
    MatrixExample_forAllRowsIf
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

add_custom_target(MatricesTraverseDocumentation ALL DEPENDS ${DOC_OUTPUTS})
