set(COMMON_EXAMPLES IterativeLinearSolverExample IterativeLinearSolverWithMonitorExample IterativeLinearSolverWithTimerExample
                    IterativeLinearSolverWithPreconditionerExample IterativeLinearSolverWithRuntimeTypesExample
)

find_package(UMFPACK)
if(UMFPACK_FOUND)
    set(UMFPACK_DEPENDING_EXAMPLES DirectLinearSolverWithMonitorExample DirectLinearSolverWithTimerExample)
endif()

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

foreach(target IN ITEMS ${UMFPACK_DEPENDING_EXAMPLES})
    add_executable(${target} ${target}.cpp)
    target_compile_definitions(${target} PUBLIC "-DHAVE_UMFPACK")
    target_link_libraries(${target} PUBLIC TNL::TNL SuiteSparse::UMFPACK)
    add_custom_command(
        COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
        OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out DEPENDS ${target}
    )
    set(DOC_OUTPUTS ${DOC_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out)
endforeach()

add_custom_target(RunLinearSolversExamples ALL DEPENDS ${DOC_OUTPUTS})

# add the dependency to the main target
add_dependencies(run-doc-examples RunLinearSolversExamples)
