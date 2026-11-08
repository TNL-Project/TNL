set(COMMON_EXAMPLES
    IterativeLinearSolverExample
    IterativeLinearSolverWithMonitorExample
    IterativeLinearSolverWithTimerExample
    IterativeLinearSolverWithPreconditionerExample
    IterativeLinearSolverWithRuntimeTypesExample
)

if(NOT TARGET SuiteSparse::UMFPACK)
    find_package(UMFPACK GLOBAL)
endif()
if(TARGET SuiteSparse::UMFPACK)
    set(UMFPACK_DEPENDING_EXAMPLES DirectLinearSolverWithMonitorExample DirectLinearSolverWithTimerExample)
endif()

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp)
endforeach()

foreach(target IN ITEMS ${UMFPACK_DEPENDING_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp BACKENDS HOST)
    target_compile_definitions(${target} PUBLIC "-DHAVE_UMFPACK")
    target_link_libraries(${target} PUBLIC TNL::TNL SuiteSparse::UMFPACK)
    # reset SuiteSparse::UMFPACK to system target (upstream sets 0)
    set_property(TARGET SuiteSparse::UMFPACK PROPERTY SYSTEM 1)
endforeach()
