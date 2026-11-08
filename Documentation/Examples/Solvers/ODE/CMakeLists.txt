set(HOST_EXAMPLES StaticODESolver-SineExample StaticODESolver-SineExample_iterate StaticODESolver-LorenzExample)

set(COMMON_EXAMPLES
    StaticODESolver-SineParallelExample
    StaticODESolver-LorenzParallelExample
    ODESolver-HeatEquationExample
    ODESolver-HeatEquationWithMonitorExample
)

set(MPI_EXAMPLES DistributedODESolver-HeatEquationExample)

foreach(target IN ITEMS ${HOST_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.h BACKENDS HOST)
endforeach()

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.h ARGS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH})
endforeach()

foreach(target IN ITEMS ${MPI_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.h MPI ARGS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH})
endforeach()

# check if the system has python3 and matplotlib
find_package(Python3 COMPONENTS Interpreter QUIET)
set(TNL_PYTHON3_MATPLOTLIB_FOUND FALSE)

if(Python3_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import matplotlib"
        RESULT_VARIABLE TNL_MATPLOTLIB_CHECK_EXIT_CODE
        OUTPUT_QUIET
    )

    if(TNL_MATPLOTLIB_CHECK_EXIT_CODE EQUAL 0)
        set(TNL_PYTHON3_MATPLOTLIB_FOUND TRUE)
    else()
        message(
            WARNING
            "The Python3 interpreter does not have the matplotlib package, some examples will not be post-processed."
        )
    endif()
else()
    message(WARNING "The Python3 interpreter was not found, some examples will not be post-processed.")
endif()

if(TNL_PYTHON3_MATPLOTLIB_FOUND)
    add_custom_command(
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/StaticODESolver-SineExample.py
            ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineExample.out
            ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineExample.png
        OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineExample.png
        DEPENDS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineExample.out
        VERBATIM
    )

    add_custom_command(
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/StaticODESolver-SineParallelExample.py
            ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineParallelExample-result.out
            ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineParallelExample.png
        OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineParallelExample.png
        DEPENDS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineParallelExample.out
        VERBATIM
    )

    add_custom_command(
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/StaticODESolver-LorenzExample.py
            ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzExample.out
            ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzExample.png
        OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzExample.png
        DEPENDS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzExample.out
        VERBATIM
    )

    add_custom_command(
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/StaticODESolver-LorenzParallelExample.py
            ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzParallelExample-result.out
            # NOTE: the script appends "-{sigma_idx}.png"
            ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzParallelExample
        OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzParallelExample-1.png
        DEPENDS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzParallelExample.out
        VERBATIM
    )

    add_custom_command(
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ODESolver-HeatEquationExample.py
            ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ODESolver-HeatEquationExample-result.out
            ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ODESolver-HeatEquationExample.png
        OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ODESolver-HeatEquationExample.png
        DEPENDS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ODESolver-HeatEquationExample.out
        VERBATIM
    )

    # the target-level dependency is needed because the custom commands are
    # defined in a different directory than the run-doc-examples target
    add_custom_target(
        ODESolversExamples-plots
        DEPENDS
            ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineExample.png
            ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineParallelExample.png
            ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzExample.png
            ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzParallelExample-1.png
            ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ODESolver-HeatEquationExample.png
    )
    add_dependencies(run-doc-examples ODESolversExamples-plots)
endif()
