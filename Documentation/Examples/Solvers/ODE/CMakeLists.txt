set( HOST_EXAMPLES
   StaticODESolver-SineExample
   StaticODESolver-LorenzExample
)

set( COMMON_EXAMPLES
   StaticODESolver-SineParallelExample
   StaticODESolver-LorenzParallelExample
   ODESolver-HeatEquationExample
   ODESolver-HeatEquationWithMonitorExample
)

foreach( target IN ITEMS ${HOST_EXAMPLES})
   add_executable( ${target} ${target}.cpp )
   target_link_libraries( ${target} PUBLIC TNL::TNL_CXX )
   add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                       OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                       DEPENDS ${target} )
   set( DOC_OUTPUTS ${DOC_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
endforeach()

if( TNL_BUILD_CUDA )
   foreach( target IN ITEMS ${COMMON_EXAMPLES} )
      add_executable( ${target} ${target}.cu )
      target_link_libraries( ${target} PUBLIC TNL::TNL_CUDA )
      add_custom_command( COMMAND ${target} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                          OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                          DEPENDS ${target} )
      set( DOC_OUTPUTS ${DOC_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
   endforeach()
else()
   foreach( target IN ITEMS ${COMMON_EXAMPLES} )
      add_executable( ${target} ${target}.cpp )
      target_link_libraries( ${target} PUBLIC TNL::TNL_CXX )
      add_custom_command( COMMAND ${target} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                          OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                          DEPENDS ${target} )
      set( DOC_OUTPUTS ${DOC_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
   endforeach()
endif()

# check if the system has python3 and matplotlib
find_package( Python3 COMPONENTS Interpreter QUIET )
set( TNL_PYTHON3_MATPLOTLIB_FOUND FALSE )
if( Python3_FOUND )
   execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import matplotlib"
      RESULT_VARIABLE TNL_MATPLOTLIB_CHECK_EXIT_CODE
      OUTPUT_QUIET
   )
   if( TNL_MATPLOTLIB_CHECK_EXIT_CODE EQUAL 0 )
      set( TNL_PYTHON3_MATPLOTLIB_FOUND TRUE )
   else()
      message( WARNING "The Python3 interpreter does not have the matplotlib package, some examples will not be post-processed.")
   endif()
else()
   message( WARNING "The Python3 interpreter was not found, some examples will not be post-processed.")
endif()

if( TNL_PYTHON3_MATPLOTLIB_FOUND )
   ADD_CUSTOM_COMMAND(
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/StaticODESolver-SineExample.py
         ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineExample.out
         ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineExample.png
      OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineExample.png
      DEPENDS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineExample.out
      VERBATIM )

   ADD_CUSTOM_COMMAND(
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/StaticODESolver-SineParallelExample.py
         ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineParallelExample-result.out
         ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineParallelExample.png
      OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineParallelExample.png
      DEPENDS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineParallelExample.out
      VERBATIM )

   ADD_CUSTOM_COMMAND(
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/StaticODESolver-LorenzExample.py
         ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzExample.out
         ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzExample.png
      OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzExample.png
      DEPENDS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzExample.out
      VERBATIM )

   ADD_CUSTOM_COMMAND(
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/StaticODESolver-LorenzParallelExample.py
         ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzParallelExample-result.out
         # NOTE: the script appends "-{sigma_idx}.png"
         ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzParallelExample
      OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzParallelExample-1.png
      DEPENDS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzParallelExample.out
      VERBATIM )

   ADD_CUSTOM_COMMAND(
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ODESolver-HeatEquationExample.py
         ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ODESolver-HeatEquationExample-result.out
         ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ODESolver-HeatEquationExample.png
      OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ODESolver-HeatEquationExample.png
      DEPENDS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ODESolver-HeatEquationExample.out
      VERBATIM )



   set( DOC_OUTPUTS ${DOC_OUTPUTS}
      ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineExample.png
      ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineParallelExample.png
      ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzExample.png
      ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzParallelExample-1.png
      ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ODESolver-HeatEquationExample.png
   )

   add_custom_target( RunODESolversExamples ALL DEPENDS ${DOC_OUTPUTS} )

   # add the dependency to the main target
   add_dependencies( run-doc-examples RunODESolversExamples )
endif()
