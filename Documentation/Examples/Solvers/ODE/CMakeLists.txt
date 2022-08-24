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
   set( HOST_OUTPUTS ${HOST_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
endforeach()

if( TNL_BUILD_CUDA )
   foreach( target IN ITEMS ${COMMON_EXAMPLES} )
      add_executable( ${target} ${target}.cu )
      target_link_libraries( ${target} PUBLIC TNL::TNL_CUDA )
      add_custom_command( COMMAND ${target} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                          OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                          DEPENDS ${target} )
      set( CUDA_OUTPUTS ${CUDA_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
   endforeach()
   add_custom_target( RunODESolversExamples ALL DEPENDS ${CUDA_OUTPUTS} )
else()
   foreach( target IN ITEMS ${COMMON_EXAMPLES} )
      add_executable( ${target} ${target}.cpp )
      target_link_libraries( ${target} PUBLIC TNL::TNL_CXX )
      add_custom_command( COMMAND ${target} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                          OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                          DEPENDS ${target} )
      set( HOST_OUTPUTS ${HOST_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
   endforeach()
   add_custom_target( RunODESolversExamples ALL DEPENDS ${HOST_OUTPUTS} )
endif()

ADD_CUSTOM_COMMAND(
   COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/StaticODESolver-SineExample.py
      ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineExample.out
      ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineExample.png
   OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineExample.png
   DEPENDS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineExample.out
   VERBATIM )

ADD_CUSTOM_COMMAND(
   COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/StaticODESolver-SineParallelExample.py
      ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineParallelExample-result.out
      ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineParallelExample.png
   OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineParallelExample.png
   DEPENDS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineParallelExample.out
   VERBATIM )

ADD_CUSTOM_COMMAND(
   COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/StaticODESolver-LorenzExample.py
      ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzExample.out
      ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzExample.png
   OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzExample.png
   DEPENDS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzExample.out
   VERBATIM )

ADD_CUSTOM_COMMAND(
   COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/StaticODESolver-LorenzParallelExample.py
      ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzParallelExample-result.out
      # NOTE: the script appends "-{sigma_idx}.png"
      ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzParallelExample
   OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzParallelExample-1.png
   DEPENDS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzParallelExample.out
   VERBATIM )

ADD_CUSTOM_COMMAND(
   COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/ODESolver-HeatEquationExample.py
      ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ODESolver-HeatEquationExample-result.out
      ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ODESolver-HeatEquationExample.png
   OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ODESolver-HeatEquationExample.png
   DEPENDS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ODESolver-HeatEquationExample.out
   VERBATIM )


ADD_CUSTOM_TARGET( ProcessODESolversExamplesResults ALL DEPENDS
   ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineExample.png
   ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineParallelExample.png
   ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzExample.png
   ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzParallelExample-1.png
   ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ODESolver-HeatEquationExample.png
)
