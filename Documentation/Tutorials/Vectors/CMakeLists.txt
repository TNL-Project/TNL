set( CUDA_EXAMPLES
     Expressions
     Reduction
)
set( HOST_EXAMPLES
     StaticVectorExample
)

if( TNL_BUILD_CUDA )
   foreach( target IN ITEMS ${CUDA_EXAMPLES} )
      cuda_add_executable( ${target} ${target}.cu )
      add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                          OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                          DEPENDS ${target} )
      set( CUDA_OUTPUTS ${CUDA_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
   endforeach()

   add_custom_target( RunTutorialsVectorsExamples-cuda ALL DEPENDS ${CUDA_OUTPUTS} )
endif()

foreach( target IN ITEMS ${HOST_EXAMPLES} )
   add_executable( ${target} ${target}.cpp )
   add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                       OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                       DEPENDS ${target} )
   set( HOST_OUTPUTS ${HOST_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
endforeach()

add_custom_target( RunTutorialsVectorsExamples ALL DEPENDS ${HOST_OUTPUTS} )
