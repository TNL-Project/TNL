set( CUDA_EXAMPLES
   ArrayAllocation
   ArrayIO
   ArrayView-1
   ArrayView-2
   ArrayViewForElements
   contains
   ElementsAccessing-1
   ElementsAccessing-2
)
set( HOST_EXAMPLES
   StaticArrayExample
)

if( TNL_BUILD_CUDA )
   foreach( target IN ITEMS ${CUDA_EXAMPLES} )
      add_executable( ${target} ${target}.cu )
      target_link_libraries( ${target} PUBLIC TNL::TNL_CUDA )
      add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                          OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                          DEPENDS ${target} )
      set( DOC_OUTPUTS ${DOC_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
   endforeach()
endif()

foreach( target IN ITEMS ${HOST_EXAMPLES} )
   add_executable( ${target} ${target}.cpp )
   target_link_libraries( ${target} PUBLIC TNL::TNL_CXX )
   add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                       OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                       DEPENDS ${target} )
   set( DOC_OUTPUTS ${DOC_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
endforeach()

add_custom_target( RunUGArraysExamples ALL DEPENDS ${DOC_OUTPUTS} )

# add the dependency to the main target
add_dependencies( run-doc-examples RunUGArraysExamples )
