set( COMMON_EXAMPLES
         ArrayExample
         ArrayExample_forElements
         ArrayViewExample
         ArrayViewExample_forElements
         VectorExample
)

set( MPI_COMMON_EXAMPLES
         DistributedArrayExample
         DistributedVectorExample
         DistributedNDArrayExample
)

SET( mpirun_parameters -np 4 -H localhost:4 )

if( TNL_BUILD_CUDA )
   foreach( target IN ITEMS ${COMMON_EXAMPLES} ${MPI_COMMON_EXAMPLES} )
      add_executable( ${target} ${target}.cu )
      target_link_libraries( ${target} PUBLIC TNL::TNL_CUDA )
   endforeach()
else()
   foreach( target IN ITEMS ${COMMON_EXAMPLES} ${MPI_COMMON_EXAMPLES} )
      add_executable( ${target} ${target}.cpp )
      target_link_libraries( ${target} PUBLIC TNL::TNL_CXX )
   endforeach()
endif()

foreach( target IN ITEMS ${COMMON_EXAMPLES} )
   add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                        OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                        DEPENDS ${target} )
   set( DOC_OUTPUTS ${DOC_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
endforeach()

if( TNL_BUILD_MPI )
   foreach( target IN ITEMS ${MPI_COMMON_EXAMPLES} )
      # enable MPI support in TNL
      target_compile_definitions( ${target} PUBLIC "-DHAVE_MPI" )
      # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
      target_link_libraries( ${target} PUBLIC MPI::MPI_CXX )

      add_custom_command( COMMAND "mpirun" ${mpirun_parameters} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}" > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                           OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                           DEPENDS ${target} )
      set( DOC_OUTPUTS ${DOC_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
   endforeach()
endif()

add_custom_target( RunContainersExamples ALL DEPENDS ${DOC_OUTPUTS} )

# add the dependency to the main target
add_dependencies( run-doc-examples RunContainersExamples )
