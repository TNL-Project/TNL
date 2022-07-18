set( GRID_DIMS 1 2 3  )

foreach( DIM IN ITEMS ${GRID_DIMS} )

   if (BUILD_CUDA)
      CUDA_ADD_EXECUTABLE( tnl-benchmark-grid-${DIM}D tnl-benchmark-grid.cu OPTIONS ${CUDA_TESTS_FLAGS} -DGRID_DIM=${DIM} )
   ELSE()
      add_executable( tnl-benchmark-grid-${DIM}D tnl-benchmark-grid.cpp )
      target_compile_definitions( tnl-benchmark-grid-${DIM}D PUBLIC -DGRID_DIM=${DIM} )
   ENDIF()

   install( TARGETS tnl-benchmark-grid-${DIM}D RUNTIME DESTINATION bin )
endforeach()

