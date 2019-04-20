add_subdirectory(Ginkgo)
add_subdirectory(Hypre)
add_subdirectory( simple-examples )
add_subdirectory( heat-equation )
add_subdirectory( transport-equation )
#add_subdirectory( navier-stokes )
#add_subdirectory( inviscid-flow )
#add_subdirectory( inviscid-flow-sw )
#add_subdirectory( inviscid-flow-vl )
#add_subdirectory( mean-curvature-flow )
add_subdirectory( flows )
add_subdirectory( turbulent-flows )

set(CPP_TARGETS tnl-optimize-ranks tnl-turbulence-generator)
set(CUDA_TARGETS tnl-optimize-ranks-cuda tnl-turbulence-generator-cuda)


# skip building host-only targets in CUDA-enabled CI jobs
if(TNL_BUILD_CPP_TARGETS)
    foreach(target IN ITEMS ${CPP_TARGETS})
        add_executable(${target} ${target}.cpp)
        target_link_libraries(${target} PUBLIC TNL::TNL)
    endforeach()
    install(TARGETS ${CPP_TARGETS} RUNTIME DESTINATION bin COMPONENT examples)
endif()

if(TNL_BUILD_CUDA)
    foreach(target IN ITEMS ${CUDA_TARGETS})
        add_executable(${target} ${target}.cu)
        target_link_libraries(${target} PUBLIC TNL::TNL)
    endforeach()
    install(TARGETS ${CUDA_TARGETS} RUNTIME DESTINATION bin COMPONENT examples)
endif()

<<<<<<< HEAD
if(TNL_BUILD_MPI)
    # enable MPI support in TNL and add MPI to the targets:
    # https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
    if(TNL_BUILD_CPP_TARGETS)
        target_compile_definitions(tnl-optimize-ranks PUBLIC "-DHAVE_MPI")
        target_link_libraries(tnl-optimize-ranks PUBLIC MPI::MPI_CXX)
    endif()
    if(TNL_BUILD_CUDA)
        target_compile_definitions(tnl-optimize-ranks-cuda PUBLIC "-DHAVE_MPI")
        target_link_libraries(tnl-optimize-ranks-cuda PUBLIC MPI::MPI_CXX)
    endif()
endif()
=======
ADD_EXECUTABLE( ObjectExample_getType ObjectExample_getType.cpp )
ADD_CUSTOM_COMMAND( COMMAND ObjectExample_getType > ObjectExample_getType.out OUTPUT ObjectExample_getType.out )

ADD_EXECUTABLE( ParameterContainerExample ParameterContainerExample.cpp )

ADD_EXECUTABLE( ParseObjectTypeExample ParseObjectTypeExample.cpp )
ADD_CUSTOM_COMMAND( COMMAND ParseObjectTypeExample > ParseObjectTypeExample.out OUTPUT ParseObjectTypeExample.out )

ADD_EXECUTABLE( StringExample StringExample.cpp )
ADD_CUSTOM_COMMAND( COMMAND StringExample > StringExample.out OUTPUT StringExample.out )

ADD_EXECUTABLE( StringExampleGetAllocatedSize StringExampleGetAllocatedSize.cpp )
ADD_CUSTOM_COMMAND( COMMAND StringExampleGetAllocatedSize > StringExampleGetAllocatedSize.out OUTPUT StringExampleGetAllocatedSize.out )

ADD_EXECUTABLE( StringExampleReplace StringExampleReplace.cpp )
ADD_CUSTOM_COMMAND( COMMAND StringExampleReplace > StringExampleReplace.out OUTPUT StringExampleReplace.out )

ADD_EXECUTABLE( StringExampleSetSize StringExampleSetSize.cpp )
ADD_CUSTOM_COMMAND( COMMAND StringExampleSetSize > StringExampleSetSize.out OUTPUT StringExampleSetSize.out )

ADD_EXECUTABLE( StringExampleSplit StringExampleSplit.cpp )
ADD_CUSTOM_COMMAND( COMMAND StringExampleSplit > StringExampleSplit.out OUTPUT StringExampleSplit.out )

ADD_EXECUTABLE( StringExampleStrip StringExampleStrip.cpp )
ADD_CUSTOM_COMMAND( COMMAND StringExampleStrip > StringExampleStrip.out OUTPUT StringExampleStrip.out )

ADD_EXECUTABLE( TimerExample TimerExample.cpp )
ADD_CUSTOM_COMMAND( COMMAND TimerExample > TimerExample.out OUTPUT TimerExample.out )

ADD_EXECUTABLE( TimerExampleLogger TimerExampleLogger.cpp )
ADD_CUSTOM_COMMAND( COMMAND TimerExampleLogger > TimerExampleLogger.out OUTPUT TimerExampleLogger.out )

ADD_EXECUTABLE( VectorExample VectorExample.cpp )

ADD_CUSTOM_TARGET( RunExamples ALL DEPENDS
   FileExample.out
   FileExampleSaveAndLoad.out
   FileNameExample.out
   FileNameExampleDistributedSystemNodeCoordinates.out
   FileNameExampleDistributedSystemNodeId.out
   ObjectExample_getType.out
   ParseObjectTypeExample.out
   StringExample.out
   StringExampleGetAllocatedSize.out
   StringExampleReplace.out
   StringExampleSplit.out
   StringExampleStrip.out
   TimerExample.out
   TimerExampleLogger.out )

if( BUILD_CUDA )
   ADD_CUSTOM_TARGET( RunExamples-cuda ALL DEPENDS
      FileExampleCuda.out )
ENDIF()
>>>>>>> 9759c6aba (added Prandtl, PrandtlKolmogorov, KOmega and KEpsilon Solvers. Prandtl not working yet)
