ADD_SUBDIRECTORY( Containers )

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


ADD_EXECUTABLE( ConfigDescriptionExample ConfigDescriptionExample.cpp )

ADD_EXECUTABLE( FileExample FileExample.cpp )
ADD_CUSTOM_COMMAND( COMMAND FileExample > FileExample.out OUTPUT FileExample.out )

IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(FileExampleCuda FileExampleCuda.cu)
   ADD_CUSTOM_COMMAND( COMMAND FileExampleCuda > FileExampleCuda.out OUTPUT FileExampleCuda.out )
ENDIF()

ADD_EXECUTABLE( FileExampleSaveAndLoad FileExampleSaveAndLoad.cpp )
ADD_CUSTOM_COMMAND( COMMAND FileExampleSaveAndLoad > FileExampleSaveAndLoad.out OUTPUT FileExampleSaveAndLoad.out )

ADD_EXECUTABLE( FileNameExample FileNameExample.cpp )
ADD_CUSTOM_COMMAND( COMMAND FileNameExample > FileNameExample.out OUTPUT FileNameExample.out )

ADD_EXECUTABLE( FileNameExampleDistributedSystemNodeCoordinates FileNameExampleDistributedSystemNodeCoordinates.cpp )
ADD_CUSTOM_COMMAND( COMMAND FileNameExampleDistributedSystemNodeCoordinates > FileNameExampleDistributedSystemNodeCoordinates.out OUTPUT FileNameExampleDistributedSystemNodeCoordinates.out )


ADD_EXECUTABLE( FileNameExampleDistributedSystemNodeId FileNameExampleDistributedSystemNodeId.cpp )
ADD_CUSTOM_COMMAND( COMMAND FileNameExampleDistributedSystemNodeId > FileNameExampleDistributedSystemNodeId.out OUTPUT FileNameExampleDistributedSystemNodeId.out )

ADD_EXECUTABLE( ListExample ListExample.cpp )
ADD_EXECUTABLE( LoggerExample LoggerExample.cpp )
ADD_EXECUTABLE( MathExample MathExample.cpp )

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
