#navier-stokes-Lax-Friedrichs

set( tnl_flow_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_flow_SOURCES     
     navierStokesLaxFriedrichs.cpp
     navierStokesLaxFriedrichs.cu )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-navier-stokes-Lax-Friedrichs navierStokesLaxFriedrichs.cu)
   target_link_libraries (tnl-navier-stokes-Lax-Friedrichs ${CUSPARSE_LIBRARY} )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE(tnl-navier-stokes-Lax-Friedrichs navierStokesLaxFriedrichs.cpp)     
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-navier-stokes-Lax-Friedrichs
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES run-navier-stokes-Lax-Friedrichs
               ${tnl_flow_SOURCES}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/flow )

#navier-stokes-Steger-Warming

set( tnl_flow_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_flow_SOURCES     
     navierStokesStegerWarming.cpp
     navierStokesStegerWarming.cu )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-navier-stokes-Steger-Warming navierStokesStegerWarming.cu)
   target_link_libraries (tnl-navier-stokes-Steger-Warming ${CUSPARSE_LIBRARY} )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE(tnl-navier-stokes-Steger-Warming navierStokesStegerWarming.cpp)     
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-navier-stokes-Steger-Warming
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES run-navier-stokes-Steger-Warming
               ${tnl_flow_SOURCES}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/flow )

#navier-stokes-Van-Leer

set( tnl_flow_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_flow_SOURCES     
     navierStokesVanLeer.cpp
     navierStokesVanLeer.cu )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-navier-stokes-Van-Leer navierStokesVanLeer.cu)
   target_link_libraries (tnl-navier-stokes-Van-Leer ${CUSPARSE_LIBRARY} )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE(tnl-navier-stokes-Van-Leer navierStokesVanLeer.cpp)     
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-navier-stokes-Van-Leer
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES run-navier-stokes-Van-Leer
               ${tnl_flow_SOURCES}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/flow )

#navier-stokes-AUSM-plus

set( tnl_flow_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_flow_SOURCES     
     navierStokesAUSMPlus.cpp
     navierStokesAUSMPlus.cu )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-navier-stokes-AUSM-plus navierStokesAUSMPlus.cu)
   target_link_libraries (tnl-navier-stokes-AUSM-plus ${CUSPARSE_LIBRARY} )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE(tnl-navier-stokes-AUSM-plus navierStokesAUSMPlus.cpp)     
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-navier-stokes-AUSM-plus
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES run-navier-stokes-AUSM-plus
               ${tnl_flow_SOURCES}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/flow )


