#k-omega-Lax-Friedrichs

set( tnl_flow_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_flow_SOURCES     
     twoEquationTurbulenceModelLaxFriedrichs.cpp
     twoEquationTurbulenceModelLaxFriedrichs.cu )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-k-omega-Lax-Friedrichs twoEquationTurbulenceModelLaxFriedrichs.cu)
   target_link_libraries (tnl-k-omega-Lax-Friedrichs ${CUSPARSE_LIBRARY} )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE(tnl-k-omega-Lax-Friedrichs twoEquationTurbulenceModelLaxFriedrichs.cpp)     
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-k-omega-Lax-Friedrichs
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES run-two-equation-turbulence-model-Lax-Friedrichs
               ${tnl_flow_SOURCES}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/flow )

#k-omega-Steger-Warming

set( tnl_flow_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_flow_SOURCES     
     twoEquationTurbulenceModelStegerWarming.cpp
     twoEquationTurbulenceModelStegerWarming.cu )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-k-omega-Steger-Warming twoEquationTurbulenceModelStegerWarming.cu)
   target_link_libraries (tnl-k-omega-Steger-Warming ${CUSPARSE_LIBRARY} )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE(tnl-k-omega-Steger-Warming twoEquationTurbulenceModelStegerWarming.cpp)     
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-k-omega-Steger-Warming
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES run-two-equation-turbulence-model-Steger-Warming
               ${tnl_flow_SOURCES}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/flow )

#k-omega-Van-Leer

set( tnl_flow_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_flow_SOURCES     
     twoEquationTurbulenceModelVanLeer.cpp
     twoEquationTurbulenceModelVanLeer.cu )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-k-omega-Van-Leer twoEquationTurbulenceModelVanLeer.cu)
   target_link_libraries (tnl-k-omega-Van-Leer ${CUSPARSE_LIBRARY} )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE(tnl-k-omega-Van-Leer twoEquationTurbulenceModelVanLeer.cpp)     
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-k-omega-Van-Leer
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES run-two-equation-turbulence-model-Van-Leer
               ${tnl_flow_SOURCES}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/flow )

#k-omega-AUSM-plus

set( tnl_flow_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_flow_SOURCES     
     twoEquationTurbulenceModelAUSMPlus.cpp
     twoEquationTurbulenceModelAUSMPlus.cu )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-k-omega-AUSM-plus twoEquationTurbulenceModelAUSMPlus.cu)
   target_link_libraries (tnl-k-omega-AUSM-plus ${CUSPARSE_LIBRARY} )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE(tnl-k-omega-AUSM-plus twoEquationTurbulenceModelAUSMPlus.cpp)     
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-k-omega-AUSM-plus
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES run-two-equation-turbulence-model-AUSM-plus
               ${tnl_flow_SOURCES}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/flow )


