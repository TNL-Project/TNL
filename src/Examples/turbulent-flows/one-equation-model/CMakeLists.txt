#one-equation-turbulence-model-Lax-Friedrichs

set( tnl_flow_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_flow_SOURCES     
     oneEquationTurbulenceModelLaxFriedrichs.cpp
     oneEquationTurbulenceModelLaxFriedrichs.cu )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-one-equation-turbulence-model-Lax-Friedrichs oneEquationTurbulenceModelLaxFriedrichs.cu)
   target_link_libraries (tnl-one-equation-turbulence-model-Lax-Friedrichs ${CUSPARSE_LIBRARY} )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE(tnl-one-equation-turbulence-model-Lax-Friedrichs oneEquationTurbulenceModelLaxFriedrichs.cpp)     
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-one-equation-turbulence-model-Lax-Friedrichs
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES run-one-equation-turbulence-model-Lax-Friedrichs
               ${tnl_flow_SOURCES}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/flow )

#one-equation-turbulence-model-Steger-Warming

set( tnl_flow_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_flow_SOURCES     
     oneEquationTurbulenceModelStegerWarming.cpp
     oneEquationTurbulenceModelStegerWarming.cu )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-one-equation-turbulence-model-Steger-Warming oneEquationTurbulenceModelStegerWarming.cu)
   target_link_libraries (tnl-one-equation-turbulence-model-Steger-Warming ${CUSPARSE_LIBRARY} )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE(tnl-one-equation-turbulence-model-Steger-Warming oneEquationTurbulenceModelStegerWarming.cpp)     
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-one-equation-turbulence-model-Steger-Warming
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES run-one-equation-turbulence-model-Steger-Warming
               ${tnl_flow_SOURCES}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/flow )

#one-equation-turbulence-model-Van-Leer

set( tnl_flow_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_flow_SOURCES     
     oneEquationTurbulenceModelVanLeer.cpp
     oneEquationTurbulenceModelVanLeer.cu )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-one-equation-turbulence-model-Van-Leer oneEquationTurbulenceModelVanLeer.cu)
   target_link_libraries (tnl-one-equation-turbulence-model-Van-Leer ${CUSPARSE_LIBRARY} )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE(tnl-one-equation-turbulence-model-Van-Leer oneEquationTurbulenceModelVanLeer.cpp)     
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-one-equation-turbulence-model-Van-Leer
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES run-one-equation-turbulence-model-Van-Leer
               ${tnl_flow_SOURCES}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/flow )

#one-equation-turbulence-model-AUSM-plus

set( tnl_flow_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_flow_SOURCES     
     oneEquationTurbulenceModelAUSMPlus.cpp
     oneEquationTurbulenceModelAUSMPlus.cu )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-one-equation-turbulence-model-AUSM-plus oneEquationTurbulenceModelAUSMPlus.cu)
   target_link_libraries (tnl-one-equation-turbulence-model-AUSM-plus ${CUSPARSE_LIBRARY} )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE(tnl-one-equation-turbulence-model-AUSM-plus oneEquationTurbulenceModelAUSMPlus.cpp)     
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-one-equation-turbulence-model-AUSM-plus
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES run-one-equation-turbulence-model-AUSM-plus
               ${tnl_flow_SOURCES}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/flow )


