#zero-equation-turbulence-model-Lax-Friedrichs

set( tnl_flow_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_flow_SOURCES     
     zeroEquationTurbulenceModelLaxFriedrichs.cpp
     zeroEquationTurbulenceModelLaxFriedrichs.cu )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-zero-equation-turbulence-model-Lax-Friedrichs zeroEquationTurbulenceModelLaxFriedrichs.cu)
   target_link_libraries (tnl-zero-equation-turbulence-model-Lax-Friedrichs ${CUSPARSE_LIBRARY} )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE(tnl-zero-equation-turbulence-model-Lax-Friedrichs zeroEquationTurbulenceModelLaxFriedrichs.cpp)     
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-zero-equation-turbulence-model-Lax-Friedrichs
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES run-zero-equation-turbulence-model-Lax-Friedrichs
               ${tnl_flow_SOURCES}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/flow )

#zero-equation-turbulence-model-Steger-Warming

set( tnl_flow_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_flow_SOURCES     
     zeroEquationTurbulenceModelStegerWarming.cpp
     zeroEquationTurbulenceModelStegerWarming.cu )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-zero-equation-turbulence-model-Steger-Warming zeroEquationTurbulenceModelStegerWarming.cu)
   target_link_libraries (tnl-zero-equation-turbulence-model-Steger-Warming ${CUSPARSE_LIBRARY} )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE(tnl-zero-equation-turbulence-model-Steger-Warming zeroEquationTurbulenceModelStegerWarming.cpp)     
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-zero-equation-turbulence-model-Steger-Warming
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES run-zero-equation-turbulence-model-Steger-Warming
               ${tnl_flow_SOURCES}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/flow )

#zero-equation-turbulence-model-Van-Leer

set( tnl_flow_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_flow_SOURCES     
     zeroEquationTurbulenceModelVanLeer.cpp
     zeroEquationTurbulenceModelVanLeer.cu )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-zero-equation-turbulence-model-Van-Leer zeroEquationTurbulenceModelVanLeer.cu)
   target_link_libraries (tnl-zero-equation-turbulence-model-Van-Leer ${CUSPARSE_LIBRARY} )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE(tnl-zero-equation-turbulence-model-Van-Leer zeroEquationTurbulenceModelVanLeer.cpp)     
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-zero-equation-turbulence-model-Van-Leer
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES run-zero-equation-turbulence-model-Van-Leer
               ${tnl_flow_SOURCES}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/flow )

#zero-equation-turbulence-model-AUSM-plus

set( tnl_flow_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_flow_SOURCES     
     zeroEquationTurbulenceModelAUSMPlus.cpp
     zeroEquationTurbulenceModelAUSMPlus.cu )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-zero-equation-turbulence-model-AUSM-plus zeroEquationTurbulenceModelAUSMPlus.cu)
   target_link_libraries (tnl-zero-equation-turbulence-model-AUSM-plus ${CUSPARSE_LIBRARY} )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE(tnl-zero-equation-turbulence-model-AUSM-plus zeroEquationTurbulenceModelAUSMPlus.cpp)     
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-zero-equation-turbulence-model-AUSM-plus
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES run-zero-equation-turbulence-model-AUSM-plus
               ${tnl_flow_SOURCES}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/flow )


