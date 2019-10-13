#include <TNL/Functions/FunctionAdapter.h>

#include "DensityBoundaryConditionBoilerModel.h"
#include "MomentumXBoundaryConditionBoilerModel.h"
#include "MomentumYBoundaryConditionBoilerModel.h"
#include "MomentumZBoundaryConditionBoilerModel.h"
#include "AUSMPlusEnergyBoundaryConditionBoilerModel.h"
#include "TurbulentEnergyBoundaryConditionBoilerModel.h"
#include "DisipationBoundaryConditionBoilerModel.h"

namespace TNL {

template< typename Mesh,
          typename Function,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class BoundaryConditionsBoilerModel
{
   public:
      typedef Mesh MeshType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef Function FunctionType;
      typedef Functions::MeshFunction< Mesh > MeshFunctionType;
      typedef typename Mesh::DeviceType DeviceType;

      typedef TNL::Operators::DensityBoundaryConditionsBoilerModel< MeshType, FunctionType, RealType, IndexType > DensityBoundaryConditionsType;
      typedef TNL::Operators::MomentumXBoundaryConditionsBoilerModel< MeshType, FunctionType, RealType, IndexType > MomentumXBoundaryConditionsType;
      typedef TNL::Operators::MomentumYBoundaryConditionsBoilerModel< MeshType, FunctionType, RealType, IndexType > MomentumYBoundaryConditionsType;
      typedef TNL::Operators::MomentumZBoundaryConditionsBoilerModel< MeshType, FunctionType, RealType, IndexType > MomentumZBoundaryConditionsType;
      typedef TNL::Operators::AUSMPlusEnergyBoundaryConditionsBoilerModel< MeshType, FunctionType, RealType, IndexType > EnergyBoundaryConditionsType;
      typedef TNL::Operators::TurbulentEnergyBoundaryConditionsBoilerModel< MeshType, FunctionType, RealType, IndexType > TurbulentEnergyBoundaryConditionsType;
      typedef TNL::Operators::DisipationBoundaryConditionsBoilerModel< MeshType, FunctionType, RealType, IndexType > DisipationBoundaryConditionsType;
      typedef CompressibleConservativeVariables< MeshType > CompressibleConservativeVariablesType;

      typedef Pointers::SharedPointer< DensityBoundaryConditionsType > DensityBoundaryConditionsTypePointer;
      typedef Pointers::SharedPointer< MomentumXBoundaryConditionsType > MomentumXBoundaryConditionsTypePointer;
      typedef Pointers::SharedPointer< MomentumYBoundaryConditionsType > MomentumYBoundaryConditionsTypePointer;
      typedef Pointers::SharedPointer< MomentumZBoundaryConditionsType > MomentumZBoundaryConditionsTypePointer;
      typedef Pointers::SharedPointer< EnergyBoundaryConditionsType > EnergyBoundaryConditionsTypePointer;
      typedef Pointers::SharedPointer< TurbulentEnergyBoundaryConditionsType > TurbulentEnergyBoundaryConditionsTypePointer;
      typedef Pointers::SharedPointer< DisipationBoundaryConditionsType > DisipationBoundaryConditionsTypePointer;
      typedef Pointers::SharedPointer< CompressibleConservativeVariablesType > CompressibleConservativeVariablesPointer;
      typedef Pointers::SharedPointer< MeshType > MeshPointer;
      typedef Pointers::SharedPointer< MeshFunctionType, DeviceType > MeshFunctionPointer;

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         config.addEntry< double >( prefix + "vertical-angle", "Vertical angle of throttle in degrees", 0 );
         config.addEntry< double >( prefix + "horizontal-angle", "Horizontal angle of throttle in degrees", 45 );
      }

      bool setup( const MeshPointer& meshPointer,
                  const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         this->densityBoundaryConditionsPointer->setup( meshPointer, parameters, prefix);
         this->momentumXBoundaryConditionsPointer->setup( meshPointer, parameters, prefix);
         this->momentumYBoundaryConditionsPointer->setup( meshPointer, parameters, prefix);
         this->momentumZBoundaryConditionsPointer->setup( meshPointer, parameters, prefix);
         this->energyBoundaryConditionsPointer->setup( meshPointer, parameters, prefix);
         setZAngle(parameters.getParameter< double >( prefix + "vertical-angle" ) * M_PI / 180.0 );
         setXYAngle(parameters.getParameter< double >( prefix + "horizontal-angle" ) * M_PI / 180.0 );
         return true;
      }

      void setCompressibleConservativeVariables(const CompressibleConservativeVariablesPointer& compressibleConservativeVariables)
      {
         this->densityBoundaryConditionsPointer->setCompressibleConservativeVariables(compressibleConservativeVariables);
         this->momentumXBoundaryConditionsPointer->setCompressibleConservativeVariables(compressibleConservativeVariables);
         this->momentumYBoundaryConditionsPointer->setCompressibleConservativeVariables(compressibleConservativeVariables);
         this->momentumZBoundaryConditionsPointer->setCompressibleConservativeVariables(compressibleConservativeVariables);
         this->energyBoundaryConditionsPointer->setCompressibleConservativeVariables(compressibleConservativeVariables);
         this->turbulentEnergyBoundaryConditionsPointer->setCompressibleConservativeVariables(compressibleConservativeVariables);
         this->disipationBoundaryConditionsPointer->setCompressibleConservativeVariables(compressibleConservativeVariables);
      }

      void setTimestep(const RealType timestep)
      {
         this->densityBoundaryConditionsPointer->setTimestep(timestep);
         this->momentumXBoundaryConditionsPointer->setTimestep(timestep);
         this->momentumYBoundaryConditionsPointer->setTimestep(timestep);
         this->momentumZBoundaryConditionsPointer->setTimestep(timestep);
         this->energyBoundaryConditionsPointer->setTimestep(timestep);   
      }

      void setGamma(const RealType gamma)
      {
         this->densityBoundaryConditionsPointer->setGamma(gamma);
         this->momentumXBoundaryConditionsPointer->setGamma(gamma);
         this->momentumYBoundaryConditionsPointer->setGamma(gamma);
         this->momentumZBoundaryConditionsPointer->setGamma(gamma);
         this->energyBoundaryConditionsPointer->setGamma(gamma);
      }

      void setZAngle(const RealType zAngle)
      {
         this->densityBoundaryConditionsPointer->setZAngle(zAngle);
         this->momentumXBoundaryConditionsPointer->setZAngle(zAngle);
         this->momentumYBoundaryConditionsPointer->setZAngle(zAngle);
         this->momentumZBoundaryConditionsPointer->setZAngle(zAngle);
         this->energyBoundaryConditionsPointer->setZAngle(zAngle);
         this->turbulentEnergyBoundaryConditionsPointer->setZAngle(zAngle);
         this->disipationBoundaryConditionsPointer->setZAngle(zAngle);
      }

      void setXYAngle(const RealType xYAngle)
      {
         this->densityBoundaryConditionsPointer->setXYAngle(xYAngle);
         this->momentumXBoundaryConditionsPointer->setXYAngle(xYAngle);
         this->momentumYBoundaryConditionsPointer->setXYAngle(xYAngle);
         this->momentumZBoundaryConditionsPointer->setXYAngle(xYAngle);
         this->turbulentEnergyBoundaryConditionsPointer->setXYAngle(xYAngle);
         this->disipationBoundaryConditionsPointer->setXYAngle(xYAngle);
      }

      void setPressure(const MeshFunctionPointer& pressure)
      {
         this->densityBoundaryConditionsPointer->setPressure(pressure);
         this->momentumXBoundaryConditionsPointer->setPressure(pressure);
         this->momentumYBoundaryConditionsPointer->setPressure(pressure);
         this->momentumZBoundaryConditionsPointer->setPressure(pressure);
         this->energyBoundaryConditionsPointer->setPressure(pressure);
      }

      void setIntensity(const RealType& intensity)
      {
         this->turbulentEnergyBoundaryConditionsPointer->setIntensity( intensity );
         this->disipationBoundaryConditionsPointer->setIntensity( intensity );
      }

      void setLengthScale( const RealType& lengthScale )
      {
         this->disipationBoundaryConditionsPointer->setLengthScale ( lengthScale );
      }

      void setTurbulenceConstant( const RealType& turbulenceConstant )
      {
         this->disipationBoundaryConditionsPointer->setTurbulenceConstant ( turbulenceConstant );
      }

      void setSpeed(const RealType cavitySpeed)
      {

      }

      void setHorizontalThrottleSpeed(const RealType startSpeed,
                                      const RealType finalSpeed,
                                      const RealType time,
                                      const RealType finalTime )
      {
         RealType horizontalThrottleSpeed = 0;
         if(time <= finalTime)
            if( time != 0 && finalTime != 0 )
               horizontalThrottleSpeed = startSpeed + ( finalSpeed - startSpeed ) * ( time / finalTime );
            else
               horizontalThrottleSpeed = 0;
         else
            horizontalThrottleSpeed = finalSpeed;
         this->momentumXBoundaryConditionsPointer->setHorizontalThrottleSpeed(horizontalThrottleSpeed);
         this->momentumYBoundaryConditionsPointer->setHorizontalThrottleSpeed(horizontalThrottleSpeed);
         this->momentumZBoundaryConditionsPointer->setHorizontalThrottleSpeed(horizontalThrottleSpeed);
         this->energyBoundaryConditionsPointer->setHorizontalThrottleSpeed(horizontalThrottleSpeed);
         this->turbulentEnergyBoundaryConditionsPointer->setHorizontalThrottleSpeed(horizontalThrottleSpeed);
         this->disipationBoundaryConditionsPointer->setHorizontalThrottleSpeed(horizontalThrottleSpeed);
      }

      void setVerticalThrottleSpeed(const RealType startSpeed,
                                    const RealType finalSpeed,
                                    const RealType time,
                                    const RealType finalTime )
      {
         RealType verticalThrottleSpeed = 0;
         if(time <= finalTime)
            if( time != 0 && finalTime != 0 )
               verticalThrottleSpeed = startSpeed + ( finalSpeed - startSpeed ) * ( time / finalTime );
            else
               verticalThrottleSpeed = 0;
         else
            verticalThrottleSpeed = finalSpeed;
         this->momentumXBoundaryConditionsPointer->setVerticalThrottleSpeed(verticalThrottleSpeed);
         this->momentumYBoundaryConditionsPointer->setVerticalThrottleSpeed(verticalThrottleSpeed);
         this->momentumZBoundaryConditionsPointer->setVerticalThrottleSpeed(verticalThrottleSpeed);
         this->energyBoundaryConditionsPointer->setVerticalThrottleSpeed(verticalThrottleSpeed);
         this->turbulentEnergyBoundaryConditionsPointer->setVerticalThrottleSpeed(verticalThrottleSpeed);
         this->disipationBoundaryConditionsPointer->setVerticalThrottleSpeed(verticalThrottleSpeed);
      }

      DensityBoundaryConditionsTypePointer& getDensityBoundaryCondition()
      {
         return this->densityBoundaryConditionsPointer;
      }

      MomentumXBoundaryConditionsTypePointer& getMomentumXBoundaryCondition()
      {
         return this->momentumXBoundaryConditionsPointer;
      }

      MomentumYBoundaryConditionsTypePointer& getMomentumYBoundaryCondition()
      {
         return this->momentumYBoundaryConditionsPointer;
      }

      MomentumZBoundaryConditionsTypePointer& getMomentumZBoundaryCondition()
      {
         return this->momentumZBoundaryConditionsPointer;
      }

      EnergyBoundaryConditionsTypePointer& getEnergyBoundaryCondition()
      {
         return this->energyBoundaryConditionsPointer;
      }

      TurbulentEnergyBoundaryConditionsTypePointer& getTurbulentEnergyBoundaryCondition()
      {
         return this->turbulentEnergyBoundaryConditionsPointer;
      }

      DisipationBoundaryConditionsTypePointer& getDisipationBoundaryCondition()
      {
         return this->disipationBoundaryConditionsPointer;
      }


   protected:
      DensityBoundaryConditionsTypePointer densityBoundaryConditionsPointer;
      MomentumXBoundaryConditionsTypePointer momentumXBoundaryConditionsPointer;
      MomentumYBoundaryConditionsTypePointer momentumYBoundaryConditionsPointer;
      MomentumZBoundaryConditionsTypePointer momentumZBoundaryConditionsPointer;
      EnergyBoundaryConditionsTypePointer energyBoundaryConditionsPointer;
      TurbulentEnergyBoundaryConditionsTypePointer turbulentEnergyBoundaryConditionsPointer;
      DisipationBoundaryConditionsTypePointer disipationBoundaryConditionsPointer;

};

} //namespace TNL
