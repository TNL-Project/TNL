#include <TNL/Functions/FunctionAdapter.h>

#include "DensityBoundaryConditionCavity.h"
#include "MomentumXBoundaryConditionCavity.h"
#include "MomentumYBoundaryConditionCavity.h"
#include "MomentumZBoundaryConditionCavity.h"
#include "AUSMPlusEnergyBoundaryConditionCavity.h"
#include "TurbulentEnergyBoundaryConditionCavity.h"

namespace TNL {

template< typename Mesh,
          typename Function,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class BoundaryConditionsCavity
{
   public:
      typedef Mesh MeshType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef Function FunctionType;
      typedef Functions::MeshFunction< Mesh > MeshFunctionType;
      typedef typename Mesh::DeviceType DeviceType;

      typedef TNL::Operators::DensityBoundaryConditionsCavity< MeshType, FunctionType, RealType, IndexType > DensityBoundaryConditionsType;
      typedef TNL::Operators::MomentumXBoundaryConditionsCavity< MeshType, FunctionType, RealType, IndexType > MomentumXBoundaryConditionsType;
      typedef TNL::Operators::MomentumYBoundaryConditionsCavity< MeshType, FunctionType, RealType, IndexType > MomentumYBoundaryConditionsType;
      typedef TNL::Operators::MomentumZBoundaryConditionsCavity< MeshType, FunctionType, RealType, IndexType > MomentumZBoundaryConditionsType;
      typedef TNL::Operators::AUSMPlusEnergyBoundaryConditionsCavity< MeshType, FunctionType, RealType, IndexType > EnergyBoundaryConditionsType;
      typedef TNL::Operators::TurbulentEnergyBoundaryConditionsCavity< MeshType, FunctionType, RealType, IndexType > TurbulentEnergyBoundaryConditionsType;
      typedef CompressibleConservativeVariables< MeshType > CompressibleConservativeVariablesType;

      typedef Pointers::SharedPointer< DensityBoundaryConditionsType > DensityBoundaryConditionsTypePointer;
      typedef Pointers::SharedPointer< MomentumXBoundaryConditionsType > MomentumXBoundaryConditionsTypePointer;
      typedef Pointers::SharedPointer< MomentumYBoundaryConditionsType > MomentumYBoundaryConditionsTypePointer;
      typedef Pointers::SharedPointer< MomentumZBoundaryConditionsType > MomentumZBoundaryConditionsTypePointer;
      typedef Pointers::SharedPointer< EnergyBoundaryConditionsType > EnergyBoundaryConditionsTypePointer;
      typedef Pointers::SharedPointer< TurbulentEnergyBoundaryConditionsType > TurbulentEnergyBoundaryConditionsTypePointer;
      typedef Pointers::SharedPointer< CompressibleConservativeVariablesType > CompressibleConservativeVariablesPointer;
      typedef Pointers::SharedPointer< MeshType > MeshPointer;
      typedef Pointers::SharedPointer< MeshFunctionType, DeviceType > MeshFunctionPointer;

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
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
         this->turbulentEnergyBoundaryConditionsPointer->setup( meshPointer, parameters, prefix);
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

      void setPressure(const MeshFunctionPointer& pressure)
      {
         this->densityBoundaryConditionsPointer->setPressure(pressure);
         this->momentumXBoundaryConditionsPointer->setPressure(pressure);
         this->momentumYBoundaryConditionsPointer->setPressure(pressure);
         this->momentumZBoundaryConditionsPointer->setPressure(pressure);
         this->energyBoundaryConditionsPointer->setPressure(pressure);
      }

      void setSpeed(const RealType cavitySpeed)
      {
         this->momentumXBoundaryConditionsPointer->setCavitySpeed(cavitySpeed);
         this->momentumYBoundaryConditionsPointer->setCavitySpeed(cavitySpeed);
         this->momentumZBoundaryConditionsPointer->setCavitySpeed(cavitySpeed);
         this->energyBoundaryConditionsPointer->setCavitySpeed(cavitySpeed);
         this->turbulentEnergyBoundaryConditionsPointer->setCavitySpeed(cavitySpeed);
      }

      void setIntensity(const RealType& intensity)
      {
         this->turbulentEnergyBoundaryConditionsPointer->setIntensity( intensity );
      }

      void setHorizontalThrottleSpeed(const RealType startSpeed,
                                      const RealType finalSpeed,
                                      const RealType time,
                                      const RealType finalTime )
      {

      }

      void setVerticalThrottleSpeed(const RealType startSpeed,
                                    const RealType finalSpeed,
                                    const RealType time,
                                    const RealType finalTime )
      {

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


   protected:
      DensityBoundaryConditionsTypePointer densityBoundaryConditionsPointer;
      MomentumXBoundaryConditionsTypePointer momentumXBoundaryConditionsPointer;
      MomentumYBoundaryConditionsTypePointer momentumYBoundaryConditionsPointer;
      MomentumZBoundaryConditionsTypePointer momentumZBoundaryConditionsPointer;
      EnergyBoundaryConditionsTypePointer energyBoundaryConditionsPointer;
      TurbulentEnergyBoundaryConditionsTypePointer turbulentEnergyBoundaryConditionsPointer;

};

} //namespace TNL
