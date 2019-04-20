#include <TNL/Functions/FunctionAdapter.h>

#include "DensityBoundaryConditionBoiler.h"
#include "MomentumXBoundaryConditionBoiler.h"
#include "MomentumYBoundaryConditionBoiler.h"
#include "MomentumZBoundaryConditionBoiler.h"
#include "EnergyBoundaryConditionBoiler.h"
#include "TurbulentEnergyBoundaryConditionBoiler.h"
#include "DisipationBoundaryConditionBoiler.h"

namespace TNL {

template< typename Mesh,
          typename Function,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class BoundaryConditionsBoiler
{
   public:
      typedef Mesh MeshType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef Function FunctionType;
      typedef Functions::MeshFunction< Mesh > MeshFunctionType;
      typedef typename Mesh::DeviceType DeviceType;

      typedef TNL::Operators::DensityBoundaryConditionsBoiler< MeshType, FunctionType, RealType, IndexType > DensityBoundaryConditionsType;
      typedef TNL::Operators::MomentumXBoundaryConditionsBoiler< MeshType, FunctionType, RealType, IndexType > MomentumXBoundaryConditionsType;
      typedef TNL::Operators::MomentumYBoundaryConditionsBoiler< MeshType, FunctionType, RealType, IndexType > MomentumYBoundaryConditionsType;
      typedef TNL::Operators::MomentumZBoundaryConditionsBoiler< MeshType, FunctionType, RealType, IndexType > MomentumZBoundaryConditionsType;
      typedef TNL::Operators::EnergyBoundaryConditionsBoiler< MeshType, FunctionType, RealType, IndexType > EnergyBoundaryConditionsType;
      typedef TNL::Operators::TurbulentEnergyBoundaryConditionsBoiler< MeshType, FunctionType, RealType, IndexType > TurbulentEnergyBoundaryConditionsType;
      typedef TNL::Operators::DisipationBoundaryConditionsBoiler< MeshType, FunctionType, RealType, IndexType > DisipationBoundaryConditionsType;
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
         return true;
      }

      void setCompressibleConservativeVariables(const CompressibleConservativeVariablesPointer& compressibleConservativeVariables)
      {
         this->densityBoundaryConditionsPointer->setCompressibleConservativeVariables(compressibleConservativeVariables);
         this->momentumXBoundaryConditionsPointer->setCompressibleConservativeVariables(compressibleConservativeVariables);
         this->momentumYBoundaryConditionsPointer->setCompressibleConservativeVariables(compressibleConservativeVariables);
         this->momentumZBoundaryConditionsPointer->setCompressibleConservativeVariables(compressibleConservativeVariables);
         this->energyBoundaryConditionsPointer->setCompressibleConservativeVariables(compressibleConservativeVariables);
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
         this->disipationBoundaryConditionsPointer->setCavitySpeed(cavitySpeed);
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
