#include <TNL/Functions/FunctionAdapter.h>
#include <TNL/Operators/NeumannBoundaryConditions.h>

namespace TNL {

template< typename Mesh,
          typename Function,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::GlobalIndexType >
class BoundaryConditionsNeumann
{
   public:
      typedef Mesh MeshType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef Function FunctionType;
      typedef Functions::MeshFunction< Mesh > MeshFunctionType;
      typedef typename Mesh::DeviceType DeviceType;

      typedef TNL::Operators::NeumannBoundaryConditions< MeshType, FunctionType, RealType, IndexType > DensityBoundaryConditionsType;
      typedef TNL::Operators::NeumannBoundaryConditions< MeshType, FunctionType, RealType, IndexType > MomentumXBoundaryConditionsType;
      typedef TNL::Operators::NeumannBoundaryConditions< MeshType, FunctionType, RealType, IndexType > MomentumYBoundaryConditionsType;
      typedef TNL::Operators::NeumannBoundaryConditions< MeshType, FunctionType, RealType, IndexType > MomentumZBoundaryConditionsType;
      typedef TNL::Operators::NeumannBoundaryConditions< MeshType, FunctionType, RealType, IndexType > EnergyBoundaryConditionsType;
      typedef TNL::Operators::NeumannBoundaryConditions< MeshType, FunctionType, RealType, IndexType > TurbulentEnergyBoundaryConditionsType;
      typedef TNL::Operators::NeumannBoundaryConditions< MeshType, FunctionType, RealType, IndexType > DisipationBoundaryConditionsType;
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
         return true;
      }

      void setCompressibleConservativeVariables(const CompressibleConservativeVariablesPointer& compressibleConservativeVariables)
      {
        
      }

      void setTimestep(const RealType timestep)
      {
            
      }

      void setGamma(const RealType gamma)
      {
         
      }

      void setPressure(const MeshFunctionPointer& pressure)
      {
         
      }

      void setSpeed(const RealType cavitySpeed)
      {
         
      }

      void setIntensity(const RealType& intensity)
      {

      }

      void setLengthScale( const RealType& lengthScale )
      {

      }

      void setTurbulenceConstant( const RealType& turbulenceConstant )
      {

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
