/***************************************************************************
                          AUSMPlus.h  -  description
                             -------------------
    begin                : Feb 18, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Functions/VectorField.h>

#include "AUSMPlusContinuity.h"
#include "AUSMPlusEnergy.h"
#include "AUSMPlusMomentumX.h"
#include "AUSMPlusMomentumY.h"
#include "AUSMPlusMomentumZ.h"
#include "AUSMPlusTurbulentEnergy.h"

#include "Examples/turbulent-flows/one-equation-model/DifferentialOperatorsRightHandSide/PrandtlKolmogorovRightHandSide/PrandtlKolmogorovOperatorRightHandSide.h"

namespace TNL {

template< typename Mesh,
	  typename OperatorRightHandSide = PrandtlKolmogorovOperatorRightHandSide < Mesh, typename Mesh::RealType, typename Mesh::IndexType >,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class AUSMPlus
{
   public:
      typedef Mesh MeshType;
      typedef Real RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< Mesh > MeshFunctionType;
      static const int Dimensions = Mesh::getMeshDimension();
      typedef Functions::VectorField< Dimensions, MeshFunctionType > VectorFieldType;

      typedef typename OperatorRightHandSide::ContinuityOperatorRightHandSideType ContinuityOperatorRightHandSideType;
      typedef typename OperatorRightHandSide::MomentumXOperatorRightHandSideType MomentumXOperatorRightHandSideType;
      typedef typename OperatorRightHandSide::MomentumYOperatorRightHandSideType MomentumYOperatorRightHandSideType;
      typedef typename OperatorRightHandSide::MomentumZOperatorRightHandSideType MomentumZOperatorRightHandSideType;
      typedef typename OperatorRightHandSide::EnergyOperatorRightHandSideType EnergyOperatorRightHandSideType;
      typedef typename OperatorRightHandSide::TurbulentEnergyOperatorRightHandSideType TurbulentEnergyOperatorRightHandSideType;
 
      typedef AUSMPlusContinuity< Mesh, ContinuityOperatorRightHandSideType, Real, Index > ContinuityOperatorType;
      typedef AUSMPlusMomentumX< Mesh, MomentumXOperatorRightHandSideType, Real, Index > MomentumXOperatorType;
      typedef AUSMPlusMomentumY< Mesh, MomentumYOperatorRightHandSideType, Real, Index > MomentumYOperatorType;
      typedef AUSMPlusMomentumZ< Mesh, MomentumZOperatorRightHandSideType, Real, Index > MomentumZOperatorType;
      typedef AUSMPlusEnergy< Mesh, EnergyOperatorRightHandSideType, Real, Index > EnergyOperatorType;
      typedef AUSMPlusTurbulentEnergy< Mesh, TurbulentEnergyOperatorRightHandSideType, Real, Index > TurbulentEnergyOperatorType;

      typedef Pointers::SharedPointer< MeshFunctionType > MeshFunctionPointer;
      typedef Pointers::SharedPointer< VectorFieldType > VectorFieldPointer;
      typedef Pointers::SharedPointer< MeshType > MeshPointer;
      
      typedef Pointers::SharedPointer< ContinuityOperatorType > ContinuityOperatorPointer;
      typedef Pointers::SharedPointer< MomentumXOperatorType > MomentumXOperatorPointer;
      typedef Pointers::SharedPointer< MomentumYOperatorType > MomentumYOperatorPointer;      
      typedef Pointers::SharedPointer< MomentumZOperatorType > MomentumZOperatorPointer;      
      typedef Pointers::SharedPointer< EnergyOperatorType > EnergyOperatorPointer;
      typedef Pointers::SharedPointer< TurbulentEnergyOperatorType > TurbulentEnergyOperatorPointer;

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         config.addEntry< double >( prefix + "dynamical-viscosity", "Value of dynamical (real) viscosity in the Navier-Stokes equation", 1.0 );
      }
      
      AUSMPlus()
         :dynamicalViscosity( 1.0 ) {}
      
      bool setup( const MeshPointer& meshPointer,
                  const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         std::cout << "AUSMPlus" << std::endl;
         this->dynamicalViscosity = parameters.getParameter< double >( prefix + "dynamical-viscosity" );
         this->momentumXOperatorPointer->setDynamicalViscosity( dynamicalViscosity );
         this->momentumYOperatorPointer->setDynamicalViscosity( dynamicalViscosity );
         this->momentumZOperatorPointer->setDynamicalViscosity( dynamicalViscosity );
         this->energyOperatorPointer->setDynamicalViscosity( dynamicalViscosity );
         this->turbulentEnergyOperatorPointer->setDynamicalViscosity( dynamicalViscosity );

         return true;
      }
      
      void setTau( const RealType& tau )
      {
         this->continuityOperatorPointer->setTau( tau );
         this->momentumXOperatorPointer->setTau( tau );
         this->momentumYOperatorPointer->setTau( tau );
         this->momentumZOperatorPointer->setTau( tau );
         this->energyOperatorPointer->setTau( tau );
         this->turbulentEnergyOperatorPointer->setTau( tau );
      }

      void setGamma( const RealType& gamma )
      {
         this->continuityOperatorPointer->setGamma( gamma );
         this->momentumXOperatorPointer->setGamma( gamma );
         this->momentumYOperatorPointer->setGamma( gamma );
         this->momentumZOperatorPointer->setGamma( gamma );
         this->energyOperatorPointer->setGamma( gamma );
         this->turbulentEnergyOperatorPointer->setGamma( gamma );
      }
      
      void setPressure( const MeshFunctionPointer& pressure )
      {
         this->continuityOperatorPointer->setPressure( pressure );
         this->momentumXOperatorPointer->setPressure( pressure );
         this->momentumYOperatorPointer->setPressure( pressure );
         this->momentumZOperatorPointer->setPressure( pressure );
         this->energyOperatorPointer->setPressure( pressure );
         this->turbulentEnergyOperatorPointer->setPressure( pressure );
      }

      void setDensity( const MeshFunctionPointer& density )
      {
         this->momentumXOperatorPointer->setDensity( density );
         this->momentumYOperatorPointer->setDensity( density );
         this->momentumZOperatorPointer->setDensity( density );
         this->energyOperatorPointer->setDensity( density );
         this->turbulentEnergyOperatorPointer->setDensity( density );
      }
      
      void setVelocity( const VectorFieldPointer& velocity )
      {
         this->continuityOperatorPointer->setVelocity( velocity );
         this->momentumXOperatorPointer->setVelocity( velocity );
         this->momentumYOperatorPointer->setVelocity( velocity );
         this->momentumZOperatorPointer->setVelocity( velocity );
         this->energyOperatorPointer->setVelocity( velocity );
         this->turbulentEnergyOperatorPointer->setVelocity( velocity );
      }

      void setTurbulentViscosity( const MeshFunctionPointer& turbulentViscosity )
      {
         this->momentumXOperatorPointer->setTurbulentViscosity( turbulentViscosity );
         this->momentumYOperatorPointer->setTurbulentViscosity( turbulentViscosity );
         this->momentumZOperatorPointer->setTurbulentViscosity( turbulentViscosity );
         this->energyOperatorPointer->setTurbulentViscosity( turbulentViscosity );
         this->turbulentEnergyOperatorPointer->setTurbulentViscosity( turbulentViscosity );
      }

      void setCharacteristicLength( RealType& characteristicLength )
      {
         this->turbulentEnergyOperatorPointer->setCharacteristicLength( characteristicLength );
      }

      void setSigmaK( RealType& sigmaK )
      {
         this->turbulentEnergyOperatorPointer->setSigmaK( sigmaK );
      }

      void setViscosityConstant( RealType& ViscosityConstant )
      {
         this->turbulentEnergyOperatorPointer->setViscosityConstant( ViscosityConstant );
      }

      void setTurbulentEnergy( const MeshFunctionPointer& turbulentEnergy )
      {
         this->momentumXOperatorPointer->setTurbulentEnergy( turbulentEnergy );
         this->momentumYOperatorPointer->setTurbulentEnergy( turbulentEnergy );
         this->momentumZOperatorPointer->setTurbulentEnergy( turbulentEnergy );
         this->energyOperatorPointer->setTurbulentEnergy( turbulentEnergy );
         this->turbulentEnergyOperatorPointer->setTurbulentEnergy( turbulentEnergy );
      }
      
      const ContinuityOperatorPointer& getContinuityOperator() const
      {
         return this->continuityOperatorPointer;
      }
      
      const MomentumXOperatorPointer& getMomentumXOperator() const
      {
         return this->momentumXOperatorPointer;
      }

      const MomentumYOperatorPointer& getMomentumYOperator() const
      {
         return this->momentumYOperatorPointer;
      }
      
      const MomentumZOperatorPointer& getMomentumZOperator() const
      {
         return this->momentumZOperatorPointer;
      }
      
      const EnergyOperatorPointer& getEnergyOperator() const
      {
         return this->energyOperatorPointer;
      }

      const TurbulentEnergyOperatorPointer& getTurbulentEnergyOperator() const
      {
         return this->turbulentEnergyOperatorPointer;
      }

   protected:
      
      ContinuityOperatorPointer continuityOperatorPointer;
      MomentumXOperatorPointer momentumXOperatorPointer;
      MomentumYOperatorPointer momentumYOperatorPointer;
      MomentumZOperatorPointer momentumZOperatorPointer;
      EnergyOperatorPointer energyOperatorPointer; 
      TurbulentEnergyOperatorPointer turbulentEnergyOperatorPointer;  
      
      RealType dynamicalViscosity;
};

} //namespace TNL
