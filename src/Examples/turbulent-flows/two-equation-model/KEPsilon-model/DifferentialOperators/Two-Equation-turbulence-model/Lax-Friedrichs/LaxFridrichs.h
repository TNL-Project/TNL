/***************************************************************************
                          LaxFridrichs.h  -  description
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

#include "LaxFridrichsContinuity.h"
#include "LaxFridrichsEnergy.h"
#include "LaxFridrichsMomentumX.h"
#include "LaxFridrichsMomentumY.h"
#include "LaxFridrichsMomentumZ.h"
#include "LaxFridrichsTurbulentEnergy.h"
#include "LaxFridrichsDisipation.h"

#include "Examples/turbulent-flows/two-equation-model/KEPsilon-model/DifferentialOperatorsRightHandSide/KEPsilonRightHandSide/KEpsilonOperatorRightHandSide.h"

namespace TNL {

template< typename Mesh,
	  typename OperatorRightHandSide = KEpsilonOperatorRightHandSide< Mesh, typename Mesh::RealType, typename Mesh::IndexType >,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class LaxFridrichs
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
      typedef typename OperatorRightHandSide::DisipationOperatorRightHandSideType DisipationOperatorRightHandSideType;
 
      typedef LaxFridrichsContinuity< Mesh, ContinuityOperatorRightHandSideType, Real, Index > ContinuityOperatorType;
      typedef LaxFridrichsMomentumX< Mesh, MomentumXOperatorRightHandSideType, Real, Index > MomentumXOperatorType;
      typedef LaxFridrichsMomentumY< Mesh, MomentumYOperatorRightHandSideType, Real, Index > MomentumYOperatorType;
      typedef LaxFridrichsMomentumZ< Mesh, MomentumZOperatorRightHandSideType, Real, Index > MomentumZOperatorType;
      typedef LaxFridrichsEnergy< Mesh, EnergyOperatorRightHandSideType, Real, Index > EnergyOperatorType;
      typedef LaxFridrichsTurbulentEnergy< Mesh, TurbulentEnergyOperatorRightHandSideType, Real, Index > TurbulentEnergyOperatorType;
      typedef LaxFridrichsDisipation< Mesh, DisipationOperatorRightHandSideType, Real, Index > DisipationOperatorType;

      typedef Pointers::SharedPointer< MeshFunctionType > MeshFunctionPointer;
      typedef Pointers::SharedPointer< VectorFieldType > VectorFieldPointer;
      typedef Pointers::SharedPointer< MeshType > MeshPointer;
      
      typedef Pointers::SharedPointer< ContinuityOperatorType > ContinuityOperatorPointer;
      typedef Pointers::SharedPointer< MomentumXOperatorType > MomentumXOperatorPointer;
      typedef Pointers::SharedPointer< MomentumYOperatorType > MomentumYOperatorPointer;      
      typedef Pointers::SharedPointer< MomentumZOperatorType > MomentumZOperatorPointer;      
      typedef Pointers::SharedPointer< EnergyOperatorType > EnergyOperatorPointer;
      typedef Pointers::SharedPointer< TurbulentEnergyOperatorType > TurbulentEnergyOperatorPointer;
      typedef Pointers::SharedPointer< DisipationOperatorType > DisipationOperatorPointer;

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         config.addEntry< double >( prefix + "numerical-viscosity", "Value of artificial (numerical) viscosity in the Lax-Fridrichs scheme", 1.0 );
         config.addEntry< double >( prefix + "dynamical-viscosity", "Value of dynamical (real) viscosity in the Navier-Stokes equation", 1.0 );
      }
      
      LaxFridrichs()
         : artificialViscosity( 1.0 ) {}
      
      bool setup( const MeshPointer& meshPointer,
                  const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         std::cout << "Lax-Friedrichs" << std::endl;
         this->dynamicalViscosity = parameters.getParameter< double >( prefix + "dynamical-viscosity" );
         this->momentumXOperatorPointer->setDynamicalViscosity( dynamicalViscosity );
         this->momentumYOperatorPointer->setDynamicalViscosity( dynamicalViscosity );
         this->momentumZOperatorPointer->setDynamicalViscosity( dynamicalViscosity );
         this->energyOperatorPointer->setDynamicalViscosity( dynamicalViscosity );
         this->turbulentEnergyOperatorPointer->setDynamicalViscosity( dynamicalViscosity );
         this->disipationOperatorPointer->setDynamicalViscosity( dynamicalViscosity );
         this->artificialViscosity = parameters.getParameter< double >( prefix + "numerical-viscosity" );
         this->continuityOperatorPointer->setArtificialViscosity( artificialViscosity );
         this->momentumXOperatorPointer->setArtificialViscosity( artificialViscosity );
         this->momentumYOperatorPointer->setArtificialViscosity( artificialViscosity );
         this->momentumZOperatorPointer->setArtificialViscosity( artificialViscosity );
         this->energyOperatorPointer->setArtificialViscosity( artificialViscosity );
         this->turbulentEnergyOperatorPointer->setArtificialViscosity( artificialViscosity );
         
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
         this->disipationOperatorPointer->setTau( tau );
      }

      void setGamma( const RealType& gamma )
      {
      }
      
      void setPressure( const MeshFunctionPointer& pressure )
      {
         this->momentumXOperatorPointer->setPressure( pressure );
         this->momentumYOperatorPointer->setPressure( pressure );
         this->momentumZOperatorPointer->setPressure( pressure );
         this->energyOperatorPointer->setPressure( pressure );
      }
      
      void setVelocity( const VectorFieldPointer& velocity )
      {
         this->continuityOperatorPointer->setVelocity( velocity );
         this->momentumXOperatorPointer->setVelocity( velocity );
         this->momentumYOperatorPointer->setVelocity( velocity );
         this->momentumZOperatorPointer->setVelocity( velocity );
         this->energyOperatorPointer->setVelocity( velocity );
         this->turbulentEnergyOperatorPointer->setVelocity( velocity );
         this->disipationOperatorPointer->setVelocity( velocity );
      }

      void setDensity( const MeshFunctionPointer& density )
      {
         this->momentumXOperatorPointer->setDensity( density );
         this->momentumYOperatorPointer->setDensity( density );
         this->momentumZOperatorPointer->setDensity( density );
         this->energyOperatorPointer->setDensity( density );
         this->disipationOperatorPointer->setDensity( density );
      }

      void setTurbulentViscosity( const MeshFunctionPointer& turbulentViscosity )
      {
         this->momentumXOperatorPointer->setTurbulentViscosity( turbulentViscosity );
         this->momentumYOperatorPointer->setTurbulentViscosity( turbulentViscosity );
         this->momentumZOperatorPointer->setTurbulentViscosity( turbulentViscosity );
         this->energyOperatorPointer->setTurbulentViscosity( turbulentViscosity );
         this->turbulentEnergyOperatorPointer->setTurbulentViscosity( turbulentViscosity );
         this->disipationOperatorPointer->setTurbulentViscosity( turbulentViscosity );
      }

      void setCharacteristicLength( RealType& characteristicLength )
      {
         this->turbulentEnergyOperatorPointer->setCharacteristicLength( characteristicLength );
      }

      void setSigmaK( RealType& sigmaK )
      {
         this->turbulentEnergyOperatorPointer->setSigmaK( sigmaK );
      }

      void setTurbulentEnergy( const MeshFunctionPointer& turbulentEnergy )
      {
         this->turbulentEnergyOperatorPointer->setTurbulentEnergy( turbulentEnergy );
         this->momentumXOperatorPointer->setTurbulentEnergy( turbulentEnergy );
         this->momentumYOperatorPointer->setTurbulentEnergy( turbulentEnergy );
         this->momentumZOperatorPointer->setTurbulentEnergy( turbulentEnergy );
         this->energyOperatorPointer->setTurbulentEnergy( turbulentEnergy );
         this->disipationOperatorPointer->setTurbulentEnergy( turbulentEnergy );
      }     
      
      void setViscosityConstant1( const RealType& viscosityConstant1 )
      {
         this->disipationOperatorPointer->setViscosityConstant1( viscosityConstant1 );
      }     
      
      void setViscosityConstant2( const RealType& viscosityConstant2 )
      {
         this->disipationOperatorPointer->setViscosityConstant2( viscosityConstant2 );
      }     
      
      void setSigmaEpsilon( const RealType& sigmaEpsilon )
      {
         this->disipationOperatorPointer->setSigmaEpsilon( sigmaEpsilon );
      }
    
      void setDisipation( const MeshFunctionPointer& disipation )
      {
          this->disipationOperatorPointer->setDisipation( disipation );
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
 
      const DisipationOperatorPointer& getDisipationOperator() const
      {
         return this->disipationOperatorPointer;
      }

   protected:
      
      ContinuityOperatorPointer continuityOperatorPointer;
      MomentumXOperatorPointer momentumXOperatorPointer;
      MomentumYOperatorPointer momentumYOperatorPointer;
      MomentumZOperatorPointer momentumZOperatorPointer;
      EnergyOperatorPointer energyOperatorPointer; 
      TurbulentEnergyOperatorPointer turbulentEnergyOperatorPointer; 
      DisipationOperatorPointer disipationOperatorPointer;    
      
      RealType artificialViscosity;
      RealType dynamicalViscosity;
};

} //namespace TNL
