/***************************************************************************
                          VanLeer.h  -  description
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

#include "VanLeerContinuity.h"
#include "VanLeerEnergy.h"
#include "VanLeerMomentumX.h"
#include "VanLeerMomentumY.h"
#include "VanLeerMomentumZ.h"
#include "VanLeerTurbulentEnergy.h"
#include "VanLeerDisipation.h"

#include "Examples/turbulent-flows/two-equation-model/KOmega-model/DifferentialOperatorsRightHandSide/KOmegaRightHandSide/KOmegaOperatorRightHandSide.h"

namespace TNL {

template< typename Mesh,
	  typename OperatorRightHandSide = KOmegaOperatorRightHandSide < Mesh, typename Mesh::RealType, typename Mesh::IndexType >,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class VanLeer
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
 
      typedef VanLeerContinuity< Mesh, ContinuityOperatorRightHandSideType, Real, Index > ContinuityOperatorType;
      typedef VanLeerMomentumX< Mesh, MomentumXOperatorRightHandSideType, Real, Index > MomentumXOperatorType;
      typedef VanLeerMomentumY< Mesh, MomentumYOperatorRightHandSideType, Real, Index > MomentumYOperatorType;
      typedef VanLeerMomentumZ< Mesh, MomentumZOperatorRightHandSideType, Real, Index > MomentumZOperatorType;
      typedef VanLeerEnergy< Mesh, EnergyOperatorRightHandSideType, Real, Index > EnergyOperatorType;
      typedef VanLeerTurbulentEnergy< Mesh, TurbulentEnergyOperatorRightHandSideType, Real, Index > TurbulentEnergyOperatorType;
      typedef VanLeerDisipation< Mesh, DisipationOperatorRightHandSideType, Real, Index > DisipationOperatorType;

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
         config.addEntry< double >( prefix + "dynamical-viscosity", "Value of dynamical (real) viscosity in the Navier-Stokes equation", 1.0 );
      }
      
      VanLeer()
         :dynamicalViscosity( 1.0 ) {}
      
      bool setup( const MeshPointer& meshPointer,
                  const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         std::cout << "VanLeer" << std::endl;
         this->dynamicalViscosity = parameters.getParameter< double >( prefix + "dynamical-viscosity" );
         this->momentumXOperatorPointer->setDynamicalViscosity( dynamicalViscosity );
         this->momentumYOperatorPointer->setDynamicalViscosity( dynamicalViscosity );
         this->momentumZOperatorPointer->setDynamicalViscosity( dynamicalViscosity );
         this->energyOperatorPointer->setDynamicalViscosity( dynamicalViscosity );
         this->turbulentEnergyOperatorPointer->setDynamicalViscosity( dynamicalViscosity );
         this->disipationOperatorPointer->setDynamicalViscosity( dynamicalViscosity );

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
         this->continuityOperatorPointer->setGamma( gamma );
         this->momentumXOperatorPointer->setGamma( gamma );
         this->momentumYOperatorPointer->setGamma( gamma );
         this->momentumZOperatorPointer->setGamma( gamma );
         this->energyOperatorPointer->setGamma( gamma );
         this->turbulentEnergyOperatorPointer->setGamma( gamma );
         this->disipationOperatorPointer->setGamma( gamma );
      }
      
      void setPressure( const MeshFunctionPointer& pressure )
      {
         this->continuityOperatorPointer->setPressure( pressure );
         this->momentumXOperatorPointer->setPressure( pressure );
         this->momentumYOperatorPointer->setPressure( pressure );
         this->momentumZOperatorPointer->setPressure( pressure );
         this->energyOperatorPointer->setPressure( pressure );
         this->turbulentEnergyOperatorPointer->setPressure( pressure );
         this->disipationOperatorPointer->setPressure( pressure );
      }

      void setDensity( const MeshFunctionPointer& density )
      {
         this->momentumXOperatorPointer->setDensity( density );
         this->momentumYOperatorPointer->setDensity( density );
         this->momentumZOperatorPointer->setDensity( density );
         this->energyOperatorPointer->setDensity( density );
         this->turbulentEnergyOperatorPointer->setDensity( density );
         this->disipationOperatorPointer->setDensity( density );
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

      void setTurbulentViscosity( const MeshFunctionPointer& turbulentViscosity )
      {
         this->momentumXOperatorPointer->setTurbulentViscosity( turbulentViscosity );
         this->momentumYOperatorPointer->setTurbulentViscosity( turbulentViscosity );
         this->momentumZOperatorPointer->setTurbulentViscosity( turbulentViscosity );
         this->energyOperatorPointer->setTurbulentViscosity( turbulentViscosity );
         this->turbulentEnergyOperatorPointer->setTurbulentViscosity( turbulentViscosity );
         this->disipationOperatorPointer->setTurbulentViscosity( turbulentViscosity );
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
         this->turbulentEnergyOperatorPointer->setTurbulentEnergy( turbulentEnergy );
         this->disipationOperatorPointer->setTurbulentEnergy( turbulentEnergy );
      }    
      
      void setSigmaEpsilon( const RealType& sigmaEpsilon )
      {
         this->disipationOperatorPointer->setSigmaEpsilon( sigmaEpsilon );
      }
    
      void setDisipation( const MeshFunctionPointer& disipation )
      {
          this->disipationOperatorPointer->setDisipation( disipation );
          this->turbulentEnergyOperatorPointer->setDisipation( disipation );
      }     
      
      void setBeta( const RealType& beta )
      {
         this->disipationOperatorPointer->setBeta( beta );
      }     
      
      void setAlpha( const RealType& alpha )
      {
        this->disipationOperatorPointer->setAlpha( alpha );
      }      
      
      void setBetaStar( const RealType& betaStar )
      {
         this->turbulentEnergyOperatorPointer->setBetaStar( betaStar );
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
      
      RealType dynamicalViscosity;
};

} //namespace TNL
