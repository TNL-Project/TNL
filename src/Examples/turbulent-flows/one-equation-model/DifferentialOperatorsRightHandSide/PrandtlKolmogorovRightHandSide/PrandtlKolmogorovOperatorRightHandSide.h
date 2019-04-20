/***************************************************************************
                          PrandtlKolmogorovOpratorRightHandSide.h  -  description
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

#include "PrandtlKolmogorovContinuityOperatorRightHandSide.h"
#include "PrandtlKolmogorovEnergyOperatorRightHandSide.h"
#include "PrandtlKolmogorovMomentumXOperatorRightHandSide.h"
#include "PrandtlKolmogorovMomentumYOperatorRightHandSide.h"
#include "PrandtlKolmogorovMomentumZOperatorRightHandSide.h"
#include "PrandtlKolmogorovTurbulentEnergyOperatorRightHandSide.h"

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class PrandtlKolmogorovOperatorRightHandSide
{
   public:
      typedef Mesh MeshType;
      typedef Real RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< Mesh > MeshFunctionType;
      static const int Dimensions = Mesh::getMeshDimension();
      typedef Functions::VectorField< Dimensions, MeshFunctionType > VectorFieldType;
 
      typedef PrandtlKolmogorovContinuityRightHandSide< Mesh, Real, Index > ContinuityOperatorRightHandSideType;
      typedef PrandtlKolmogorovMomentumXRightHandSide< Mesh, Real, Index > MomentumXOperatorRightHandSideType;
      typedef PrandtlKolmogorovMomentumYRightHandSide< Mesh, Real, Index > MomentumYOperatorRightHandSideType;
      typedef PrandtlKolmogorovMomentumZRightHandSide< Mesh, Real, Index > MomentumZOperatorRightHandSideType;
      typedef PrandtlKolmogorovEnergyRightHandSide< Mesh, Real, Index > EnergyOperatorRightHandSideType;
      typedef PrandtlKolmogorovTurbulentEnergyRightHandSide< Mesh, Real, Index > TurbulentEnergyOperatorRightHandSideType;

      typedef Pointers::SharedPointer< MeshFunctionType > MeshFunctionPointer;
      typedef Pointers::SharedPointer< VectorFieldType > VectorFieldPointer;
      typedef Pointers::SharedPointer< MeshType > MeshPointer;
      
      typedef Pointers::SharedPointer< ContinuityOperatorRightHandSideType > ContinuityOperatorRightHandSidePointer;
      typedef Pointers::SharedPointer< MomentumXOperatorRightHandSideType > MomentumXOperatorRightHandSidePointer;
      typedef Pointers::SharedPointer< MomentumYOperatorRightHandSideType > MomentumYOperatorRightHandSidePointer;      
      typedef Pointers::SharedPointer< MomentumZOperatorRightHandSideType > MomentumZOperatorRightHandSidePointer;      
      typedef Pointers::SharedPointer< EnergyOperatorRightHandSideType > EnergyOperatorRightHandSidePointer;
      typedef Pointers::SharedPointer< TurbulentEnergyOperatorRightHandSideType > TurbulentEnergyOperatorRightHandSidePointer;

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         config.addEntry< double >( prefix + "dynamical-viscosity", "Value of dynamical (real) viscosity in the Navier-Stokes equation", 1.0 );
      }
      
      PrandtlKolmogorovOperatorRightHandSide()
         : dynamicalViscosity( 1.0 ) {}
      
      bool setup( const MeshPointer& meshPointer,
                  const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         this->dynamicalViscosity = parameters.getParameter< double >( prefix + "dynamical-viscosity" );
         this->momentumXOperatorRightHandSidePointer->setDynamicalViscosity( dynamicalViscosity );
         this->momentumYOperatorRightHandSidePointer->setDynamicalViscosity( dynamicalViscosity );
         this->momentumZOperatorRightHandSidePointer->setDynamicalViscosity( dynamicalViscosity );
         this->energyOperatorRightHandSidePointer->setDynamicalViscosity( dynamicalViscosity );
         this->turbulentEnergyOperatorRightHandSidePointer->setDynamicalViscosity( dynamicalViscosity );
         return true;
      }
      
      void setVelocity( const VectorFieldPointer& velocity )
      {
         this->continuityOperatorRightHandSidePointer->setVelocity( velocity );
         this->momentumXOperatorRightHandSidePointer->setVelocity( velocity );
         this->momentumYOperatorRightHandSidePointer->setVelocity( velocity );
         this->momentumZOperatorRightHandSidePointer->setVelocity( velocity );
         this->energyOperatorRightHandSidePointer->setVelocity( velocity );
         this->turbulentEnergyOperatorRightHandSidePointer->setVelocity( velocity );
      }

      void setTurbulentViscosity( const MeshFunctionPointer& turbulentViscosity )
      {
         this->momentumXOperatorRightHandSidePointer->setTurbulentViscosity( turbulentViscosity );
         this->momentumYOperatorRightHandSidePointer->setTurbulentViscosity( turbulentViscosity );
         this->momentumZOperatorRightHandSidePointer->setTurbulentViscosity( turbulentViscosity );
         this->energyOperatorRightHandSidePointer->setTurbulentViscosity( turbulentViscosity );
         this->turbulentEnergyOperatorRightHandSidePointer->setTurbulentViscosity( turbulentViscosity );
      }

      void setTurbulentEnergy( const MeshFunctionPointer& turbulentEnergy )
      {
         this->momentumXOperatorRightHandSidePointer->setTurbulentEnergy( turbulentEnergy );
         this->momentumYOperatorRightHandSidePointer->setTurbulentEnergy( turbulentEnergy );
         this->momentumZOperatorRightHandSidePointer->setTurbulentEnergy( turbulentEnergy );
         this->energyOperatorRightHandSidePointer->setTurbulentEnergy( turbulentEnergy );
         this->turbulentEnergyOperatorRightHandSidePointer->setTurbulentEnergy( turbulentEnergy );
      }

      void setDensity( const MeshFunctionPointer& density )
      {
         this->momentumXOperatorRightHandSidePointer->setDensity( density );
         this->momentumYOperatorRightHandSidePointer->setDensity( density );
         this->momentumZOperatorRightHandSidePointer->setDensity( density );
         this->energyOperatorRightHandSidePointer->setDensity( density );
         this->turbulentEnergyOperatorRightHandSidePointer->setDenity( density );
      }
      
      const ContinuityOperatorRightHandSidePointer& getContinuityOperatorRightHandSide() const
      {
         return this->continuityOperatorRightHandSidePointer;
      }
      
      const MomentumXOperatorRightHandSidePointer& getMomentumXOperatorRightHandSide() const
      {
         return this->momentumXOperatorRightHandSidePointer;
      }

      const MomentumYOperatorRightHandSidePointer& getMomentumYOperatorRightHandSide() const
      {
         return this->momentumYOperatorRightHandSidePointer;
      }
      
      const MomentumZOperatorRightHandSidePointer& getMomentumZOperatorRightHandSide() const
      {
         return this->momentumZOperatorRightHandSidePointer;
      }
      
      const EnergyOperatorRightHandSidePointer& getEnergyOperatorRightHandSide() const
      {
         return this->energyOperatorRightHandSidePointer;
      }
      
      const TurbulentEnergyOperatorRightHandSidePointer& getTurbulentEnergyOperatorRightHandSide() const
      {
         return this->turbulentEnergyOperatorRightHandSidePointer;
      }

   protected:
      
      ContinuityOperatorRightHandSidePointer continuityOperatorPointer;
      MomentumXOperatorRightHandSidePointer momentumXOperatorPointer;
      MomentumYOperatorRightHandSidePointer momentumYOperatorPointer;
      MomentumZOperatorRightHandSidePointer momentumZOperatorPointer;
      EnergyOperatorRightHandSidePointer energyOperatorPointer; 
      TurbulentEnergyOperatorRightHandSidePointer turbulentEnergyOperatorPointer;  
      
      RealType dynamicalViscosity;
};

} //namespace TNL
