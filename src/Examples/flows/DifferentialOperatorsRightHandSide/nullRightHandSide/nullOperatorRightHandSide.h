/***************************************************************************
                          nullOpratorRightHandSide.h  -  description
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

#include "nullContinuityOperatorRightHandSide.h"
#include "nullEnergyOperatorRightHandSide.h"
#include "nullMomentumXOperatorRightHandSide.h"
#include "nullMomentumYOperatorRightHandSide.h"
#include "nullMomentumZOperatorRightHandSide.h"

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class NullOperatorRightHandSide
{
   public:
      typedef Mesh MeshType;
      typedef Real RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< Mesh > MeshFunctionType;
      static const int Dimensions = Mesh::getMeshDimension();
      typedef Functions::VectorField< Dimensions, MeshFunctionType > VectorFieldType;
 
      typedef nullContinuityRightHandSide< Mesh, Real, Index > ContinuityOperatorRightHandSideType;
      typedef nullMomentumXRightHandSide< Mesh, Real, Index > MomentumXOperatorRightHandSideType;
      typedef nullMomentumYRightHandSide< Mesh, Real, Index > MomentumYOperatorRightHandSideType;
      typedef nullMomentumZRightHandSide< Mesh, Real, Index > MomentumZOperatorRightHandSideType;
      typedef nullEnergyRightHandSide< Mesh, Real, Index > EnergyOperatorRightHandSideType;

      typedef Pointers::SharedPointer< MeshFunctionType > MeshFunctionPointer;
      typedef Pointers::SharedPointer< VectorFieldType > VectorFieldPointer;
      typedef Pointers::SharedPointer< MeshType > MeshPointer;
      
      typedef Pointers::SharedPointer< ContinuityOperatorRightHandSideType > ContinuityOperatorRightHandSidePointer;
      typedef Pointers::SharedPointer< MomentumXOperatorRightHandSideType > MomentumXOperatorRightHandSidePointer;
      typedef Pointers::SharedPointer< MomentumYOperatorRightHandSideType > MomentumYOperatorRightHandSidePointer;      
      typedef Pointers::SharedPointer< MomentumZOperatorRightHandSideType > MomentumZOperatorRightHandSidePointer;      
      typedef Pointers::SharedPointer< EnergyOperatorRightHandSideType > EnergyOperatorRightHandSidePointer;

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         config.addEntry< double >( prefix + "dynamical-viscosity", "Value of dynamical (real) viscosity in the Navier-Stokes equation", 1.0 );
      }
      
      NullOperatorRightHandSide()
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
         return true;
      }
      
      void setVelocity( const VectorFieldPointer& velocity )
      {
         this->continuityOperatorRightHandSidePointer->setVelocity( velocity );
         this->momentumXOperatorRightHandSidePointer->setVelocity( velocity );
         this->momentumYOperatorRightHandSidePointer->setVelocity( velocity );
         this->momentumZOperatorRightHandSidePointer->setVelocity( velocity );
         this->energyOperatorRightHandSidePointer->setVelocity( velocity );
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

   protected:
      
      ContinuityOperatorRightHandSidePointer continuityOperatorPointer;
      MomentumXOperatorRightHandSidePointer momentumXOperatorPointer;
      MomentumYOperatorRightHandSidePointer momentumYOperatorPointer;
      MomentumZOperatorRightHandSidePointer momentumZOperatorPointer;
      EnergyOperatorRightHandSidePointer energyOperatorPointer;  
      
      RealType dynamicalViscosity;
};

} //namespace TNL
