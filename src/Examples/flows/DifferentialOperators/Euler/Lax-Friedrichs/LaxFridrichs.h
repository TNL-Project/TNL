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

namespace TNL {

template< typename Mesh,
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
 
      typedef LaxFridrichsContinuity< Mesh, Real, Index > ContinuityOperatorType;
      typedef LaxFridrichsMomentumX< Mesh, Real, Index > MomentumXOperatorType;
      typedef LaxFridrichsMomentumY< Mesh, Real, Index > MomentumYOperatorType;
      typedef LaxFridrichsMomentumZ< Mesh, Real, Index > MomentumZOperatorType;
      typedef LaxFridrichsEnergy< Mesh, Real, Index > EnergyOperatorType;

      typedef SharedPointer< MeshFunctionType > MeshFunctionPointer;
      typedef SharedPointer< VectorFieldType > VectorFieldPointer;
      typedef SharedPointer< MeshType > MeshPointer;
      
      typedef SharedPointer< ContinuityOperatorType > ContinuityOperatorPointer;
      typedef SharedPointer< MomentumXOperatorType > MomentumXOperatorPointer;
      typedef SharedPointer< MomentumYOperatorType > MomentumYOperatorPointer;      
      typedef SharedPointer< MomentumZOperatorType > MomentumZOperatorPointer;      
      typedef SharedPointer< EnergyOperatorType > EnergyOperatorPointer;

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         config.addEntry< double >( prefix + "numerical-viscosity", "Value of artificial (numerical) viscosity in the Lax-Fridrichs scheme", 1.0 );
      }
      
      LaxFridrichs()
         : artificialViscosity( 1.0 ) {}
      
      bool setup( const MeshPointer& meshPointer,
                  const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         this->artificialViscosity = parameters.getParameter< double >( prefix + "numerical-viscosity" );
         this->continuityOperatorPointer->setArtificialViscosity( artificialViscosity );
         this->momentumXOperatorPointer->setArtificialViscosity( artificialViscosity );
         this->momentumYOperatorPointer->setArtificialViscosity( artificialViscosity );
         this->momentumZOperatorPointer->setArtificialViscosity( artificialViscosity );
         this->energyOperatorPointer->setArtificialViscosity( artificialViscosity );
         
         return true;
      }
      
      void setTau( const RealType& tau )
      {
         this->continuityOperatorPointer->setTau( tau );
         this->momentumXOperatorPointer->setTau( tau );
         this->momentumYOperatorPointer->setTau( tau );
         this->momentumZOperatorPointer->setTau( tau );
         this->energyOperatorPointer->setTau( tau );
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

      void setDensity( const MeshFunctionPointer& density )
      {
      }
      
      void setVelocity( const VectorFieldPointer& velocity )
      {
         this->continuityOperatorPointer->setVelocity( velocity );
         this->momentumXOperatorPointer->setVelocity( velocity );
         this->momentumYOperatorPointer->setVelocity( velocity );
         this->momentumZOperatorPointer->setVelocity( velocity );
         this->energyOperatorPointer->setVelocity( velocity );
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

   protected:
      
      ContinuityOperatorPointer continuityOperatorPointer;
      MomentumXOperatorPointer momentumXOperatorPointer;
      MomentumYOperatorPointer momentumYOperatorPointer;
      MomentumZOperatorPointer momentumZOperatorPointer;
      EnergyOperatorPointer energyOperatorPointer;  
      
      RealType artificialViscosity;
};

} //namespace TNL
