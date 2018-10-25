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

namespace TNL {

template< typename Mesh,
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
 
      typedef VanLeerContinuity< Mesh, Real, Index > ContinuityOperatorType;
      typedef VanLeerMomentumX< Mesh, Real, Index > MomentumXOperatorType;
      typedef VanLeerMomentumY< Mesh, Real, Index > MomentumYOperatorType;
      typedef VanLeerMomentumZ< Mesh, Real, Index > MomentumZOperatorType;
      typedef VanLeerEnergy< Mesh, Real, Index > EnergyOperatorType;

      typedef Pointers::SharedPointer< MeshFunctionType > MeshFunctionPointer;
      typedef Pointers::SharedPointer< VectorFieldType > VectorFieldPointer;
      typedef Pointers::SharedPointer< MeshType > MeshPointer;
      
      typedef Pointers::SharedPointer< ContinuityOperatorType > ContinuityOperatorPointer;
      typedef Pointers::SharedPointer< MomentumXOperatorType > MomentumXOperatorPointer;
      typedef Pointers::SharedPointer< MomentumYOperatorType > MomentumYOperatorPointer;      
      typedef Pointers::SharedPointer< MomentumZOperatorType > MomentumZOperatorPointer;      
      typedef Pointers::SharedPointer< EnergyOperatorType > EnergyOperatorPointer;

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
         this->dynamicalViscosity = parameters.getParameter< double >( prefix + "dynamical-viscosity" );
         this->momentumXOperatorPointer->setDynamicalViscosity( dynamicalViscosity );
         this->momentumYOperatorPointer->setDynamicalViscosity( dynamicalViscosity );
         this->momentumZOperatorPointer->setDynamicalViscosity( dynamicalViscosity );
         this->energyOperatorPointer->setDynamicalViscosity( dynamicalViscosity );

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
         this->continuityOperatorPointer->setGamma( gamma );
         this->momentumXOperatorPointer->setGamma( gamma );
         this->momentumYOperatorPointer->setGamma( gamma );
         this->momentumZOperatorPointer->setGamma( gamma );
         this->energyOperatorPointer->setGamma( gamma );
      }
      
      void setPressure( const MeshFunctionPointer& pressure )
      {
         this->continuityOperatorPointer->setPressure( pressure );
         this->momentumXOperatorPointer->setPressure( pressure );
         this->momentumYOperatorPointer->setPressure( pressure );
         this->momentumZOperatorPointer->setPressure( pressure );
         this->energyOperatorPointer->setPressure( pressure );
      }

      void setDensity( const MeshFunctionPointer& density )
      {
         this->momentumXOperatorPointer->setDensity( density );
         this->momentumYOperatorPointer->setDensity( density );
         this->momentumZOperatorPointer->setDensity( density );
         this->energyOperatorPointer->setDensity( density );
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
      
      RealType dynamicalViscosity;
};

} //namespace TNL
