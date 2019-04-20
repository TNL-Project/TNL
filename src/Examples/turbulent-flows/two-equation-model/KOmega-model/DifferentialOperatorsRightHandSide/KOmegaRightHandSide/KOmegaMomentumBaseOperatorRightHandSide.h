/***************************************************************************
                          KOmegaMomentumRightHandSideBase.h  -  description
                             -------------------
    begin                : Feb 17, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#pragma once

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class KOmegaMomentumRightHandSideBase
{
   public:
      
      typedef Real RealType;
      typedef Index IndexType;
      typedef Mesh MeshType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      static const int Dimensions = MeshType::getMeshDimension();
      typedef Functions::VectorField< Dimensions, MeshFunctionType > VelocityFieldType;
      typedef Pointers::SharedPointer< MeshFunctionType > MeshFunctionPointer;
      typedef Pointers::SharedPointer< VelocityFieldType > VelocityFieldPointer;
      
      KOmegaMomentumRightHandSideBase()
       : dynamicalViscosity( 1.0 ){};

      void setVelocity( const VelocityFieldPointer& velocity )
      {
          this->velocity = velocity;
      }
      
      void setDynamicalViscosity( const RealType& dynamicalViscosity )
      {
         this->dynamicalViscosity = dynamicalViscosity;
      }

      void setTurbulentViscosity( const MeshFunctionPointer& turbulentViscosity )
      {
         this->turbulentViscosity = turbulentViscosity;
      }
    
      void setTurbulentEnergy( const MeshFunctionPointer& turbulentEnergy )
      {
          this->turbulentEnergy = turbulentEnergy;
      }
    
      void setDensity( const MeshFunctionPointer& density )
      {
          this->density = density;
      }

      protected:

         MeshFunctionPointer turbulentViscosity;

         MeshFunctionPointer turbulentEnergy;

         MeshFunctionPointer density;
         
         VelocityFieldPointer velocity;

         RealType dynamicalViscosity;
};

} //namespace TNL
