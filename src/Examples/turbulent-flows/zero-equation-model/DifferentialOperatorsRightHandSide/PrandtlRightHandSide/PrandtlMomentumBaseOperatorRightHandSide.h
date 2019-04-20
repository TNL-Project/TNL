/***************************************************************************
                          PrandtlMomentumRightHandSideBase.h  -  description
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
class PrandtlMomentumRightHandSideBase
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
      
      PrandtlMomentumRightHandSideBase()
       : dynamicalViscosity( 1.0 ){};

      void setVelocity( const VelocityFieldPointer& velocity )
      {
          this->velocity = velocity;
      };
      
      void setDynamicalViscosity( const RealType& dynamicalViscosity )
      {
         this->dynamicalViscosity = dynamicalViscosity;
      };

      void setDensity( const MeshFunctionType& density )
      {
         this->density = density;
      };

      void setTurbulentViscosity( const MeshFunctionPointer& turbulentViscosity )
      {
         this->turbulentViscosity = turbulentViscosity;
      };

      protected:
         
         VelocityFieldPointer velocity;

         MeshFunctionPointer turbulentViscosity;

         RealType dynamicalViscosity;
};

} //namespace TNL
