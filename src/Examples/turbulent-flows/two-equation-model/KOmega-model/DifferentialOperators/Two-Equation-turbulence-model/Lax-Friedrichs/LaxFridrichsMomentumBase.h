/***************************************************************************
                          LaxFridrichsMomentumBase.h  -  description
                             -------------------
    begin                : Feb 17, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#pragma once

namespace TNL {

template< typename Mesh,
	  typename OperatorRightHandSide,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class LaxFridrichsMomentumBase
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
      typedef OperatorRightHandSide OperatorRightHandSideType;
      
      LaxFridrichsMomentumBase()
       : artificialViscosity( 1.0 ){};

      void setTau(const Real& tau)
      {
          this->tau = tau;
      };
      
      void setVelocity( const VelocityFieldPointer& velocity )
      {
          this->velocity = velocity;
	  this->rightHandSide.setVelocity(velocity);
      };
      
      void setPressure( const MeshFunctionPointer& pressure )
      {
          this->pressure = pressure;
      };

      void setArtificialViscosity( const RealType& artificialViscosity )
      {
         this->artificialViscosity = artificialViscosity;
      }

      void setDensity( const MeshFunctionPointer& density )
      {
          this->rightHandSide.setDensity( density );
      };

      void setDynamicalViscosity( const RealType& dynamicalViscosity )
      {
	 this->rightHandSide.setDynamicalViscosity(dynamicalViscosity);
      }

      void setTurbulentViscosity( const MeshFunctionPointer& turbulentViscosity )
      {
	 this->rightHandSide.setTurbulentViscosity(turbulentViscosity);
      }

      void setTurbulentEnergy( const MeshFunctionPointer& turbulentEnergy )
      {
	 this->rightHandSide.setTurbulentEnergy(turbulentEnergy);
      }  

      protected:
         
         RealType tau;
         
         VelocityFieldPointer velocity;

	 OperatorRightHandSideType rightHandSide;
         
         MeshFunctionPointer pressure;
         
         RealType artificialViscosity;
};

} //namespace TNL
