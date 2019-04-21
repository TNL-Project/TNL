/***************************************************************************
                          VanLeerMomentumBase.h  -  description
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
class VanLeerMomentumBase
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
      

      void setTau(const Real& tau)
      {
          this->tau = tau;
      };

      void setGamma(const Real& gamma)
      {
          this->gamma = gamma;
      };
      
      void setVelocity( const VelocityFieldPointer& velocity )
      {
          this->velocity = velocity;
	  this->rightHandSide.setVelocity(velocity);
      };

      void setDensity( const MeshFunctionPointer& density )
      {
          this->density = density;
          this->rightHandSide.setDensity( density );
      };
      
      void setPressure( const MeshFunctionPointer& pressure )
      {
          this->pressure = pressure;
      };

      void setDynamicalViscosity( const RealType& dynamicalViscosity )
      {
         this->dynamicalViscosity = dynamicalViscosity;
	 this->rightHandSide.setDynamicalViscosity(dynamicalViscosity);
      } 

      void setTurbulentEnergy( const MeshFunctionPointer& turbulentEnergy )
      {
	 this->rightHandSide.setTurbulentEnergy(turbulentEnergy);
      };   

      void setTurbulentViscosity( const MeshFunctionPointer& turbulentViscosity )
      {
	 this->rightHandSide.setTurbulentViscosity(turbulentViscosity);
      } 

      RealType positiveMainMomentumFlux( const RealType& density, const RealType& velocity, const RealType& pressure ) const
      {
         const RealType& speedOfSound = std::sqrt( this->gamma * pressure / density );
         const RealType& machNumber = velocity / speedOfSound;
         if ( machNumber <= -1.0 )
            return 0;
        else if ( machNumber <= 1.0 )
            return density * speedOfSound * speedOfSound / ( 2 * this->gamma ) * ( machNumber + 1.0 ) * ( machNumber + 1.0 ) * ( 1.0 + ( this->gamma - 1.0 ) * machNumber / 2.0 );
        else 
            return density * velocity * velocity + pressure;
      };

      RealType negativeMainMomentumFlux( const RealType& density, const RealType& velocity, const RealType& pressure ) const
      {
         const RealType& speedOfSound = std::sqrt( this->gamma * pressure / density );
         const RealType& machNumber = velocity / speedOfSound;
         if ( machNumber <= -1.0 )
            return density * velocity * velocity + pressure;
        else if ( machNumber <= 1.0 )
            return - density * speedOfSound * speedOfSound / ( 2 * this->gamma ) * ( machNumber - 1.0 ) * ( machNumber - 1.0 ) * ( - 1.0 + ( this->gamma - 1.0 ) * machNumber / 2.0 );
        else 
            return 0; 
      };

      RealType positiveOtherMomentumFlux( const RealType& density, const RealType& velocity_main, const RealType& velocity_other, const RealType& pressure ) const
      {
         const RealType& speedOfSound = std::sqrt( this->gamma * pressure / density );
         const RealType& machNumber = velocity_main / speedOfSound;
         if ( machNumber <= -1.0 )
            return 0.0;
        else if ( machNumber <= 1.0 )
            return density * speedOfSound / 4.0 * ( machNumber + 1.0 ) * ( machNumber + 1.0 ) * velocity_other;
        else 
            return density * velocity_main * velocity_other;
      };

      RealType negativeOtherMomentumFlux( const RealType& density, const RealType& velocity_main, const RealType& velocity_other, const RealType& pressure ) const
      {
         const RealType& speedOfSound = std::sqrt( this->gamma * pressure / density );
         const RealType& machNumber = velocity_main / speedOfSound;
         if ( machNumber <= -1.0 )
            return density * velocity_main * velocity_other;
        else if ( machNumber <= 1.0 )
            return - density * speedOfSound / 4 * ( machNumber - 1.0 ) * ( machNumber - 1.0 ) * velocity_other;
        else 
            return 0.0;
      };

      protected:
         
         RealType tau;

         RealType gamma;
         
         VelocityFieldPointer velocity;

	 OperatorRightHandSideType rightHandSide;
         
         MeshFunctionPointer pressure;

         RealType dynamicalViscosity;

         MeshFunctionPointer density;

};

} //namespace TNL
