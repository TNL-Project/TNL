/***************************************************************************
                          AUSMPlusMomentumBase.h  -  description
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
class AUSMPlusMomentumBase
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
      };

      void setTurbulentEnergy( const MeshFunctionPointer& turbulentEnergy )
      {
	 this->rightHandSide.setTurbulentEnergy(turbulentEnergy);
      };
   
      void setTurbulentViscosity( const MeshFunctionPointer& turbulentViscosity )
      {
	 this->rightHandSide.setTurbulentViscosity(turbulentViscosity);
      };

      RealType MainMomentumFlux( const RealType& LeftDensity,
                                 const RealType& RightDensity,
                                 const RealType& LeftVelocity,
                                 const RealType& RightVelocity,
                                 const RealType& LeftPressure,
                                 const RealType& RightPressure ) const
      {
         const RealType& LeftSpeedOfSound = std::sqrt( std::abs( this->gamma * LeftPressure / LeftDensity ) );
         const RealType& RightSpeedOfSound = std::sqrt( std::abs( this->gamma * RightPressure / RightDensity ) );
         const RealType& BorderSpeedOfSound = 0.5 * ( LeftSpeedOfSound + RightSpeedOfSound );
         const RealType& LeftMachNumber = LeftVelocity / BorderSpeedOfSound;
         const RealType& RightMachNumber = RightVelocity / BorderSpeedOfSound;
         RealType MachSplitingPlus = 0;
         RealType MachSplitingMinus = 0;
         RealType MachBorderPlus = 0;
         RealType MachBorderMinus = 0;
         RealType PressureSplitingPlus = 0;
         RealType PressureSplitingMinus = 0;
         RealType PressureBorder = 0;
         if ( LeftMachNumber <= -1.0 )
         {
            MachSplitingPlus = 0;
            PressureSplitingPlus = 0;
         }
         else if ( LeftMachNumber <= 1.0 )
         {
            MachSplitingPlus = 1.0 / 4.0 * ( LeftMachNumber + 1.0 ) * ( LeftMachNumber + 1.0 )
                             + 1.0 / 8.0 * ( LeftMachNumber * LeftMachNumber - 1.0 ) * ( LeftMachNumber * LeftMachNumber - 1.0 );
            PressureSplitingPlus = 1.0 / 4.0 * ( LeftMachNumber + 1.0 ) * ( LeftMachNumber + 1.0 ) * (2.0 - LeftMachNumber )
                                 + 3.0 / 16.0 * LeftMachNumber * ( LeftMachNumber * LeftMachNumber - 1.0 ) * ( LeftMachNumber * LeftMachNumber - 1.0 );
         }
         else
         { 
            MachSplitingPlus = LeftMachNumber;
            PressureSplitingPlus = 1.0;
         }
         if ( RightMachNumber <= -1.0 )
         {
            MachSplitingMinus = RightMachNumber;
            PressureSplitingMinus = 1.0;
         }
         else if ( RightMachNumber <= 1.0 )
         {
            MachSplitingMinus = - 1.0 / 4.0 * ( RightMachNumber - 1.0 ) * ( RightMachNumber - 1.0 )
                                - 1.0 / 8.0 * ( RightMachNumber * RightMachNumber - 1.0 ) * ( RightMachNumber * RightMachNumber - 1.0 );
            PressureSplitingMinus = 1.0 / 4.0 * ( RightMachNumber - 1.0 ) * ( RightMachNumber - 1.0 ) * (2.0 + RightMachNumber )
                                  - 3.0 / 16.0 * RightMachNumber * ( RightMachNumber * RightMachNumber - 1.0 ) * ( RightMachNumber * RightMachNumber - 1.0 );
         }
         else
         { 
            MachSplitingMinus = 0;
            PressureSplitingMinus = 0;
         }
         MachBorderPlus = 0.5 * ( ( MachSplitingPlus + MachSplitingMinus ) + std::abs( MachSplitingPlus + MachSplitingMinus ));
         MachBorderMinus = 0.5 * ( ( MachSplitingPlus + MachSplitingMinus ) - std::abs( MachSplitingPlus + MachSplitingMinus ));
         PressureBorder = PressureSplitingPlus * LeftPressure + PressureSplitingMinus * RightPressure;
         return BorderSpeedOfSound * ( MachBorderPlus * LeftDensity * LeftVelocity + MachBorderMinus * RightDensity * RightVelocity ) + PressureBorder;
      }

      RealType OtherMomentumFlux( const RealType& LeftDensity,
                                  const RealType& RightDensity,
                                  const RealType& LeftVelocity,
                                  const RealType& RightVelocity,
                                  const RealType& LeftOtherVelocity,
                                  const RealType& RightOtherVelocity,
                                  const RealType& LeftPressure,
                                  const RealType& RightPressure ) const
      {
         const RealType& LeftSpeedOfSound = std::sqrt( std::abs( this->gamma * LeftPressure / LeftDensity ) );
         const RealType& RightSpeedOfSound = std::sqrt( std::abs( this->gamma * RightPressure / RightDensity ) );
         const RealType& BorderSpeedOfSound = 0.5 * ( LeftSpeedOfSound + RightSpeedOfSound );
         const RealType& LeftMachNumber = LeftVelocity / BorderSpeedOfSound;
         const RealType& RightMachNumber = RightVelocity / BorderSpeedOfSound;
         RealType MachSplitingPlus = 0;
         RealType MachSplitingMinus = 0;
         RealType MachBorderPlus = 0;
         RealType MachBorderMinus = 0;
         if ( LeftMachNumber <= -1.0 )
         {
            MachSplitingPlus = 0;
         }
         else if ( LeftMachNumber <= 1.0 )
         {
            MachSplitingPlus = 1.0 / 4.0 * ( LeftMachNumber + 1.0 ) * ( LeftMachNumber + 1.0 )
                             + 1.0 / 8.0 * ( LeftMachNumber * LeftMachNumber - 1.0 ) * ( LeftMachNumber * LeftMachNumber - 1.0 );
         }
         else
         { 
            MachSplitingPlus = LeftMachNumber;
         }
         if ( RightMachNumber <= -1.0 )
         {
            MachSplitingMinus = RightMachNumber;
         }
         else if ( RightMachNumber <= 1.0 )
         {
            MachSplitingMinus = - 1.0 / 4.0 * ( RightMachNumber - 1.0 ) * ( RightMachNumber - 1.0 )
                                - 1.0 / 8.0 * ( RightMachNumber * RightMachNumber - 1.0 ) * ( RightMachNumber * RightMachNumber - 1.0 );
         }
         else
         { 
            MachSplitingMinus = 0;
         }
         MachBorderPlus = 0.5 * ( ( MachSplitingPlus + MachSplitingMinus ) + std::abs( MachSplitingPlus + MachSplitingMinus ));
         MachBorderMinus = 0.5 * ( ( MachSplitingPlus + MachSplitingMinus ) - std::abs( MachSplitingPlus + MachSplitingMinus ));
         return BorderSpeedOfSound * ( MachBorderPlus * LeftDensity * LeftOtherVelocity + MachBorderMinus * RightDensity * RightOtherVelocity );
      }

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
