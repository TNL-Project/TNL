/***************************************************************************
                          AUSMPlusTurbulentEnergy.h  -  description
                             -------------------
    begin                : Feb 17, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Functions/VectorField.h>
#include <TNL/Pointers/SharedPointer.h>

namespace TNL {

   
template< typename Mesh,
          typename OperatorRightHandSide,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class AUSMPlusTurbulentEnergyBase
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

      static String getType()
      {
         return String( "AUSMPlusTurbulentEnergy< " ) +
             MeshType::getType() + ", " +
             TNL::getType< Real >() + ", " +
             TNL::getType< Index >() + " >"; 
      }

      void setTau(const Real& tau)
      {
          this->tau = tau;
      };

      void setGamma(const Real& gamma)
      {
          this->gamma = gamma;
      };
      
      void setPressure( const MeshFunctionPointer& pressure )
      {
          this->pressure = pressure;
      };

      void setDensity( const MeshFunctionPointer& density )
      {
          this->density = density;
	  this->rightHandSide.setDensity(density);
      };
      
      void setVelocity( const VelocityFieldPointer& velocity )
      {
          this->velocity = velocity;
	  this->rightHandSide.setVelocity(velocity);
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
      };

      void setCharacteristicLength( RealType& characteristicLength )
      {
         this->rightHandSide.setCharacteristicLength( characteristicLength );
      }

      void setSigmaK( RealType& sigmaK )
      {
         this->rightHandSide.setSigmaK( sigmaK );
      }

      void setViscosityConstant( RealType& ViscosityConstant )
      {
         this->rightHandSide.setViscosityConstant( ViscosityConstant );
      } 

      RealType TurbulentEnergyFlux( const RealType& LeftDensity,
                                    const RealType& RightDensity,
                                    const RealType& LeftVelocity,
                                    const RealType& RightVelocity,
                                    const RealType& LeftPressure,
                                    const RealType& RightPressure,
                                    const RealType& LeftTurbulentEnergy,
                                    const RealType& RightTurbulentEnergy ) const
      {
         const RealType& LeftSpeedOfSound = std::sqrt( this->gamma * LeftPressure / LeftDensity );
         const RealType& RightSpeedOfSound = std::sqrt( this->gamma * RightPressure / RightDensity );
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
            MachSplitingPlus = 1.0 / 2.0 * ( LeftMachNumber + 1.0 ) * ( LeftMachNumber + 1.0 )
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
            MachSplitingMinus = - 1.0 / 2.0 * ( RightMachNumber - 1.0 ) * ( RightMachNumber - 1.0 )
                                - 1.0 / 8.0 * ( RightMachNumber * RightMachNumber - 1.0 ) * ( RightMachNumber * RightMachNumber - 1.0 );
         }
         else
         { 
            MachSplitingMinus = 0;
         }
         MachBorderPlus = 0.5 * ( ( MachSplitingPlus + MachSplitingMinus ) + std::abs( MachSplitingPlus + MachSplitingMinus ));
         MachBorderMinus = 0.5 * ( ( MachSplitingPlus + MachSplitingMinus ) - std::abs( MachSplitingPlus + MachSplitingMinus ));
         return BorderSpeedOfSound * ( MachBorderPlus * LeftTurbulentEnergy + MachBorderMinus * RightTurbulentEnergy );
      };

      protected:
         
         RealType tau;

         RealType gamma;
         
         VelocityFieldPointer velocity;

	 OperatorRightHandSideType rightHandSide;

         MeshFunctionPointer pressure;

         MeshFunctionPointer density;
         
};

   
template< typename Mesh,
	  typename OperatorRightHandSide,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class AUSMPlusTurbulentEnergy
{
};



template< typename MeshReal,
          typename Device,
          typename MeshIndex,
	  typename OperatorRightHandSide,
          typename Real,
          typename Index >
class AUSMPlusTurbulentEnergy< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, OperatorRightHandSide, Real, Index >
   : public AUSMPlusTurbulentEnergyBase< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, OperatorRightHandSide, Real, Index >
{
   public:
      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef AUSMPlusTurbulentEnergyBase< MeshType, OperatorRightHandSide, Real, Index > BaseType;
      
      using typename BaseType::RealType;
      using typename BaseType::IndexType;
      using typename BaseType::DeviceType;
      using typename BaseType::CoordinatesType;
      using typename BaseType::MeshFunctionType;
      using typename BaseType::VelocityFieldType;
      using typename BaseType::VelocityFieldPointer;
      using BaseType::Dimensions;

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         static_assert( MeshEntity::getEntityDimension() == 1, "Wrong mesh entity dimensions." ); 
         static_assert( MeshFunction::getEntitiesDimension() == 1, "Wrong preimage function" ); 
         const typename MeshEntity::template NeighborEntities< 1 >& neighborEntities = entity.getNeighborEntities(); 

         const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1 >();
 
         const IndexType& center = entity.getIndex(); 
         const IndexType& east   = neighborEntities.template getEntityIndex<  1 >(); 
         const IndexType& west   = neighborEntities.template getEntityIndex< -1 >();

         const RealType& pressure_center = this->pressure.template getData< DeviceType >()[ center ];
         const RealType& pressure_west   = this->pressure.template getData< DeviceType >()[ west ];
         const RealType& pressure_east   = this->pressure.template getData< DeviceType >()[ east ];

         const RealType& density_center = this->density.template getData< DeviceType >()[ center ];
         const RealType& density_west   = this->density.template getData< DeviceType >()[ west ];
         const RealType& density_east   = this->density.template getData< DeviceType >()[ east ];

         const RealType& velocity_x_center = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_x_west   = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_x_east   = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ east ];

         return  hxInverse * (
                                   this->TurbulentEnergyFlux( density_west  , density_center, velocity_x_west  , velocity_x_center, pressure_west  , pressure_center, u[ west   ], u[ center ] )
                                -  this->TurbulentEnergyFlux( density_center, density_east  , velocity_x_center, velocity_x_east  , pressure_center, pressure_east  , u[ center ], u[ east ]   )
                             )
               +
                 this->rightHandSide(u, entity, time);
      }

      /*template< typename MeshEntity >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const MeshEntity& entity ) const;

      template< typename MeshEntity, typename Vector, typename MatrixRow >
      __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               const MeshFunctionType& u,
                               Vector& b,
                               MatrixRow& matrixRow ) const;*/
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
	  typename OperatorRightHandSide,
          typename Real,
          typename Index >
class AUSMPlusTurbulentEnergy< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, OperatorRightHandSide, Real, Index >
   : public AUSMPlusTurbulentEnergyBase< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, OperatorRightHandSide, Real, Index >
{
   public:
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef AUSMPlusTurbulentEnergyBase< MeshType, OperatorRightHandSide, Real, Index > BaseType;
      
      using typename BaseType::RealType;
      using typename BaseType::IndexType;
      using typename BaseType::DeviceType;
      using typename BaseType::CoordinatesType;
      using typename BaseType::MeshFunctionType;
      using typename BaseType::VelocityFieldType;
      using typename BaseType::VelocityFieldPointer;
      using BaseType::Dimensions;      

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         static_assert( MeshEntity::getEntityDimension() == 2, "Wrong mesh entity dimensions." ); 
         static_assert( MeshFunction::getEntitiesDimension() == 2, "Wrong preimage function" ); 
         const typename MeshEntity::template NeighborEntities< 2 >& neighborEntities = entity.getNeighborEntities(); 

         //rho
         const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1, 0 >(); 
         const RealType& hyInverse = entity.getMesh().template getSpaceStepsProducts< 0, -1 >(); 

         const IndexType& center = entity.getIndex(); 
         const IndexType& east   = neighborEntities.template getEntityIndex<  1,  0 >(); 
         const IndexType& west   = neighborEntities.template getEntityIndex< -1,  0 >(); 
         const IndexType& north  = neighborEntities.template getEntityIndex<  0,  1 >(); 
         const IndexType& south  = neighborEntities.template getEntityIndex<  0, -1 >();

         const RealType& pressure_center = this->pressure.template getData< DeviceType >()[ center ];
         const RealType& pressure_west   = this->pressure.template getData< DeviceType >()[ west ];
         const RealType& pressure_east   = this->pressure.template getData< DeviceType >()[ east ];
         const RealType& pressure_north  = this->pressure.template getData< DeviceType >()[ north ];
         const RealType& pressure_south  = this->pressure.template getData< DeviceType >()[ south ];

         const RealType& density_center = this->density.template getData< DeviceType >()[ center ];
         const RealType& density_west   = this->density.template getData< DeviceType >()[ west ];
         const RealType& density_east   = this->density.template getData< DeviceType >()[ east ];
         const RealType& density_north  = this->density.template getData< DeviceType >()[ north ];
         const RealType& density_south  = this->density.template getData< DeviceType >()[ south ];

         const RealType& velocity_x_center = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_x_west   = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_x_east   = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ east ];

         const RealType& velocity_y_center = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_y_north  = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ north ];
         const RealType& velocity_y_south  = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ south ];
         
         return  hxInverse * (
                                   this->TurbulentEnergyFlux( density_west   , density_center , velocity_x_west  , velocity_x_center, pressure_west  , pressure_center, u[ west   ], u[ center ] )
                                -  this->TurbulentEnergyFlux( density_center , density_east   , velocity_x_center, velocity_x_east  , pressure_center, pressure_east  , u[ center ], u[ east ]   )
                             )
               + hyInverse * (
                                   this->TurbulentEnergyFlux( density_north , density_center, velocity_y_north , velocity_y_center, pressure_north , pressure_center, u[ north  ], u[ center ] )
                                -  this->TurbulentEnergyFlux( density_center, density_south , velocity_y_center, velocity_y_south , pressure_center, pressure_south , u[ center ], u[ south ]  )
                             )
               +
                 this->rightHandSide(u, entity, time); 
      }

      /*template< typename MeshEntity >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const MeshEntity& entity ) const;

      template< typename MeshEntity, typename Vector, typename MatrixRow >
      __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               const MeshFunctionType& u,
                               Vector& b,
                               MatrixRow& matrixRow ) const;*/
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
	  typename OperatorRightHandSide,
          typename Real,
          typename Index >
class AUSMPlusTurbulentEnergy< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, OperatorRightHandSide, Real, Index >
   : public AUSMPlusTurbulentEnergyBase< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, OperatorRightHandSide, Real, Index >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef AUSMPlusTurbulentEnergyBase< MeshType, OperatorRightHandSide, Real, Index > BaseType;
      
      using typename BaseType::RealType;
      using typename BaseType::IndexType;
      using typename BaseType::DeviceType;
      using typename BaseType::CoordinatesType;
      using typename BaseType::MeshFunctionType;
      using typename BaseType::VelocityFieldType;
      using typename BaseType::VelocityFieldPointer;
      using BaseType::Dimensions;

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         static_assert( MeshEntity::getEntityDimension() == 3, "Wrong mesh entity dimensions." ); 
         static_assert( MeshFunction::getEntitiesDimension() == 3, "Wrong preimage function" ); 
         const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities(); 

         //rho
         const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1,  0,  0 >(); 
         const RealType& hyInverse = entity.getMesh().template getSpaceStepsProducts<  0, -1,  0 >(); 
         const RealType& hzInverse = entity.getMesh().template getSpaceStepsProducts<  0,  0, -1 >();
 
         const IndexType& center = entity.getIndex(); 
         const IndexType& east   = neighborEntities.template getEntityIndex<  1,  0,  0 >(); 
         const IndexType& west   = neighborEntities.template getEntityIndex< -1,  0,  0 >(); 
         const IndexType& north  = neighborEntities.template getEntityIndex<  0,  1,  0 >(); 
         const IndexType& south  = neighborEntities.template getEntityIndex<  0, -1,  0 >();
         const IndexType& up     = neighborEntities.template getEntityIndex<  0,  0,  1 >(); 
         const IndexType& down   = neighborEntities.template getEntityIndex<  0,  0, -1 >();

         const RealType& pressure_center = this->pressure.template getData< DeviceType >()[ center ];
         const RealType& pressure_west   = this->pressure.template getData< DeviceType >()[ west ];
         const RealType& pressure_east   = this->pressure.template getData< DeviceType >()[ east ];
         const RealType& pressure_north  = this->pressure.template getData< DeviceType >()[ north ];
         const RealType& pressure_south  = this->pressure.template getData< DeviceType >()[ south ];
         const RealType& pressure_up     = this->pressure.template getData< DeviceType >()[ up ];
         const RealType& pressure_down   = this->pressure.template getData< DeviceType >()[ down ];
         
         const RealType& density_center = this->density.template getData< DeviceType >()[ center ];
         const RealType& density_west   = this->density.template getData< DeviceType >()[ west ];
         const RealType& density_east   = this->density.template getData< DeviceType >()[ east ];
         const RealType& density_north  = this->density.template getData< DeviceType >()[ north ];
         const RealType& density_south  = this->density.template getData< DeviceType >()[ south ];
         const RealType& density_up     = this->density.template getData< DeviceType >()[ up ];
         const RealType& density_down   = this->density.template getData< DeviceType >()[ down ];
         
         const RealType& velocity_x_center = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_x_west   = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_x_east   = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ east ];

         const RealType& velocity_y_center = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_y_north  = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ north ];
         const RealType& velocity_y_south  = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ south ];

         const RealType& velocity_z_center = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_z_up     = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ up ];
         const RealType& velocity_z_down   = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ down ];
         
         return  hxInverse * (
                                   this->TurbulentEnergyFlux( density_west  , density_center, velocity_x_west  , velocity_x_center, pressure_west  , pressure_center, u[ west ]  , u[ center ] )
                                -  this->TurbulentEnergyFlux( density_center, density_east  , velocity_x_center, velocity_x_east  , pressure_center, pressure_east  , u[ center ], u[ east ]   )
                             )
               + hyInverse * (
                                   this->TurbulentEnergyFlux( density_north , density_center, velocity_y_north , velocity_y_center, pressure_north , pressure_center, u[ north ] , u[ center ] )
                                -  this->TurbulentEnergyFlux( density_center, density_south , velocity_y_center, velocity_y_south , pressure_center, pressure_south , u[ center ], u[ south ]  )
                             )
               + hzInverse * (
                                   this->TurbulentEnergyFlux( density_up    , density_center, velocity_z_up    , velocity_z_center, pressure_up    , pressure_center, u[ up ]    , u[ center ])
                                -  this->TurbulentEnergyFlux( density_center, density_down  , velocity_z_center, velocity_z_down  , pressure_center, pressure_down  , u[ center ], u[ down ]  )
                             )
               +
                 this->rightHandSide(u, entity, time);
         
      }

      /*template< typename MeshEntity >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const MeshEntity& entity ) const;

      template< typename MeshEntity, typename Vector, typename MatrixRow >
      __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               const MeshFunctionType& u,
                               Vector& b,
                               MatrixRow& matrixRow ) const;*/
};


} //namespace TNL
