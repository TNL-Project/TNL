/***************************************************************************
                          StegerWarmingTurbulentEnergy.h  -  description
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
class StegerWarmingTurbulentEnergyBase
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
         return String( "StegerWarmingTurbulentEnergy< " ) +
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
      
      void setVelocity( const VelocityFieldPointer& velocity )
      {
          this->velocity = velocity;
	  this->rightHandSide.setVelocity(velocity);
      };

      void setDensity( const MeshFunctionPointer& density )
      {
          this->density = density;
	  this->rightHandSide.setDensity(density);
      };

      void setDynamicalViscosity( const RealType& dynamicalViscosity )
      {
	 this->rightHandSide.setDynamicalViscosity(dynamicalViscosity);
      }; 

      void setTurbulentViscosity( const MeshFunctionPointer& turbulentViscosity )
      {
	 this->rightHandSide.setTurbulentViscosity(turbulentViscosity);
      }; 

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

      RealType positiveTurbulentEnergyFlux( const RealType& density, const RealType& velocity, const RealType& pressure, const RealType& turbulentEnergy ) const
      {
         const RealType& speedOfSound = std::sqrt( this->gamma * pressure / density );
         const RealType& machNumber = velocity / speedOfSound;
         if ( machNumber <= -1.0 )
            return 0.0;
        else if ( machNumber <= 0.0 )
            return turbulentEnergy * speedOfSound / ( 2 * this->gamma ) * ( machNumber + 1.0 );
        else if ( machNumber <= 1.0 )
            return turbulentEnergy * speedOfSound / ( 2 * this->gamma ) * ( ( 2.0 * this->gamma - 1.0 ) * machNumber + 1.0 );
        else 
            return turbulentEnergy * velocity;
      };

      RealType negativeTurbulentEnergyFlux( const RealType& density, const RealType& velocity, const RealType& pressure, const RealType& turbulentEnergy ) const
      {
         const RealType& speedOfSound = std::sqrt( this->gamma * pressure / density );
         const RealType& machNumber = velocity / speedOfSound;
         if ( machNumber <= -1.0 )
            return turbulentEnergy * velocity;
        else if ( machNumber <= 0.0 )
            return turbulentEnergy * speedOfSound / ( 2 * this->gamma ) * ( ( 2.0 * this->gamma - 1.0 ) * machNumber - 1.0 );
        else if ( machNumber <= 1.0 )
            return turbulentEnergy * speedOfSound / ( 2 * this->gamma ) * ( machNumber - 1.0 );
        else 
            return 0.0;
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
class StegerWarmingTurbulentEnergy
{
};



template< typename MeshReal,
          typename Device,
          typename MeshIndex,
	  typename OperatorRightHandSide,
          typename Real,
          typename Index >
class StegerWarmingTurbulentEnergy< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, OperatorRightHandSide, Real, Index >
   : public StegerWarmingTurbulentEnergyBase< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, OperatorRightHandSide, Real, Index >
{
   public:
      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef StegerWarmingTurbulentEnergyBase< MeshType, OperatorRightHandSide, Real, Index > BaseType;
      
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
         const IndexType& east   = neighborEntities.template getEntityIndex< 1 >(); 
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

         return -hxInverse * (
                                   this->positiveTurbulentEnergyFlux( density_center, velocity_x_center, pressure_center, u[ center ] )
                                -  this->positiveTurbulentEnergyFlux( density_west  , velocity_x_west  , pressure_west  , u[ west   ] )
                                -  this->negativeTurbulentEnergyFlux( density_center, velocity_x_center, pressure_center, u[ center ] )
                                +  this->negativeTurbulentEnergyFlux( density_east  , velocity_x_east  , pressure_east  , u[ east   ] )
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
class StegerWarmingTurbulentEnergy< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, OperatorRightHandSide, Real, Index >
   : public StegerWarmingTurbulentEnergyBase< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, OperatorRightHandSide, Real, Index >
{
   public:
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef StegerWarmingTurbulentEnergyBase< MeshType, OperatorRightHandSide, Real, Index > BaseType;
      
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
         const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1,  0 >(); 
         const RealType& hyInverse = entity.getMesh().template getSpaceStepsProducts<  0, -1 >(); 

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
         
         return -hxInverse * (
                                   this->positiveTurbulentEnergyFlux( density_center, velocity_x_center, pressure_center, u[ center ] )
                                -  this->positiveTurbulentEnergyFlux( density_west  , velocity_x_west  , pressure_west  , u[ west   ] )
                                -  this->negativeTurbulentEnergyFlux( density_center, velocity_x_center, pressure_center, u[ center ] )
                                +  this->negativeTurbulentEnergyFlux( density_east  , velocity_x_east  , pressure_east  , u[ east   ] )
                             )
                -hyInverse * (
                                   this->positiveTurbulentEnergyFlux( density_center, velocity_y_center, pressure_center, u[ center ] )
                                -  this->positiveTurbulentEnergyFlux( density_south , velocity_y_south , pressure_south , u[ south  ] )
                                -  this->negativeTurbulentEnergyFlux( density_center, velocity_y_center, pressure_center, u[ center ] )
                                +  this->negativeTurbulentEnergyFlux( density_north , velocity_y_north , pressure_north , u[ north  ] )
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
class StegerWarmingTurbulentEnergy< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, OperatorRightHandSide, Real, Index >
   : public StegerWarmingTurbulentEnergyBase< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, OperatorRightHandSide, Real, Index >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef StegerWarmingTurbulentEnergyBase< MeshType, OperatorRightHandSide, Real, Index > BaseType;
      
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
         
         return -hxInverse * (
                                   this->positiveTurbulentEnergyFlux( density_center, velocity_x_center, pressure_center, u[ center ] )
                                -  this->positiveTurbulentEnergyFlux( density_west  , velocity_x_west  , pressure_west  , u[ west   ] )
                                -  this->negativeTurbulentEnergyFlux( density_center, velocity_x_center, pressure_center, u[ center ] )
                                +  this->negativeTurbulentEnergyFlux( density_east  , velocity_x_east  , pressure_east  , u[ east   ] )
                             )
                -hyInverse * (
                                   this->positiveTurbulentEnergyFlux( density_center, velocity_y_center, pressure_center, u[ center ] )
                                -  this->positiveTurbulentEnergyFlux( density_south , velocity_y_south , pressure_south , u[ south  ] )
                                -  this->negativeTurbulentEnergyFlux( density_center, velocity_y_center, pressure_center, u[ center ] )
                                +  this->negativeTurbulentEnergyFlux( density_north , velocity_y_north , pressure_north , u[ north  ] )
                             )
                -hzInverse * (
                                   this->positiveTurbulentEnergyFlux( density_center, velocity_z_center, pressure_center, u[ center ] )
                                -  this->positiveTurbulentEnergyFlux( density_down  , velocity_z_down  , pressure_down  , u[ down   ] )
                                -  this->negativeTurbulentEnergyFlux( density_center, velocity_z_center, pressure_center, u[ center ] )
                                +  this->negativeTurbulentEnergyFlux( density_up    , velocity_z_up    , pressure_up    , u[ up     ] )
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
