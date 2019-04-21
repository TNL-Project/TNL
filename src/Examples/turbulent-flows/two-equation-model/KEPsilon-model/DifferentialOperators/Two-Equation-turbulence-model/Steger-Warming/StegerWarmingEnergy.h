/***************************************************************************
                          StegerWarmingEnergy.h  -  description
                             -------------------
    begin                : Feb 17, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>

namespace TNL {
   
template< typename Mesh,
	  typename OperatorRightHandSide,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class StegerWarmingEnergyBase
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
         return String( "StegerWarmingEnergy< " ) +
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
      
      void setVelocity( const VelocityFieldPointer& velocity )
      {
          this->velocity = velocity;
	  this->rightHandSide.setVelocity(velocity);
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
class StegerWarmingEnergy
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
	  typename OperatorRightHandSide,
          typename Real,
          typename Index >
class StegerWarmingEnergy< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, OperatorRightHandSide, Real, Index >
   : public StegerWarmingEnergyBase< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, OperatorRightHandSide, Real, Index >
{
   public:

      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef StegerWarmingEnergyBase< MeshType, OperatorRightHandSide, Real, Index > BaseType;
      
      using typename BaseType::RealType;
      using typename BaseType::IndexType;
      using typename BaseType::DeviceType;
      using typename BaseType::CoordinatesType;
      using typename BaseType::MeshFunctionType;
      using typename BaseType::MeshFunctionPointer;
      using typename BaseType::VelocityFieldType;
      using typename BaseType::VelocityFieldPointer;
      using BaseType::Dimensions;      

      RealType positiveEnergyFlux( const RealType& density, const RealType& velocity_main, const RealType& pressure ) const
      {
         const RealType& speedOfSound = std::sqrt( this->gamma * pressure / density );
         const RealType& machNumber = velocity_main / speedOfSound;
         if ( machNumber <= -1.0 )
            return 0.0;
        else if ( machNumber <= 0.0 )
            return density * speedOfSound * speedOfSound * speedOfSound / ( 2.0 * this->gamma ) 
                 * ( 
                     ( ( machNumber + 1.0 ) * 0.5 *  ( machNumber * machNumber / ( speedOfSound * speedOfSound ) ) )
                   + ( machNumber * ( machNumber + 1.0 ) )
                   + ( ( machNumber + 1.0 ) / ( this->gamma - 1.0 ) )
                   );
        else if ( machNumber <= 1.0 )
            return density * speedOfSound * speedOfSound * speedOfSound / ( 2.0 * this->gamma ) 
                 * ( 
                     ( ( ( 2.0 * this->gamma - 1.0 ) * machNumber + 1.0 ) * 0.5 * ( machNumber * machNumber / ( speedOfSound * speedOfSound ) ) )
                   + ( machNumber * ( machNumber + 1.0 ) )
                   + ( ( machNumber + 1.0 ) / ( this->gamma - 1.0 ) )
                   );
        else   
            return velocity_main * ( pressure + pressure / ( this->gamma - 1.0 ) + 0.5 * density * ( velocity_main * velocity_main ) );
      };
      
      RealType negativeEnergyFlux( const RealType& density, const RealType& velocity_main, const RealType& pressure ) const
      {
         const RealType& speedOfSound = std::sqrt( this->gamma * pressure / density );
         const RealType& machNumber = velocity_main / speedOfSound;
         if ( machNumber <= -1.0 )
            return velocity_main * ( pressure + pressure / ( this->gamma - 1.0 ) + 0.5 * density * ( velocity_main * velocity_main ) );
        else if ( machNumber <= 0.0 )
            return density * speedOfSound * speedOfSound * speedOfSound / ( 2.0 * this->gamma ) 
                 * ( 
                     ( ( ( 2.0 * this->gamma - 1.0 ) * machNumber - 1.0 ) * 0.5 * ( machNumber * machNumber / ( speedOfSound * speedOfSound ) ) )
                   - ( machNumber * ( machNumber - 1.0 ) )
                   + ( ( machNumber - 1.0 ) / ( this->gamma - 1.0 ) )
                   );
        else if ( machNumber <= 1.0 )
            return density * speedOfSound * speedOfSound * speedOfSound / ( 2.0 * this->gamma ) 
                 * ( 
                     ( ( machNumber - 1.0 ) * 0.5 * ( machNumber * machNumber / ( speedOfSound * speedOfSound ) ) )
                   - ( machNumber * ( machNumber - 1.0 ) )
                   + ( ( machNumber - 1.0 ) / ( this->gamma - 1.0 ) )
                   );
        else 
            return 0.0;
      };      

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
         const RealType& velocity_x_east   = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_x_west   = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ west ];

         return -hxInverse * ( 
                                   this->positiveEnergyFlux( density_center, velocity_x_center, pressure_center)
                                 - this->positiveEnergyFlux( density_west  , velocity_x_west  , pressure_west  )
                                 - this->negativeEnergyFlux( density_center, velocity_x_center, pressure_center)
                                 + this->negativeEnergyFlux( density_east  , velocity_x_east  , pressure_east  ) 
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
class StegerWarmingEnergy< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, OperatorRightHandSide, Real, Index >
   : public StegerWarmingEnergyBase< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, OperatorRightHandSide, Real, Index >
{
   public:
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef StegerWarmingEnergyBase< MeshType, OperatorRightHandSide, Real, Index > BaseType;
      
      using typename BaseType::RealType;
      using typename BaseType::IndexType;
      using typename BaseType::DeviceType;
      using typename BaseType::CoordinatesType;
      using typename BaseType::MeshFunctionType;
      using typename BaseType::MeshFunctionPointer;
      using typename BaseType::VelocityFieldType;
      using typename BaseType::VelocityFieldPointer;
      using BaseType::Dimensions;

      RealType positiveEnergyFlux( const RealType& density, const RealType& velocity_main, const RealType& velocity_other1, const RealType& pressure ) const
      {
         const RealType& speedOfSound = std::sqrt( this->gamma * pressure / density );
         const RealType& machNumber = velocity_main / speedOfSound;
         if ( machNumber <= -1.0 )
            return 0.0;
        else if ( machNumber <= 0.0 )
            return density * speedOfSound * speedOfSound * speedOfSound / ( 2.0 * this->gamma ) 
                 * ( 
                     ( ( machNumber + 1.0 ) * 0.5 *  ( machNumber * machNumber + ( velocity_other1 * velocity_other1 ) / ( speedOfSound * speedOfSound ) ) )
                   + ( machNumber * ( machNumber + 1.0 ) )
                   + ( ( machNumber + 1.0 ) / ( this->gamma - 1.0 ) )
                   );
        else if ( machNumber <= 1.0 )
            return density * speedOfSound * speedOfSound * speedOfSound / ( 2.0 * this->gamma ) 
                 * ( 
                     ( ( ( 2.0 * this->gamma - 1.0 ) * machNumber + 1.0 ) * 0.5 * ( machNumber * machNumber + ( velocity_other1 * velocity_other1 ) / ( speedOfSound * speedOfSound ) ) )
                   + ( machNumber * ( machNumber + 1.0 ) )
                   + ( ( machNumber + 1.0 ) / ( this->gamma - 1.0 ) )
                   );
        else   
            return velocity_main * ( pressure + pressure / ( this->gamma - 1.0 ) + 0.5 * density * ( velocity_main * velocity_main + velocity_other1 * velocity_other1 ) );
      };

      RealType negativeEnergyFlux( const RealType& density, const RealType& velocity_main, const RealType& velocity_other1, const RealType& pressure ) const
      {
         const RealType& speedOfSound = std::sqrt( this->gamma * pressure / density );
         const RealType& machNumber = velocity_main / speedOfSound;
         if ( machNumber <= -1.0 )
            return velocity_main * ( pressure + pressure / ( this->gamma - 1.0 ) + 0.5 * density * ( velocity_main * velocity_main + velocity_other1 * velocity_other1 ) );
        else if ( machNumber <= 0.0 )
            return density * speedOfSound * speedOfSound * speedOfSound / ( 2.0 * this->gamma ) 
                 * ( 
                     ( ( ( 2.0 * this->gamma - 1.0 ) * machNumber - 1.0 ) * 0.5 * ( machNumber * machNumber + ( velocity_other1 * velocity_other1 ) / ( speedOfSound * speedOfSound ) ) )
                   - ( machNumber * ( machNumber - 1.0 ) )
                   + ( ( machNumber - 1.0 ) / ( this->gamma - 1.0 ) )
                   );
        else if ( machNumber <= 1.0 )
            return density * speedOfSound * speedOfSound * speedOfSound / ( 2.0 * this->gamma ) 
                 * ( 
                     ( ( machNumber - 1.0 ) * 0.5 * ( machNumber * machNumber + ( velocity_other1 * velocity_other1 ) / ( speedOfSound * speedOfSound ) ) )
                   - ( machNumber * ( machNumber - 1.0 ) )
                   + ( ( machNumber - 1.0 ) / ( this->gamma - 1.0 ) )
                   );
        else 
            return 0.0;
      };  

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         static_assert( MeshEntity::getEntityDimension() == 2, "Wrong mesh entity dimensions." ); 
         static_assert( MeshFunction::getEntitiesDimension() == 2, "Wrong preimage function" ); 
         const typename MeshEntity::template NeighborEntities< 2 >& neighborEntities = entity.getNeighborEntities(); 
 
         const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1,  0 >(); 
         const RealType& hyInverse = entity.getMesh().template getSpaceStepsProducts<  0, -1 >(); 

         const IndexType& center    = entity.getIndex(); 
         const IndexType& east      = neighborEntities.template getEntityIndex<  1,  0 >(); 
         const IndexType& west      = neighborEntities.template getEntityIndex< -1,  0 >(); 
         const IndexType& north     = neighborEntities.template getEntityIndex<  0,  1 >(); 
         const IndexType& south     = neighborEntities.template getEntityIndex<  0, -1 >();

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

         const RealType& velocity_x_center    = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_x_east      = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_x_west      = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_x_south     = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ south ];
         const RealType& velocity_x_north     = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ north ];      

         const RealType& velocity_y_center    = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_y_east      = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_y_west      = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_y_north     = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ north ];
         const RealType& velocity_y_south     = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ south ];     
         
         return -hxInverse * ( 
                                   this->positiveEnergyFlux( density_center, velocity_x_center, velocity_y_center, pressure_center)
                                 - this->positiveEnergyFlux( density_west  , velocity_x_west  , velocity_y_west  , pressure_west  )
                                 - this->negativeEnergyFlux( density_center, velocity_x_center, velocity_y_center, pressure_center)
                                 + this->negativeEnergyFlux( density_east  , velocity_x_east  , velocity_y_east  , pressure_east  ) 
                             ) 
                -hyInverse * ( 
                                   this->positiveEnergyFlux( density_center, velocity_y_center, velocity_x_center, pressure_center)
                                 - this->positiveEnergyFlux( density_south , velocity_y_south , velocity_x_south , pressure_south )
                                 - this->negativeEnergyFlux( density_center, velocity_y_center, velocity_x_center, pressure_center)
                                 + this->negativeEnergyFlux( density_north , velocity_y_north , velocity_x_north , pressure_north ) 
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
class StegerWarmingEnergy< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, OperatorRightHandSide, Real, Index >
   : public StegerWarmingEnergyBase< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, OperatorRightHandSide, Real, Index >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef StegerWarmingEnergyBase< MeshType, OperatorRightHandSide, Real, Index > BaseType;
      
      using typename BaseType::RealType;
      using typename BaseType::IndexType;
      using typename BaseType::DeviceType;
      using typename BaseType::CoordinatesType;
      using typename BaseType::MeshFunctionType;
      using typename BaseType::MeshFunctionPointer;
      using typename BaseType::VelocityFieldType;
      using typename BaseType::VelocityFieldPointer;
      using BaseType::Dimensions;      

      RealType positiveEnergyFlux( const RealType& density, const RealType& velocity_main, const RealType& velocity_other1, const RealType& velocity_other2, const RealType& pressure ) const
      {
         const RealType& speedOfSound = std::sqrt( this->gamma * pressure / density );
         const RealType& machNumber = velocity_main / speedOfSound;
         if ( machNumber <= -1.0 )
            return 0.0;
        else if ( machNumber <= 0.0 )
            return density * speedOfSound * speedOfSound * speedOfSound / ( 2.0 * this->gamma ) 
                 * ( 
                     ( ( machNumber + 1.0 ) * 0.5 *  ( machNumber * machNumber + ( velocity_other1 * velocity_other1 + velocity_other2 * velocity_other2 ) / ( speedOfSound * speedOfSound ) ) )
                   + ( machNumber * ( machNumber + 1.0 ) )
                   + ( ( machNumber + 1.0 ) / ( this->gamma - 1.0 ) )
                   );
        else if ( machNumber <= 1.0 )
            return density * speedOfSound * speedOfSound * speedOfSound / ( 2.0 * this->gamma ) 
                 * ( 
                     ( ( ( 2.0 * this->gamma - 1.0 ) * machNumber + 1.0 ) * 0.5 * ( machNumber * machNumber + ( velocity_other1 * velocity_other1 + velocity_other2 * velocity_other2 ) / ( speedOfSound * speedOfSound ) ) )
                   + ( machNumber * ( machNumber + 1.0 ) )
                   + ( ( machNumber + 1.0 ) / ( this->gamma - 1.0 ) )
                   );
        else   
            return velocity_main * ( pressure + pressure / ( this->gamma - 1.0 ) + 0.5 * density * ( velocity_main * velocity_main + velocity_other1 * velocity_other1 + velocity_other2 * velocity_other2 ) );
      };

      RealType negativeEnergyFlux( const RealType& density, const RealType& velocity_main, const RealType& velocity_other1, const RealType& velocity_other2, const RealType& pressure ) const
      {
         const RealType& speedOfSound = std::sqrt( this->gamma * pressure / density );
         const RealType& machNumber = velocity_main / speedOfSound;
         if ( machNumber <= -1.0 )
            return velocity_main * ( pressure + pressure / ( this->gamma - 1.0 ) + 0.5 * density * ( velocity_main * velocity_main + velocity_other1 * velocity_other1 + velocity_other2 * velocity_other2 ) );
        else if ( machNumber <= 0.0 )
            return density * speedOfSound * speedOfSound * speedOfSound / ( 2.0 * this->gamma ) 
                 * ( 
                     ( ( ( 2.0 * this->gamma - 1.0 ) * machNumber - 1.0 ) * 0.5 * ( machNumber * machNumber + ( velocity_other1 * velocity_other1 + velocity_other2 * velocity_other2 ) / ( speedOfSound * speedOfSound ) ) )
                   - ( machNumber * ( machNumber - 1.0 ) )
                   + ( ( machNumber - 1.0 ) / ( this->gamma - 1.0 ) )
                   );
        else if ( machNumber <= 1.0 )
            return density * speedOfSound * speedOfSound * speedOfSound / ( 2.0 * this->gamma ) 
                 * ( 
                     ( ( machNumber - 1.0 ) * 0.5 * ( machNumber * machNumber + ( velocity_other1 * velocity_other1 + velocity_other2 * velocity_other2 ) / ( speedOfSound * speedOfSound ) ) )
                   - ( machNumber * ( machNumber - 1.0 ) )
                   + ( ( machNumber - 1.0 ) / ( this->gamma - 1.0 ) )
                   );
        else 
            return 0.0;
      };      

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         static_assert( MeshEntity::getEntityDimension() == 3, "Wrong mesh entity dimensions." ); 
         static_assert( MeshFunction::getEntitiesDimension() == 3, "Wrong preimage function" ); 
         const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities(); 
 
         const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1,  0,  0 >(); 
         const RealType& hyInverse = entity.getMesh().template getSpaceStepsProducts<  0, -1,  0 >(); 
         const RealType& hzInverse = entity.getMesh().template getSpaceStepsProducts<  0,  0, -1 >();
 
         const IndexType& center    = entity.getIndex(); 
         const IndexType& east      = neighborEntities.template getEntityIndex<  1,  0,  0 >(); 
         const IndexType& west      = neighborEntities.template getEntityIndex< -1,  0,  0 >(); 
         const IndexType& north     = neighborEntities.template getEntityIndex<  0,  1,  0 >(); 
         const IndexType& south     = neighborEntities.template getEntityIndex<  0, -1,  0 >();
         const IndexType& up        = neighborEntities.template getEntityIndex<  0,  0,  1 >(); 
         const IndexType& down      = neighborEntities.template getEntityIndex<  0,  0, -1 >();

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
         
         const RealType& velocity_x_center    = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_x_east      = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_x_west      = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_x_north     = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ north ];
         const RealType& velocity_x_south     = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ south ];
         const RealType& velocity_x_up        = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ up ];
         const RealType& velocity_x_down      = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ down ];

         const RealType& velocity_y_center    = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_y_east      = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_y_west      = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_y_north     = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ north ];
         const RealType& velocity_y_south     = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ south ];
         const RealType& velocity_y_up        = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ up ];
         const RealType& velocity_y_down      = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ down ];

         const RealType& velocity_z_center    = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_z_east      = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_z_west      = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_z_north     = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ north ];
         const RealType& velocity_z_south     = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ south ];
         const RealType& velocity_z_up        = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ up ];
         const RealType& velocity_z_down      = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ down ];        
         
         return -hxInverse * ( 
                                   this->positiveEnergyFlux( density_center, velocity_x_center, velocity_y_center, velocity_z_center, pressure_center)
                                 - this->positiveEnergyFlux( density_west  , velocity_x_west  , velocity_y_west  , velocity_z_west  , pressure_west  )
                                 - this->negativeEnergyFlux( density_center, velocity_x_center, velocity_y_center, velocity_z_center, pressure_center)
                                 + this->negativeEnergyFlux( density_east  , velocity_x_east  , velocity_y_east  , velocity_z_east  , pressure_east  ) 
                             ) 
                -hyInverse * ( 
                                   this->positiveEnergyFlux( density_center, velocity_y_center, velocity_x_center, velocity_z_center, pressure_center)
                                 - this->positiveEnergyFlux( density_south , velocity_y_south , velocity_x_south , velocity_z_south , pressure_south )
                                 - this->negativeEnergyFlux( density_center, velocity_y_center, velocity_x_center, velocity_z_center, pressure_center)
                                 + this->negativeEnergyFlux( density_north , velocity_y_north , velocity_x_north , velocity_z_north , pressure_north ) 
                             )
                -hzInverse * ( 
                                   this->positiveEnergyFlux( density_center, velocity_z_center, velocity_x_center, velocity_y_center, pressure_center)
                                 - this->positiveEnergyFlux( density_down  , velocity_z_down  , velocity_x_down  , velocity_y_down  , pressure_down  )
                                 - this->negativeEnergyFlux( density_center, velocity_z_center, velocity_x_center, velocity_y_center, pressure_center)
                                 + this->negativeEnergyFlux( density_up    , velocity_z_up    , velocity_x_up    , velocity_y_up    , pressure_up    ) 
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
