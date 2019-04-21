/***************************************************************************
                          PrandtlKolmogorovTurbulentEnergyOperatorRightHandSide.h  -  description
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
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class PrandtlKolmogorovTurbulentEnergyRightHandSideBase
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
      typedef Pointers::SharedPointer< VelocityFieldType > VelocityFieldPointer;
      typedef Pointers::SharedPointer< MeshFunctionType > MeshFunctionPointer;

    
      static String getType()
      {
         return String( "LaxFridrichsTurbulentEnergy< " ) +
             MeshType::getType() + ", " +
             TNL::getType< Real >() + ", " +
             TNL::getType< Index >() + " >"; 
      }

      void setVelocity( const VelocityFieldPointer& velocity )
      {
          this->velocity = velocity;
      }
    
      void setTurbulentViscosity( const MeshFunctionPointer& turbulentViscosity )
      {
          this->turbulentViscosity = turbulentViscosity;
      }     
      
      void setDynamicalViscosity( const RealType& dynamicalViscosity )
      {
         this->dynamicalViscosity = dynamicalViscosity;
      }     
      
      void setCharacteristicLength( const RealType& characteristicLength )
      {
         this->characteristicLength = characteristicLength;
      }     
      
      void setViscosityConstant( const RealType& viscosityConstant )
      {
         this->viscosityConstant = viscosityConstant;
      }     
      
      void setSigmaK( const RealType& sigmaK )
      {
         this->sigmaK = sigmaK;
      }
    
      void setTurbulentEnergy( const MeshFunctionPointer& turbulentEnergy )
      {
          this->turbulentEnergy = turbulentEnergy;
      }
    
      void setDensity( const MeshFunctionPointer& density )
      {
          this->density = density;
      }

      void setCaracteristicLength( RealType& characteristicLength )
      {
         this->characteristicLength = characteristicLength;
      }

      void setSigmaK( RealType& sigmaK )
      {
         this->sigmaK = sigmaK;
      }

      void setViscosityConstant( RealType& ViscosityConstant )
      {
         this->viscosityConstant = ViscosityConstant;
      }

      protected:

         MeshFunctionPointer turbulentViscosity;

         MeshFunctionPointer turbulentEnergy;

         MeshFunctionPointer density;
         
         VelocityFieldPointer velocity;
         
         RealType dynamicalViscosity;

         RealType characteristicLength;

         RealType viscosityConstant;

         RealType sigmaK;
};

   
template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class PrandtlKolmogorovTurbulentEnergyRightHandSide
{
};



template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class PrandtlKolmogorovTurbulentEnergyRightHandSide< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >
   : public PrandtlKolmogorovTurbulentEnergyRightHandSideBase< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef PrandtlKolmogorovTurbulentEnergyRightHandSideBase< MeshType, Real, Index > BaseType;
      
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

         const RealType& hxSquareInverse = entity.getMesh().template getSpaceStepsProducts< -2 >(); 

         const IndexType& center = entity.getIndex(); 
         const IndexType& east = neighborEntities.template getEntityIndex<  1 >(); 
         const IndexType& west = neighborEntities.template getEntityIndex< -1 >();

         const RealType& turbulentViscosity_center    = this->turbulentViscosity.template getData< DeviceType >()[ center ];         
         const RealType& turbulentViscosity_west      = this->turbulentViscosity.template getData< DeviceType >()[ west ];
         const RealType& turbulentViscosity_east      = this->turbulentViscosity.template getData< DeviceType >()[ east ];

         const RealType& turbulentEnergy_center    = this->turbulentEnergy.template getData< DeviceType >()[ center ];
         const RealType& turbulentEnergy_west      = this->turbulentEnergy.template getData< DeviceType >()[ west ];
         const RealType& turbulentEnergy_east      = this->turbulentEnergy.template getData< DeviceType >()[ east ];

         const RealType& density_center      = this->density.template getData< DeviceType >()[ center ];

         const RealType& velocity_x_east   = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_x_west   = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_x_center = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ center ];

         return
                  4.0 / 3.0 * ( velocity_x_east - velocity_x_west ) * hxInverse / 2
                * this->dynamicalViscosity
                * ( velocity_x_east - velocity_x_west ) * hxInverse / 2

                - density_center / this->characteristicLength * std::pow( std::abs( turbulentEnergy_center ), 2.0 / 3.0 )

                + this->dynamicalViscosity
                * ( turbulentEnergy_east - 2 * turbulentEnergy_center + turbulentEnergy_west )
                * hxSquareInverse
                + ( turbulentEnergy_east * turbulentViscosity_east - turbulentEnergy_center * turbulentViscosity_west
                  - turbulentEnergy_center * turbulentViscosity_center + turbulentEnergy_west * turbulentViscosity_west
                  ) * hxSquareInverse / this->sigmaK;
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
          typename Real,
          typename Index >
class PrandtlKolmogorovTurbulentEnergyRightHandSide< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >
   : public PrandtlKolmogorovTurbulentEnergyRightHandSideBase< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef PrandtlKolmogorovTurbulentEnergyRightHandSideBase< MeshType, Real, Index > BaseType;
      
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
 
         const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1,  0 >(); 
         const RealType& hyInverse = entity.getMesh().template getSpaceStepsProducts<  0, -1 >();

         const RealType& hxSquareInverse = entity.getMesh().template getSpaceStepsProducts< -2,  0 >(); 
         const RealType& hySquareInverse = entity.getMesh().template getSpaceStepsProducts<  0, -2 >();  

         const IndexType& center    = entity.getIndex(); 
         const IndexType& east      = neighborEntities.template getEntityIndex<  1,  0 >(); 
         const IndexType& west      = neighborEntities.template getEntityIndex< -1,  0 >(); 
         const IndexType& north     = neighborEntities.template getEntityIndex<  0,  1 >(); 
         const IndexType& south     = neighborEntities.template getEntityIndex<  0, -1 >();
         const IndexType& southEast = neighborEntities.template getEntityIndex<  1, -1 >();
         const IndexType& southWest = neighborEntities.template getEntityIndex< -1, -1 >();
         const IndexType& northEast = neighborEntities.template getEntityIndex<  1,  1 >();
         const IndexType& northWest = neighborEntities.template getEntityIndex< -1,  1 >();

         const RealType& turbulentViscosity_center    = this->turbulentViscosity.template getData< DeviceType >()[ center ];         
         const RealType& turbulentViscosity_west      = this->turbulentViscosity.template getData< DeviceType >()[ west ];
         const RealType& turbulentViscosity_east      = this->turbulentViscosity.template getData< DeviceType >()[ east ];
         const RealType& turbulentViscosity_north     = this->turbulentViscosity.template getData< DeviceType >()[ north ];
         const RealType& turbulentViscosity_south     = this->turbulentViscosity.template getData< DeviceType >()[ south ];
         
         const RealType& turbulentEnergy_center    = this->turbulentEnergy.template getData< DeviceType >()[ center ];
         const RealType& turbulentEnergy_west      = this->turbulentEnergy.template getData< DeviceType >()[ west ];
         const RealType& turbulentEnergy_east      = this->turbulentEnergy.template getData< DeviceType >()[ east ];
         const RealType& turbulentEnergy_north     = this->turbulentEnergy.template getData< DeviceType >()[ north ];
         const RealType& turbulentEnergy_south     = this->turbulentEnergy.template getData< DeviceType >()[ south ];
         
         const RealType& density_center    = this->density.template getData< DeviceType >()[ center ];

         const RealType& velocity_x_north     = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ north ];
         const RealType& velocity_x_south     = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ south ];
         const RealType& velocity_x_east      = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_x_west      = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_x_center    = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_x_southEast = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ southEast ];
         const RealType& velocity_x_southWest = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ southWest ];
         const RealType& velocity_x_northEast = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ northEast ];
         const RealType& velocity_x_northWest = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ northWest ];         

         const RealType& velocity_y_north     = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ north ];
         const RealType& velocity_y_south     = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ south ];
         const RealType& velocity_y_east      = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_y_west      = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_y_center    = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_y_southEast = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ southEast ];
         const RealType& velocity_y_southWest = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ southWest ];
         const RealType& velocity_y_northEast = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ northEast ];
         const RealType& velocity_y_northWest = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ northWest ];         
         
         return
                 ( 4.0 / 3.0 * ( velocity_x_east  - velocity_x_west  ) * hxInverse / 2
                 - 2.0 / 3.0 * ( velocity_y_north - velocity_y_south ) * hyInverse / 2
                 ) * this->dynamicalViscosity 
                 * ( velocity_x_east  - velocity_x_west  ) * hxInverse / 2

                + ( ( velocity_x_north - velocity_x_south ) * hyInverse / 2
                  + ( velocity_y_east  - velocity_y_west  ) * hxInverse / 2
                  ) * this->dynamicalViscosity 
                * ( velocity_y_east  - velocity_y_west  ) * hxInverse / 2

                + ( ( velocity_y_east  - velocity_y_west  ) * hxInverse / 2
                  + ( velocity_x_north - velocity_x_south ) * hyInverse / 2
                  ) * this->dynamicalViscosity
                * ( velocity_x_north  - velocity_x_south ) * hyInverse / 2

                + ( 4.0 / 3.0 * ( velocity_y_north - velocity_y_south ) * hyInverse / 2
                  - 2.0 / 3.0 * ( velocity_x_east  - velocity_x_west  ) * hxInverse / 2
                  ) * this->dynamicalViscosity
                * ( velocity_y_north - velocity_y_south ) * hyInverse / 2

                - density_center / this->characteristicLength * std::pow( std::abs( turbulentEnergy_center ), 2.0 / 3.0 )

                + this->dynamicalViscosity
                * ( turbulentEnergy_east - 2 * turbulentEnergy_center + turbulentEnergy_west )
                * hxSquareInverse
                + ( turbulentEnergy_east * turbulentViscosity_east - turbulentEnergy_center * turbulentViscosity_west
                  - turbulentEnergy_center * turbulentViscosity_center + turbulentEnergy_west * turbulentViscosity_west
                  ) * hxSquareInverse / this->sigmaK

                + this->dynamicalViscosity
                * ( turbulentEnergy_north - 2 * turbulentEnergy_center + turbulentEnergy_south )
                * hySquareInverse
                + ( turbulentEnergy_north * turbulentViscosity_north - turbulentEnergy_center * turbulentViscosity_south
                  - turbulentEnergy_center * turbulentViscosity_center + turbulentEnergy_south * turbulentViscosity_south
                  ) * hySquareInverse / this->sigmaK;
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
          typename Real,
          typename Index >
class PrandtlKolmogorovTurbulentEnergyRightHandSide< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >
   : public PrandtlKolmogorovTurbulentEnergyRightHandSideBase< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef PrandtlKolmogorovTurbulentEnergyRightHandSideBase< MeshType, Real, Index > BaseType;
      
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
 
         const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1,  0,  0 >(); 
         const RealType& hyInverse = entity.getMesh().template getSpaceStepsProducts<  0, -1,  0 >(); 
         const RealType& hzInverse = entity.getMesh().template getSpaceStepsProducts<  0,  0, -1 >(); 

         const RealType& hxSquareInverse = entity.getMesh().template getSpaceStepsProducts< -2,  0,  0 >(); 
         const RealType& hySquareInverse = entity.getMesh().template getSpaceStepsProducts<  0, -2,  0 >(); 
         const RealType& hzSquareInverse = entity.getMesh().template getSpaceStepsProducts<  0,  0, -2 >(); 

         const IndexType& center    = entity.getIndex(); 
         const IndexType& east      = neighborEntities.template getEntityIndex<  1,  0,  0 >(); 
         const IndexType& west      = neighborEntities.template getEntityIndex< -1,  0,  0 >(); 
         const IndexType& north     = neighborEntities.template getEntityIndex<  0,  1,  0 >(); 
         const IndexType& south     = neighborEntities.template getEntityIndex<  0, -1,  0 >();
         const IndexType& up        = neighborEntities.template getEntityIndex<  0,  0,  1 >(); 
         const IndexType& down      = neighborEntities.template getEntityIndex<  0,  0, -1 >();
         const IndexType& northWest = neighborEntities.template getEntityIndex< -1,  1,  0 >(); 
         const IndexType& northEast = neighborEntities.template getEntityIndex<  1,  1,  0 >(); 
         const IndexType& southWest = neighborEntities.template getEntityIndex< -1, -1,  0 >();
         const IndexType& southEast = neighborEntities.template getEntityIndex<  1, -1,  0 >();
         const IndexType& upWest    = neighborEntities.template getEntityIndex< -1,  0,  1 >();
         const IndexType& upEast    = neighborEntities.template getEntityIndex<  1,  0,  1 >();
         const IndexType& upSouth   = neighborEntities.template getEntityIndex<  0, -1,  1 >();
         const IndexType& upNorth   = neighborEntities.template getEntityIndex<  0,  1,  1 >();
         const IndexType& downWest  = neighborEntities.template getEntityIndex< -1,  0, -1 >();
         const IndexType& downEast  = neighborEntities.template getEntityIndex<  1,  0, -1 >();
         const IndexType& downSouth = neighborEntities.template getEntityIndex<  0, -1, -1 >();
         const IndexType& downNorth = neighborEntities.template getEntityIndex<  0,  1, -1 >();

         const RealType& turbulentViscosity_center    = this->turbulentViscosity.template getData< DeviceType >()[ center ];
         const RealType& turbulentViscosity_west      = this->turbulentViscosity.template getData< DeviceType >()[ west ];
         const RealType& turbulentViscosity_east      = this->turbulentViscosity.template getData< DeviceType >()[ east ];
         const RealType& turbulentViscosity_north     = this->turbulentViscosity.template getData< DeviceType >()[ north ];
         const RealType& turbulentViscosity_south     = this->turbulentViscosity.template getData< DeviceType >()[ south ];
         const RealType& turbulentViscosity_up        = this->turbulentViscosity.template getData< DeviceType >()[ up ];
         const RealType& turbulentViscosity_down      = this->turbulentViscosity.template getData< DeviceType >()[ down ];

         const RealType& turbulentEnergy_center    = this->turbulentEnergy.template getData< DeviceType >()[ center ];
         const RealType& turbulentEnergy_west      = this->turbulentEnergy.template getData< DeviceType >()[ west ];
         const RealType& turbulentEnergy_east      = this->turbulentEnergy.template getData< DeviceType >()[ east ];
         const RealType& turbulentEnergy_north     = this->turbulentEnergy.template getData< DeviceType >()[ north ];
         const RealType& turbulentEnergy_south     = this->turbulentEnergy.template getData< DeviceType >()[ south ];
         const RealType& turbulentEnergy_up        = this->turbulentEnergy.template getData< DeviceType >()[ up ];
         const RealType& turbulentEnergy_down      = this->turbulentEnergy.template getData< DeviceType >()[ down ];
         
         const RealType& density_center    = this->density.template getData< DeviceType >()[ center ];
         
         const RealType& velocity_x_north     = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ north ];
         const RealType& velocity_x_south     = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ south ];
         const RealType& velocity_x_east      = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_x_west      = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_x_up        = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ up ];
         const RealType& velocity_x_down      = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ down ];
         const RealType& velocity_x_center    = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_x_northWest = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ northWest ];
         const RealType& velocity_x_northEast = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ northEast ];
         const RealType& velocity_x_southWest = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ southWest ];
         const RealType& velocity_x_southEast = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ southEast ];
         const RealType& velocity_x_upWest    = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ upWest ];
         const RealType& velocity_x_downWest  = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ downWest ];
         const RealType& velocity_x_upEast    = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ upEast ];
         const RealType& velocity_x_downEast  = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ downEast ];

         const RealType& velocity_y_north     = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ north ];
         const RealType& velocity_y_south     = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ south ];
         const RealType& velocity_y_east      = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_y_west      = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_y_up        = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ up ];
         const RealType& velocity_y_down      = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ down ];
         const RealType& velocity_y_center    = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_y_northWest = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ northWest ];
         const RealType& velocity_y_northEast = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ northEast ];
         const RealType& velocity_y_southWest = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ southWest ];
         const RealType& velocity_y_southEast = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ southEast ];
         const RealType& velocity_y_upNorth   = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ upNorth ];
         const RealType& velocity_y_upSouth   = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ upSouth ];
         const RealType& velocity_y_downNorth = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ downNorth ];
         const RealType& velocity_y_downSouth = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ downSouth ];

         const RealType& velocity_z_north     = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ north ];
         const RealType& velocity_z_south     = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ south ]; 
         const RealType& velocity_z_east      = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_z_west      = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ west ]; 
         const RealType& velocity_z_up        = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ up ];
         const RealType& velocity_z_down      = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ down ]; 
         const RealType& velocity_z_center    = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ center ]; 
         const RealType& velocity_z_upWest    = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ upWest ]; 
         const RealType& velocity_z_upEast    = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ upEast ]; 
         const RealType& velocity_z_upNorth   = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ upNorth ]; 
         const RealType& velocity_z_upSouth   = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ upSouth ]; 
         const RealType& velocity_z_downWest  = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ downWest ]; 
         const RealType& velocity_z_downEast  = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ downEast ]; 
         const RealType& velocity_z_downNorth = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ downNorth ]; 
         const RealType& velocity_z_downSouth = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ downSouth ];         
         
         return 
// 3D uT_11_x
                  ( 4.0 / 3.0 * ( velocity_x_east  - velocity_x_west  ) * hxInverse / 2
                  - 2.0 / 3.0 * ( velocity_y_north - velocity_x_south ) * hyInverse / 2
                  - 2.0 / 3.0 * ( velocity_z_up    - velocity_z_down  ) * hzInverse / 2
                  ) * this->dynamicalViscosity
                  * ( velocity_x_east  - velocity_x_west  ) * hxInverse / 2
// vT_12_x
                + ( ( velocity_x_north - velocity_x_south ) * hyInverse / 2
                  + ( velocity_y_east  - velocity_y_west  ) * hxInverse / 2
                  ) * this->dynamicalViscosity
                  * ( velocity_y_east  - velocity_y_west  ) * hxInverse / 2
// wT_13_x
                + ( ( velocity_x_up   - velocity_x_down ) * hzInverse / 2
                  + ( velocity_z_east - velocity_z_west ) * hxInverse / 2
                  ) * this->dynamicalViscosity
                  * ( velocity_z_east  - velocity_z_west  ) * hxInverse / 2
// uT_21_y
                + ( ( velocity_y_east  - velocity_y_west   ) * hxInverse / 2
                  + ( velocity_x_north - velocity_x_south  ) * hyInverse / 2
                  ) * this->dynamicalViscosity
                  * ( velocity_x_north - velocity_x_south  ) * hyInverse / 2
// 3D vT_22_y
                + ( 4.0 / 3.0 * ( velocity_y_north - velocity_y_south ) * hyInverse / 2
                  - 2.0 / 3.0 * ( velocity_x_east  - velocity_x_west  ) * hxInverse / 2
                  - 2.0 / 3.0 * ( velocity_z_up    - velocity_z_down  ) * hzInverse / 2
                  ) * this->dynamicalViscosity
                  * ( velocity_y_north - velocity_y_south ) * hyInverse / 2
// wT_23_y
                + ( ( velocity_y_up    - velocity_y_down  ) * hzInverse / 2
                  + ( velocity_z_north - velocity_z_south ) * hyInverse / 2
                  ) * this->dynamicalViscosity
                  * ( velocity_z_north - velocity_z_south ) * hyInverse / 2
// uT_31_z
                + ( ( velocity_x_up   - velocity_x_down ) * hzInverse / 2
                  + ( velocity_z_east - velocity_z_west ) * hxInverse / 2
                  ) * this->dynamicalViscosity
                  * ( velocity_x_up   - velocity_x_down ) * hzInverse / 2
// T_32_z
                + ( ( velocity_z_north - velocity_z_south ) * hyInverse / 2
                  + ( velocity_y_up    - velocity_y_down  ) * hzInverse / 2
                  ) * this->dynamicalViscosity
                  * ( velocity_y_up    - velocity_y_down  ) * hzInverse / 2
// 3D T_33_z
                + ( 4.0 / 3.0 * ( velocity_z_up    - velocity_z_down  ) * hzInverse / 2
                  - 2.0 / 3.0 * ( velocity_y_north - velocity_y_south ) * hyInverse / 2
                  - 2.0 / 3.0 * ( velocity_x_east  - velocity_x_west  ) * hxInverse / 2
                  ) * this->dynamicalViscosity
                  * ( velocity_z_up    - velocity_z_down  ) * hzInverse / 2

                - density_center / this->characteristicLength * std::pow( std::abs( turbulentEnergy_center ), 2.0 / 3.0 )

                + this->dynamicalViscosity
                * ( turbulentEnergy_east - 2 * turbulentEnergy_center + turbulentEnergy_west )
                * hxSquareInverse
                + ( turbulentEnergy_east * turbulentViscosity_east - turbulentEnergy_center * turbulentViscosity_west
                  - turbulentEnergy_center * turbulentViscosity_center + turbulentEnergy_west * turbulentViscosity_west
                  ) * hxSquareInverse / this->sigmaK

                + this->dynamicalViscosity
                * ( turbulentEnergy_north - 2 * turbulentEnergy_center + turbulentEnergy_south )
                * hySquareInverse
                + ( turbulentEnergy_north * turbulentViscosity_north - turbulentEnergy_center * turbulentViscosity_south
                  - turbulentEnergy_center * turbulentViscosity_center + turbulentEnergy_south * turbulentViscosity_south
                  ) * hySquareInverse / this->sigmaK

                + this->dynamicalViscosity
                * ( turbulentEnergy_up - 2 * turbulentEnergy_center + turbulentEnergy_down )
                * hySquareInverse
                + ( turbulentEnergy_up * turbulentViscosity_up - turbulentEnergy_center * turbulentViscosity_down
                  - turbulentEnergy_center * turbulentViscosity_center + turbulentEnergy_down * turbulentViscosity_down
                  ) * hySquareInverse / this->sigmaK;                  
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
