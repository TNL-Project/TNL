/***************************************************************************
                          KEpsilonDisipationOperatorRightHandSide.h  -  description
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
class KEpsilonDisipationRightHandSideBase
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
         return String( "LaxFridrichsContinuity< " ) +
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
      
      void setViscosityConstant1( const RealType& viscosityConstant1 )
      {
         this->viscosityConstant1 = viscosityConstant1;
      }     
      
      void setViscosityConstant2( const RealType& viscosityConstant2 )
      {
         this->viscosityConstant2 = viscosityConstant2;
      }     
      
      void setSigmaEpsilon( const RealType& sigmaEpsilon )
      {
         this->sigmaEpsilon = sigmaEpsilon;
      }
    
      void setTurbulentEnergy( const MeshFunctionPointer& turbulentEnergy )
      {
          this->turbulentEnergy = turbulentEnergy;
      }
    
      void setDisipation( const MeshFunctionPointer& disipation )
      {
          this->disipation = disipation;
      }
    
      void setDensity( const MeshFunctionPointer& density )
      {
          this->density = density;
      }  

      protected:

         MeshFunctionPointer turbulentViscosity;

         MeshFunctionPointer turbulentEnergy;

         MeshFunctionPointer disipation;

         MeshFunctionPointer density;
         
         VelocityFieldPointer velocity;
         
         RealType dynamicalViscosity;

         RealType viscosityConstant1;

         RealType viscosityConstant2;

         RealType sigmaEpsilon;
};

   
template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class KEpsilonDisipationRightHandSide
{
};



template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class KEpsilonDisipationRightHandSide< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >
   : public KEpsilonDisipationRightHandSideBase< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef KEpsilonDisipationRightHandSideBase< MeshType, Real, Index > BaseType;
      
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

         const RealType& disipation_center    = this->disipation.template getData< DeviceType >()[ center ];
         const RealType& disipation_west      = this->disipation.template getData< DeviceType >()[ west ];
         const RealType& disipation_east      = this->disipation.template getData< DeviceType >()[ east ];

         const RealType& density_center      = this->density.template getData< DeviceType >()[ center ];

         const RealType& turbulentEnergy_center    = this->turbulentEnergy.template getData< DeviceType >()[ center ]; 

         const RealType& velocity_x_east   = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_x_west   = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_x_center = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ center ];

         return
                this->dynamicalViscosity
                * ( disipation_east - 2 * disipation_center + disipation_west )
                * hxSquareInverse
                + ( disipation_east * turbulentViscosity_east - disipation_center * turbulentViscosity_west
                  - disipation_center * turbulentViscosity_center + disipation_west * turbulentViscosity_west
                  ) * hxSquareInverse / this->sigmaEpsilon

                - this->viscosityConstant1 * disipation_center / turbulentEnergy_center
                * 4.0 / 3.0 * ( velocity_x_east - velocity_x_west ) * hxInverse / 2
                * turbulentViscosity_center
                * ( velocity_x_east - velocity_x_west ) * hxInverse / 2

                - this->viscosityConstant2 * density_center * disipation_center * disipation_center / turbulentEnergy_center;
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
class KEpsilonDisipationRightHandSide< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >
   : public KEpsilonDisipationRightHandSideBase< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef KEpsilonDisipationRightHandSideBase< MeshType, Real, Index > BaseType;
      
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
         
         const RealType& disipation_center    = this->disipation.template getData< DeviceType >()[ center ];
         const RealType& disipation_west      = this->disipation.template getData< DeviceType >()[ west ];
         const RealType& disipation_east      = this->disipation.template getData< DeviceType >()[ east ];
         const RealType& disipation_north     = this->disipation.template getData< DeviceType >()[ north ];
         const RealType& disipation_south     = this->disipation.template getData< DeviceType >()[ south ];
         
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
                 this->dynamicalViscosity
                * ( disipation_east - 2 * disipation_center + disipation_west )
                * hxSquareInverse
                + ( disipation_east * turbulentViscosity_east - disipation_center * turbulentViscosity_west
                  - disipation_center * turbulentViscosity_center + disipation_west * turbulentViscosity_west
                  ) * hxSquareInverse / this->sigmaEpsilon

                + this->dynamicalViscosity
                * ( disipation_north - 2 * disipation_center + disipation_south )
                * hySquareInverse
                + ( disipation_north * turbulentViscosity_north - disipation_center * turbulentViscosity_south
                  - disipation_center * turbulentViscosity_center + disipation_south * turbulentViscosity_south
                  ) * hySquareInverse / this->sigmaEpsioln

               - this->viscosityConstant1 * disipation_center / turbulentEnergy_center   
             * ( ( 4.0 / 3.0 * ( velocity_x_east  - velocity_x_west  ) * hxInverse / 2
                 - 2.0 / 3.0 * ( velocity_y_north - velocity_y_south ) * hyInverse / 2
                 ) * turbulentViscosity_center 
                 * ( velocity_x_east  - velocity_x_west  ) * hxInverse / 2

                + ( ( velocity_x_north - velocity_x_south ) * hyInverse / 2
                  + ( velocity_y_east  - velocity_y_west  ) * hxInverse / 2
                  ) * turbulentViscosity_center 
                * ( velocity_y_east  - velocity_y_west  ) * hxInverse / 2

                + ( ( velocity_y_east  - velocity_y_west  ) * hxInverse / 2
                  + ( velocity_x_north - velocity_x_south ) * hyInverse / 2
                  ) * turbulentViscosity_center
                * ( velocity_x_north  - velocity_x_south ) * hyInverse / 2

                + ( 4.0 / 3.0 * ( velocity_y_north - velocity_y_south ) * hyInverse / 2
                  - 2.0 / 3.0 * ( velocity_x_east  - velocity_x_west  ) * hxInverse / 2
                  ) * turbulentViscosity_center
                * ( velocity_y_north - velocity_y_south ) * hyInverse / 2
                )

                - this->viscosityConstant2 * density_center * disipation_center * disipation_center / turbulentEnergy_center;
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
class KEpsilonDisipationRightHandSide< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >
   : public KEpsilonDisipationRightHandSideBase< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef KEpsilonDisipationRightHandSideBase< MeshType, Real, Index > BaseType;
      
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

         const RealType& disipation_center    = this->disipation.template getData< DeviceType >()[ center ];
         const RealType& disipation_west      = this->disipation.template getData< DeviceType >()[ west ];
         const RealType& disipation_east      = this->disipation.template getData< DeviceType >()[ east ];
         const RealType& disipation_north     = this->disipation.template getData< DeviceType >()[ north ];
         const RealType& disipation_south     = this->disipation.template getData< DeviceType >()[ south ];
         const RealType& disipation_up        = this->disipation.template getData< DeviceType >()[ up ];
         const RealType& disipation_down      = this->disipation.template getData< DeviceType >()[ down ];
         
         const RealType& density_center    = this->density.template getData< DeviceType >()[ center ];
         const RealType& density_west      = this->density.template getData< DeviceType >()[ west ];
         const RealType& density_east      = this->density.template getData< DeviceType >()[ east ];
         const RealType& density_north     = this->density.template getData< DeviceType >()[ north ];
         const RealType& density_south     = this->density.template getData< DeviceType >()[ south ];
         const RealType& density_up        = this->density.template getData< DeviceType >()[ up ];
         const RealType& density_down      = this->density.template getData< DeviceType >()[ down ];
         
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
                  this->dynamicalViscosity
                * ( disipation_east - 2 * disipation_center + disipation_west )
                * hxSquareInverse
                + ( disipation_east * turbulentViscosity_east - disipation_center * turbulentViscosity_west
                  - disipation_center * turbulentViscosity_center + disipation_west * turbulentViscosity_west
                  ) * hxSquareInverse / this->sigmaEpsilon

                + this->dynamicalViscosity
                * ( disipation_north - 2 * disipation_center + disipation_south )
                * hySquareInverse
                + ( disipation_north * turbulentViscosity_north - disipation_center * turbulentViscosity_south
                  - disipation_center * turbulentViscosity_center + disipation_south * turbulentViscosity_south
                  ) * hySquareInverse / this->sigmaEpsilon

                + this->dynamicalViscosity
                * ( disipation_up - 2 * disipation_center + disipation_down )
                * hySquareInverse
                + ( disipation_up * turbulentViscosity_up - disipation_center * turbulentViscosity_down
                  - disipation_center * turbulentViscosity_center + disipation_down * turbulentViscosity_down
                  ) * hySquareInverse / this->sigmaEpsilon

                - this->viscosityConstant1 * disipation_center / turbulentEnergy_center      
              * ( ( 4.0 / 3.0 * ( velocity_x_east  - velocity_x_west  ) * hxInverse / 2
                  - 2.0 / 3.0 * ( velocity_y_north - velocity_x_south ) * hyInverse / 2
                  - 2.0 / 3.0 * ( velocity_z_up    - velocity_z_down  ) * hzInverse / 2
                  ) * turbulentViscosity_center
                  * ( velocity_x_east  - velocity_x_west  ) * hxInverse / 2

                + ( ( velocity_x_north - velocity_x_south ) * hyInverse / 2
                  + ( velocity_y_east  - velocity_y_west  ) * hxInverse / 2
                  ) * turbulentViscosity_center
                  * ( velocity_y_east  - velocity_y_west  ) * hxInverse / 2

                + ( ( velocity_x_up   - velocity_x_down ) * hzInverse / 2
                  + ( velocity_z_east - velocity_z_west ) * hxInverse / 2
                  ) * turbulentViscosity_center
                  * ( velocity_z_east  - velocity_z_west  ) * hxInverse / 2

                + ( ( velocity_y_east  - velocity_y_west   ) * hxInverse / 2
                  + ( velocity_x_north - velocity_x_south  ) * hyInverse / 2
                  ) * turbulentViscosity_center
                  * ( velocity_x_north - velocity_x_south  ) * hyInverse / 2

                + ( 4.0 / 3.0 * ( velocity_y_north - velocity_y_south ) * hyInverse / 2
                  - 2.0 / 3.0 * ( velocity_x_east  - velocity_x_west  ) * hxInverse / 2
                  - 2.0 / 3.0 * ( velocity_z_up    - velocity_z_down  ) * hzInverse / 2
                  ) * turbulentViscosity_center
                  * ( velocity_y_north - velocity_y_south ) * hyInverse / 2

                + ( ( velocity_y_up    - velocity_y_down  ) * hzInverse / 2
                  + ( velocity_z_north - velocity_z_south ) * hyInverse / 2
                  ) * turbulentViscosity_center
                  * ( velocity_z_north - velocity_z_south ) * hyInverse / 2

                + ( ( velocity_x_up   - velocity_x_down ) * hzInverse / 2
                  + ( velocity_z_east - velocity_z_west ) * hxInverse / 2
                  ) * turbulentViscosity_center
                  * ( velocity_x_up   - velocity_x_down ) * hzInverse / 2

                + ( ( velocity_z_north - velocity_z_south ) * hyInverse / 2
                  + ( velocity_y_up    - velocity_y_down  ) * hzInverse / 2
                  ) * turbulentViscosity_center
                  * ( velocity_y_up    - velocity_y_down  ) * hzInverse / 2

                + ( 4.0 / 3.0 * ( velocity_z_up    - velocity_z_down  ) * hzInverse / 2
                  - 2.0 / 3.0 * ( velocity_y_north - velocity_y_south ) * hyInverse / 2
                  - 2.0 / 3.0 * ( velocity_x_east  - velocity_x_west  ) * hxInverse / 2
                  ) * turbulentViscosity_center
                  * ( velocity_z_up    - velocity_z_down  ) * hzInverse / 2
                )

                - this->viscosityConstant2 * density_center * disipation_center * disipation_center / turbulentEnergy_center;                
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
