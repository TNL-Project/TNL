/***************************************************************************
                          TurbulentViscosityGetter.h  -  description
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
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Functions/MeshFunctionEvaluator.h>

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class TurbulentViscosityGetterBase : public Functions::Domain< Mesh::getMeshDimension(), Functions::MeshDomain >
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

      TurbulentViscosityGetterBase( MeshFunctionPointer density,
                                    VelocityFieldPointer velocity,
                                    const RealType& mixingLength )
      : density( density ), velocity( velocity ), mixingLength( mixingLength ) {}
      
      static String getType()
      {
         return String( "TurbulentViscosityGetter< " ) +
             MeshType::getType() + ", " +
             TNL::getType< Real >() + ", " +
             TNL::getType< Index >() + " >"; 
      }

      protected:
            const MeshFunctionPointer density;

            const VelocityFieldPointer velocity;

            const RealType mixingLength;


      
};
   
template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class TurbulentViscosityGetter
{
};



template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class TurbulentViscosityGetter< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >
   : public TurbulentViscosityGetterBase< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef TurbulentViscosityGetterBase< MeshType, Real, Index > BaseType;
      
      using typename BaseType::RealType;
      using typename BaseType::IndexType;
      using typename BaseType::DeviceType;
      using typename BaseType::CoordinatesType;
      using typename BaseType::MeshFunctionType;
      using typename BaseType::VelocityFieldType;
      using typename BaseType::VelocityFieldPointer;
      using BaseType::Dimensions;

      template< typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
               return 0;
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
class TurbulentViscosityGetter< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >
   : public TurbulentViscosityGetterBase< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef TurbulentViscosityGetterBase< MeshType, Real, Index > BaseType;
      
      using typename BaseType::RealType;
      using typename BaseType::IndexType;
      using typename BaseType::DeviceType;
      using typename BaseType::CoordinatesType;
      using typename BaseType::MeshFunctionType;
      using typename BaseType::VelocityFieldType;
      using typename BaseType::VelocityFieldPointer;
      using BaseType::Dimensions;

      template< typename EntityType >
      __cuda_callable__
      RealType operator()( const EntityType& meshEntity,
                           const RealType& time = 0.0 ) const
      { 
         const typename EntityType::template NeighborEntities< 2 >& neighborEntities = meshEntity.getNeighborEntities(); 
         const RealType rho = this->density.template getData< DeviceType >()( meshEntity );
         const RealType& hxInverse = meshEntity.getMesh().template getSpaceStepsProducts< -1,  0 >();
         const RealType& hyInverse = meshEntity.getMesh().template getSpaceStepsProducts<  0, -1 >();

         const IndexType& center    = meshEntity.getIndex(); 
         const IndexType& east      = neighborEntities.template getEntityIndex<  1,  0 >(); 
         const IndexType& west      = neighborEntities.template getEntityIndex< -1,  0 >(); 
         const IndexType& north     = neighborEntities.template getEntityIndex<  0,  1 >(); 
         const IndexType& south     = neighborEntities.template getEntityIndex<  0, -1 >();
 
        const RealType& velocity_x_north     = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ north ];
        const RealType& velocity_x_south     = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ south ];
        const RealType& velocity_x_center    = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ center ];

        const RealType& velocity_y_east      = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ east ];
        const RealType& velocity_y_west      = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ west ];
        const RealType& velocity_y_center    = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ center ];

        return rho * this->mixingLength * this->mixingLength *
               (-1.0) *
               (  ( velocity_y_east  - velocity_y_west  ) * hxInverse * 0.5
               -  ( velocity_x_north - velocity_x_south ) * hyInverse * 0.5
               )
               *
               (  ( velocity_x_north - velocity_x_south ) * hyInverse * 0.5
               -  ( velocity_y_east  - velocity_y_west  ) * hxInverse * 0.5
                );
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
class TurbulentViscosityGetter< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >
   : public TurbulentViscosityGetterBase< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef TurbulentViscosityGetterBase< MeshType, Real, Index > BaseType;
      
      using typename BaseType::RealType;
      using typename BaseType::IndexType;
      using typename BaseType::DeviceType;
      using typename BaseType::CoordinatesType;
      using typename BaseType::MeshFunctionType;
      using typename BaseType::VelocityFieldType;
      using typename BaseType::VelocityFieldPointer;
      using BaseType::Dimensions;

      template< typename EntityType >
      __cuda_callable__
      RealType operator()( const EntityType& meshEntity,
                           const RealType& time = 0.0 ) const
      { 
        const typename EntityType::template NeighborEntities< 3 >& neighborEntities = meshEntity.getNeighborEntities(); 
        const RealType rho = this->density.template getData< DeviceType >()( meshEntity );

        const RealType& hxInverse = meshEntity.getMesh().template getSpaceStepsProducts< -1,  0,  0 >();
        const RealType& hyInverse = meshEntity.getMesh().template getSpaceStepsProducts<  0, -1,  0 >();
        const RealType& hzInverse = meshEntity.getMesh().template getSpaceStepsProducts<  0,  0, -1 >();

        const IndexType& center    = meshEntity.getIndex(); 
        const IndexType& east      = neighborEntities.template getEntityIndex<  1,  0,  0 >(); 
        const IndexType& west      = neighborEntities.template getEntityIndex< -1,  0,  0 >(); 
        const IndexType& north     = neighborEntities.template getEntityIndex<  0,  1,  0 >(); 
        const IndexType& south     = neighborEntities.template getEntityIndex<  0, -1,  0 >();
        const IndexType& up        = neighborEntities.template getEntityIndex<  0,  0,  1 >(); 
        const IndexType& down      = neighborEntities.template getEntityIndex<  0,  0, -1 >();

        const RealType& velocity_x_north     = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ north ];
        const RealType& velocity_x_south     = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ south ];
        const RealType& velocity_x_center    = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ center ];
        const RealType& velocity_x_up        = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ up ];
        const RealType& velocity_x_down      = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ down ];

        const RealType& velocity_y_east      = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ east ];
        const RealType& velocity_y_west      = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ west ];
        const RealType& velocity_y_center    = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ center ];
        const RealType& velocity_y_up        = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ up ];
        const RealType& velocity_y_down      = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ down ];

        const RealType& velocity_z_north     = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ north ];
        const RealType& velocity_z_south     = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ south ]; 
        const RealType& velocity_z_east      = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ east ];
        const RealType& velocity_z_west      = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ west ]; 
        const RealType& velocity_z_center    = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ center ]; 

        return rho * this->mixingLength * this->mixingLength *
               (  ( velocity_y_east - velocity_y_west ) * hxInverse * 0.5
               -  ( velocity_x_north - velocity_x_south ) * hyInverse * 0.5
               )
               *
               (  ( velocity_z_north - velocity_z_south ) * hyInverse * 0.5
               -  ( velocity_y_up - velocity_y_down ) * hzInverse * 0.5
               )
               *
               (  ( velocity_x_up - velocity_x_down ) * hzInverse * 0.5
               -  ( velocity_z_east - velocity_z_west ) * hxInverse * 0.5
               )
               +
               (  ( velocity_z_east - velocity_z_west ) * hxInverse * 0.5
               -  ( velocity_x_up - velocity_x_down ) * hzInverse * 0.5
               )
               *
               (  ( velocity_x_north - velocity_x_south ) * hyInverse * 0.5
               -  ( velocity_y_east - velocity_y_west ) * hxInverse * 0.5
               )
               *
               (  ( velocity_y_up - velocity_y_down ) * hzInverse * 0.5
               -  ( velocity_z_north - velocity_z_south ) *hyInverse * 0.5
               );
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
