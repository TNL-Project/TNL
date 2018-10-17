/***************************************************************************
                          LaxFridrichsContinuity.h  -  description
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
#include <TNL/SharedPointer.h>

namespace TNL {

   
template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class LaxFridrichsContinuityBase
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
      typedef SharedPointer< VelocityFieldType > VelocityFieldPointer;

      LaxFridrichsContinuityBase()
       : artificialViscosity( 1.0 ){};
      
      static String getType()
      {
         return String( "LaxFridrichsContinuity< " ) +
             MeshType::getType() + ", " +
             TNL::getType< Real >() + ", " +
             TNL::getType< Index >() + " >"; 
      }

      void setTau(const Real& tau)
      {
          this->tau = tau;
      };
      
      void setVelocity( const VelocityFieldPointer& velocity )
      {
          this->velocity = velocity;
      };
      
      void setArtificialViscosity( const RealType& artificialViscosity )
      {
         this->artificialViscosity = artificialViscosity;
      }


      protected:
         
         RealType tau;
         
         VelocityFieldPointer velocity;
         
         RealType artificialViscosity;
};

   
template< typename Mesh,
	  int OperatorRightHandSide = 0,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class LaxFridrichsContinuity
{
};



template< typename MeshReal,
          typename Device,
          typename MeshIndex,
	  int OperatorRightHandSide,
          typename Real,
          typename Index >
class LaxFridrichsContinuity< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, OperatorRightHandSide, Real, Index >
   : public LaxFridrichsContinuityBase< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef LaxFridrichsContinuityBase< MeshType, Real, Index > BaseType;
      
      using typename BaseType::RealType;
      using typename BaseType::IndexType;
      using typename BaseType::DeviceType;
      using typename BaseType::CoordinatesType;
      using typename BaseType::MeshFunctionType;
      using typename BaseType::VelocityFieldType;
      using typename BaseType::VelocityFieldPointer;
      using BaseType::Dimensions;

/*      template < int OperatorRHS >
      __cuda_callable__
      Real rightHandSide()
      {
      }*/

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

         const RealType& velocity_x_west = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_x_east = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ east ];

//         return 1.0 / ( 2.0 * this->tau ) * this->artificialViscosity * ( u[ west ] - 2.0 * u[ center ]  + u[ east ] ) 
//               - 0.5 * ( u[ west ] * velocity_x_west - u[ east ] * velocity_x_east ) * hxInverse;
         return 1.0 / ( 2.0 * this->tau ) * this->artificialViscosity * ( u[ west ] - 2.0 * u[ center ]  + u[ east ] ) 
               - 0.5 * ( u[ east ] * velocity_x_east - u[ west ] * velocity_x_west ) * hxInverse;
               //+ rightHandSide<OperatorRightHandSide>();
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
/*
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
	  int OperatorRightHandSide,
          typename Real,
          typename Index >
      template<> //Euler
__cuda_callable__
Real
LaxFridrichsContinuity< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, OperatorRightHandSide, Real, Index >::
rightHandSide<1>()
      {
         return 0;
      }
/*
      template<> //Navier-Stokes
      __cuda_callable__
      RealType rightHandSide<1>()
      {
         return 0;
      }
*/


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
	  int OperatorRightHandSide,
          typename Real,
          typename Index >
class LaxFridrichsContinuity< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, OperatorRightHandSide, Real, Index >
   : public LaxFridrichsContinuityBase< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef LaxFridrichsContinuityBase< MeshType, Real, Index > BaseType;
      
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

         const RealType& velocity_x_west  = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_x_east  = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_y_north = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ north ];
         const RealType& velocity_y_south = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ south ];
         
         return 1.0 / ( 4.0 * this->tau ) * this->artificialViscosity * ( u[ west ] + u[ east ] + u[ south ] + u[ north ] - 4.0 * u[ center ] ) 
                       - 0.5 * ( ( u[ east ] * velocity_x_east - u[ west ] * velocity_x_west ) * hxInverse
                               + ( u[ north ] * velocity_y_north - u[ south ] * velocity_y_south ) * hyInverse );
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
	  int OperatorRightHandSide,
          typename Real,
          typename Index >
class LaxFridrichsContinuity< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, OperatorRightHandSide, Real, Index >
   : public LaxFridrichsContinuityBase< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef LaxFridrichsContinuityBase< MeshType, Real, Index > BaseType;
      
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
         
         const RealType& velocity_x_west  = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_x_east  = this->velocity.template getData< TNL::Devices::Host >()[ 0 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_y_north = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ north ];
         const RealType& velocity_y_south = this->velocity.template getData< TNL::Devices::Host >()[ 1 ].template getData< DeviceType >()[ south ];
         const RealType& velocity_z_up    = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ up ];
         const RealType& velocity_z_down  = this->velocity.template getData< TNL::Devices::Host >()[ 2 ].template getData< DeviceType >()[ down ];
         
         return 1.0 / ( 6.0 * this->tau ) * this->artificialViscosity *
                ( u[ west ] + u[ east ] + u[ south ] + u[ north ] + u[ up ] + u[ down ]- 6.0 * u[ center ] ) 
                - 0.5 * ( ( u[ east ] * velocity_x_east - u[ west ] * velocity_x_west ) * hxInverse
                        + ( u[ north ] * velocity_y_north - u[ south ] * velocity_y_south ) * hyInverse
                        + ( u[ up ] * velocity_z_up - u[ down ] * velocity_z_down ) * hzInverse );
         
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
