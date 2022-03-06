// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/GridDetails/NeighborGridEntityGetter.h>
#include <TNL/Meshes/GridDetails/Grid1D.h>
#include <TNL/Meshes/GridDetails/Grid2D.h>
#include <TNL/Meshes/GridDetails/Grid3D.h>

namespace TNL {
namespace Meshes {

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighborEntityDimension   |
 * +-----------------+---------------------------+
 * |       1         |              1            |
 * +-----------------+---------------------------+
 */
template< typename Real, typename Device, typename Index, typename Config >
class NeighborGridEntityGetter< GridEntity< Meshes::Grid< 1, Real, Device, Index >, 1, Config >,
                                1,
                                GridEntityStencilStorageTag< GridEntityNoStencil > >
{
public:
   static constexpr int EntityDimension = 1;
   static constexpr int NeighborEntityDimension = 1;
   using GridType = Meshes::Grid< 1, Real, Device, Index >;
   using GridEntityType = GridEntity< GridType, EntityDimension, Config >;
   using NeighborGridEntityType = GridEntity< GridType, NeighborEntityDimension, Config >;
   using RealType = Real;
   using IndexType = Index;
   using CoordinatesType = typename GridType::CoordinatesType;
   using GridEntityGetterType = GridEntityGetter< GridType, NeighborGridEntityType >;

   __cuda_callable__
   inline NeighborGridEntityGetter( const GridEntityType& entity ) : entity( entity ) {}

   template< int step >
   __cuda_callable__
   inline NeighborGridEntityType
   getEntity() const
   {
      TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "wrong coordinates" );
      TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
      TNL_ASSERT( entity.getCoordinates() + CoordinatesType( step ) >= CoordinatesType( 0 )
                     && entity.getCoordinates() + CoordinatesType( step ) < entity.getMesh().getDimensions(),
                  std::cerr << "entity.getCoordinates() = " << entity.getCoordinates() << " entity.getMesh().getDimensions() = "
                            << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension );
      return NeighborGridEntity( CoordinatesType( entity.getCoordinates().x() + step ) );
   }

   template< int step >
   __cuda_callable__
   inline IndexType
   getEntityIndex() const
   {
      TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "wrong coordinates" );
      TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
      TNL_ASSERT( entity.getCoordinates() + CoordinatesType( step ) >= CoordinatesType( 0 )
                     && entity.getCoordinates() + CoordinatesType( step ) < entity.getMesh().getDimensions(),
                  std::cerr << "entity.getCoordinates() = " << entity.getCoordinates() << " entity.getMesh().getDimensions() = "
                            << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension );
      return this->entity.getIndex() + step;
   }

   __cuda_callable__
   void
   refresh( const GridType& grid, const IndexType& entityIndex ){};

protected:
   const GridEntityType& entity;
};

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighborEntityDimension   |
 * +-----------------+---------------------------+
 * |       1         |              0            |
 * +-----------------+---------------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class NeighborGridEntityGetter<
   GridEntity< Meshes::Grid< 1, Real, Device, Index >, 1, Config >,
   0 >
{
public:
   static constexpr int EntityDimension = 1;
   static constexpr int NeighborEntityDimension = 0;
   using GridType = Meshes::Grid< 1, Real, Device, Index >;
   using GridEntityType = GridEntity< GridType, EntityDimension, Config >;
   using NeighborGridEntityType = GridEntity< GridType, NeighborEntityDimension, Config >;
   using RealType = Real;
   using IndexType = Index;
   using CoordinatesType = typename GridType::CoordinatesType;
   using GridEntityGetterType = GridEntityGetter< GridType, NeighborGridEntityType >;

   __cuda_callable__
   inline NeighborGridEntityGetter( const GridEntityType& entity ) : entity( entity ) {}

   template< int step >
   __cuda_callable__
   inline NeighborGridEntityType
   getEntity() const
   {
      TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "wrong coordinates" );
      TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
      TNL_ASSERT( entity.getCoordinates().x() + step + ( step < 0 ) >= CoordinatesType( 0 )
                     && entity.getCoordinates().x() + step + ( step < 0 ) <= entity.getMesh().getDimensions(),
                  std::cerr << "entity.getCoordinates() = " << entity.getCoordinates() << " entity.getMesh().getDimensions() = "
                            << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension );
      return NeighborGridEntity( CoordinatesType( entity.getCoordinates().x() + step + ( step < 0 ) ) );
   }

   template< int step >
   __cuda_callable__
   inline IndexType
   getEntityIndex() const
   {
      TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "wrong coordinates" );
      TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
      TNL_ASSERT( entity.getCoordinates().x() + step + ( step < 0 ) >= CoordinatesType( 0 ).x()
                     && entity.getCoordinates().x() + step + ( step < 0 ) <= entity.getMesh().getDimensions().x(),
                  std::cerr << "entity.getCoordinates() = " << entity.getCoordinates() << " entity.getMesh().getDimensions() = "
                            << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension );
      return this->entity.getIndex() + step + ( step < 0 );
   }

   __cuda_callable__
   void
   refresh( const GridType& grid, const IndexType& entityIndex ){};

protected:
   const GridEntityType& entity;

   // NeighborGridEntityGetter(){};
};

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighborEntityDimension   |
 * +-----------------+---------------------------+
 * |       0         |              1            |
 * +-----------------+---------------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class NeighborGridEntityGetter<
   GridEntity< Meshes::Grid< 1, Real, Device, Index >, 0, Config >,
   1>
{
public:
   static constexpr int EntityDimension = 0;
   static constexpr int NeighborEntityDimension = 1;
   using GridType = Meshes::Grid< 1, Real, Device, Index >;
   using GridEntityType = GridEntity< GridType, EntityDimension, Config >;
   using NeighborGridEntityType = GridEntity< GridType, NeighborEntityDimension, Config >;
   using RealType = Real;
   using IndexType = Index;
   using CoordinatesType = typename GridType::CoordinatesType;
   using GridEntityGetterType = GridEntityGetter< GridType, NeighborGridEntityType >;

   __cuda_callable__
   inline NeighborGridEntityGetter( const GridEntityType& entity ) : entity( entity ) {}

   void
   test() const
   {
      std::cerr << "***" << std::endl;
   };

   template< int step >
   __cuda_callable__
   inline NeighborGridEntityType
   getEntity() const
   {
      TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "wrong coordinates" );
      TNL_ASSERT_LE( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
      TNL_ASSERT( entity.getCoordinates().x() + step - ( step > 0 ) >= CoordinatesType( 0 )
                     && entity.getCoordinates().x() + step - ( step > 0 ) < entity.getMesh().getDimensions(),
                  std::cerr << "entity.getCoordinates() = " << entity.getCoordinates() << " entity.getMesh().getDimensions() = "
                            << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension );
      return NeighborGridEntity( CoordinatesType( entity.getCoordinates().x() + step - ( step > 0 ) ) );
   }

   template< int step >
   __cuda_callable__
   inline IndexType
   getEntityIndex() const
   {
      TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "wrong coordinates" );
      TNL_ASSERT_LE( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
      TNL_ASSERT( entity.getCoordinates().x() + step - ( step > 0 ) >= 0
                     && entity.getCoordinates().x() + step - ( step > 0 ) < entity.getMesh().getDimensions().x(),
                  std::cerr << "entity.getCoordinates() = " << entity.getCoordinates() << " entity.getMesh().getDimensions() = "
                            << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension );
      return this->entity.getIndex() + step - ( step > 0 );
   }

   __cuda_callable__
   void
   refresh( const GridType& grid, const IndexType& entityIndex ){};

protected:
   const GridEntityType& entity;
};

/****
 * +-----------------+---------------------------+
 * | EntityDimenions | NeighborEntityDimension   |
 * +-----------------+---------------------------+
 * |       0         |              0            |
 * +-----------------+---------------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class NeighborGridEntityGetter<
   GridEntity< Meshes::Grid< 1, Real, Device, Index >, 0, Config >,
   0 >
{
public:
   static constexpr int EntityDimension = 0;
   static constexpr int NeighborEntityDimension = 0;
   using GridType = Meshes::Grid< 1, Real, Device, Index >;
   using GridEntityType = GridEntity< GridType, EntityDimension, Config >;
   using NeighborGridEntityType = GridEntity< GridType, NeighborEntityDimension, Config >;
   using RealType = Real;
   using IndexType = Index;
   using CoordinatesType = typename GridType::CoordinatesType;
   using GridEntityGetterType = GridEntityGetter< GridType, NeighborGridEntityType >;

   __cuda_callable__
   inline NeighborGridEntityGetter( const GridEntityType& entity ) : entity( entity ) {}

   template< int step >
   __cuda_callable__
   inline NeighborGridEntityType
   getEntity() const
   {
      TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "wrong coordinates" );
      TNL_ASSERT_LE( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
      TNL_ASSERT( entity.getCoordinates().x() + step >= CoordinatesType( 0 )
                     && entity.getCoordinates().x() + step <= entity.getMesh().getDimensions(),
                  std::cerr << "entity.getCoordinates() = " << entity.getCoordinates() << " entity.getMesh().getDimensions() = "
                            << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension );
      return NeighborGridEntity( CoordinatesType( entity.getCoordinates().x() + step ) );
   }

   template< int step >
   __cuda_callable__
   inline IndexType
   getEntityIndex() const
   {
      TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "wrong coordinates" );
      TNL_ASSERT_LE( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
      TNL_ASSERT( entity.getCoordinates().x() + step >= CoordinatesType( 0 )
                     && entity.getCoordinates().x() + step <= entity.getMesh().getDimensions(),
                  std::cerr << "entity.getCoordinates() = " << entity.getCoordinates() << " entity.getMesh().getDimensions() = "
                            << entity.getMesh().getDimensions() << " EntityDimension = " << EntityDimension );

      return this->entity.getIndex() + step;
   }

   __cuda_callable__
   void
   refresh( const GridType& grid, const IndexType& entityIndex ){};

protected:
   const GridEntityType& entity;
};

}  // namespace Meshes
}  // namespace TNL
