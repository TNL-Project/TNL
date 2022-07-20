
#pragma once

#include <TNL/Meshes/GridDetails/GridEntityGetter.h>
#include <TNL/Meshes/GridEntity.h>

//template< int Dimension, typename Real, typename Device, typename Index >
//class Grid;

namespace TNL {
namespace Meshes {

/****
 * 1D grid
 */
template< typename Real, typename Device, typename Index, int EntityDimension >
class GridEntityGetter< Meshes::NDGrid< 1, Real, Device, Index >, EntityDimension >
{
public:
   static constexpr int entityDimension = EntityDimension;

   using NDGrid = Meshes::NDGrid< 1, Real, Device, Index >;
   using Entity = GridEntity< Grid< 1, Real, Device, Index >, EntityDimension >;
   using Coordinate = typename NDGrid::CoordinatesType;

   __cuda_callable__
   inline static Index
   getEntityIndex( const NDGrid& grid, const Entity& entity )
   {
      TNL_ASSERT_GE( entity.getCoordinates(), Coordinate( 0 ), "wrong coordinates" );
      TNL_ASSERT_LT( entity.getCoordinates(), grid.getDimensions() + entity.getBasis(), "wrong coordinates" );

      return entity.getCoordinates().x();
   }
};

/****
 * 2D grid
 */
template< typename Real, typename Device, typename Index >
class GridEntityGetter< Meshes::NDGrid< 2, Real, Device, Index >, 2 >
{
public:
   static constexpr int entityDimension = 2;

   using NDGrid = Meshes::NDGrid< 2, Real, Device, Index >;
   using Entity = GridEntity< Grid< 2, Real, Device, Index >, entityDimension >;
   using Coordinate = typename NDGrid::CoordinatesType;

   __cuda_callable__
   inline static Index
   getEntityIndex( const NDGrid& grid, const Entity& entity )
   {
      TNL_ASSERT_GE( entity.getCoordinates(), Coordinate( 0, 0 ), "wrong coordinates" );
      TNL_ASSERT_LT( entity.getCoordinates(), grid.getDimensions() + entity.getBasis(), "wrong coordinates" );

      return entity.getCoordinates().y() * grid.getDimensions().x() + entity.getCoordinates().x();
   }
};

template< typename Real, typename Device, typename Index >
class GridEntityGetter< Meshes::NDGrid< 2, Real, Device, Index >, 1 >
{
public:
   static constexpr int entityDimension = 1;

   using NDGrid = Meshes::NDGrid< 2, Real, Device, Index >;
   using Entity = GridEntity< Grid< 2, Real, Device, Index >, entityDimension >;
   using Coordinate = typename NDGrid::CoordinatesType;

   __cuda_callable__
   inline static Index
   getEntityIndex( const NDGrid& grid, const Entity& entity )
   {
      TNL_ASSERT_GE( entity.getCoordinates(), Coordinate( 0, 0 ), "wrong coordinates" );
      TNL_ASSERT_LT( entity.getCoordinates(), grid.getDimensions() + entity.getBasis(), "wrong coordinates" );

      const Coordinate& coordinates = entity.getCoordinates();
      const Coordinate& dimensions = grid.getDimensions();

      if( entity.getOrientation() == 0 )
         return coordinates.y() * ( dimensions.x() ) + coordinates.x();

      return grid.template getOrientedEntitiesCount< 1, 0 >() + coordinates.y() * ( dimensions.x() + 1 ) + coordinates.x();
   }
};

template< typename Real, typename Device, typename Index >
class GridEntityGetter< Meshes::NDGrid< 2, Real, Device, Index >, 0 >
{
public:
   static constexpr int entityDimension = 0;

   using NDGrid = Meshes::NDGrid< 2, Real, Device, Index >;
   using Entity = GridEntity< Grid< 2, Real, Device, Index >, entityDimension >;
   using Coordinate = typename NDGrid::CoordinatesType;

   __cuda_callable__
   inline static Index
   getEntityIndex( const NDGrid& grid, const Entity& entity )
   {
      TNL_ASSERT_GE( entity.getCoordinates(), Coordinate( 0, 0 ), "wrong coordinates" );
      TNL_ASSERT_LT( entity.getCoordinates(), grid.getDimensions() + entity.getBasis(), "wrong coordinates" );

      const Coordinate& coordinates = entity.getCoordinates();
      const Coordinate& dimensions = grid.getDimensions();

      return coordinates.y() * ( dimensions.x() + 1 ) + coordinates.x();
   }
};

/****
 * 3D grid
 */
template< typename Real, typename Device, typename Index >
class GridEntityGetter< Meshes::NDGrid< 3, Real, Device, Index >, 3 >
{
public:
   static constexpr int entityDimension = 3;

   using NDGrid = Meshes::NDGrid< 3, Real, Device, Index >;
   using Entity = GridEntity< Grid< 3, Real, Device, Index >, entityDimension >;
   using Coordinate = typename NDGrid::CoordinatesType;

   __cuda_callable__
   inline static Index
   getEntityIndex( const NDGrid& grid, const Entity& entity )
   {
      TNL_ASSERT_GE( entity.getCoordinates(), Coordinate( 0, 0, 0 ), "wrong coordinates" );
      TNL_ASSERT_LT( entity.getCoordinates(), grid.getDimensions() + entity.getBasis(), "wrong coordinates" );

      const Coordinate coordinates = entity.getCoordinates();
      const Coordinate dimensions = grid.getDimensions();

      return ( coordinates.z() * dimensions.y() + coordinates.y() ) * dimensions.x() + coordinates.x();
   }
};

template< typename Real, typename Device, typename Index >
class GridEntityGetter< Meshes::NDGrid< 3, Real, Device, Index >, 2 >
{
public:
   static constexpr int entityDimension = 2;

   using NDGrid = Meshes::NDGrid< 3, Real, Device, Index >;
   using Entity = GridEntity< Grid< 3, Real, Device, Index >, entityDimension >;
   using Coordinate = typename NDGrid::CoordinatesType;

   __cuda_callable__
   inline static Index
   getEntityIndex( const NDGrid& grid, const Entity& entity )
   {
      TNL_ASSERT_GE( entity.getCoordinates(), Coordinate( 0, 0, 0 ), "wrong coordinates" );
      TNL_ASSERT_LT( entity.getCoordinates(), grid.getDimensions() + entity.getBasis(), "wrong coordinates" );

      const Coordinate& coordinates = entity.getCoordinates();
      const Coordinate& dimensions = grid.getDimensions();

      if( entity.getOrientation() == 0 )
         return ( coordinates.z() * dimensions.y() + coordinates.y() ) * ( dimensions.x() ) + coordinates.x();

      if( entity.getOrientation() == 1 )
         return grid.template getOrientedEntitiesCount< 2, 0 >()
              + ( coordinates.z() * ( dimensions.y() + 1 ) + coordinates.y() ) * dimensions.x() + coordinates.x();

      return grid.template getOrientedEntitiesCount< 2, 1 >() + grid.template getOrientedEntitiesCount< 2, 0 >()
           + ( coordinates.z() * dimensions.y() + coordinates.y() ) * ( dimensions.x() + 1 ) + coordinates.x();
   }
};

template< typename Real, typename Device, typename Index >
class GridEntityGetter< Meshes::NDGrid< 3, Real, Device, Index >, 1 >
{
public:
   static constexpr int entityDimension = 1;

   using NDGrid = Meshes::NDGrid< 3, Real, Device, Index >;
   using Entity = GridEntity< Grid< 3, Real, Device, Index >, entityDimension >;
   using Coordinate = typename NDGrid::CoordinatesType;

   __cuda_callable__
   inline static Index
   getEntityIndex( const NDGrid& grid, const Entity& entity )
   {
      TNL_ASSERT_GE( entity.getCoordinates(), Coordinate( 0, 0, 0 ), "wrong coordinates" );
      TNL_ASSERT_LT( entity.getCoordinates(), grid.getDimensions() + entity.getBasis(), "wrong coordinates" );

      const Coordinate& coordinates = entity.getCoordinates();
      const Coordinate& dimensions = grid.getDimensions();

      if( entity.getOrientation() == 0 )
         return ( coordinates.z() * ( dimensions.y() + 1 ) + coordinates.y() ) * dimensions.x() + coordinates.x();

      if( entity.getOrientation() == 1 )
         return grid.template getOrientedEntitiesCount< 1, 0 >()
              + ( coordinates.z() * dimensions.y() + coordinates.y() ) * ( dimensions.x() + 1 ) + coordinates.x();

      return grid.template getOrientedEntitiesCount< 1, 1 >() + grid.template getOrientedEntitiesCount< 1, 0 >()
           + ( coordinates.z() * ( dimensions.y() + 1 ) + coordinates.y() ) * ( dimensions.x() + 1 ) + coordinates.x();
   }
};

template< typename Real, typename Device, typename Index >
class GridEntityGetter< Meshes::NDGrid< 3, Real, Device, Index >, 0 >
{
public:
   static constexpr int entityDimension = 0;

   using NDGrid = Meshes::NDGrid< 3, Real, Device, Index >;
   using Entity = GridEntity< Grid< 3, Real, Device, Index >, entityDimension >;
   using Coordinate = typename NDGrid::CoordinatesType;

   __cuda_callable__
   inline static Index
   getEntityIndex( const NDGrid& grid, const Entity& entity )
   {
      TNL_ASSERT_GE( entity.getCoordinates(), Coordinate( 0, 0, 0 ), "wrong coordinates" );
      TNL_ASSERT_LT( entity.getCoordinates(), grid.getDimensions() + entity.getBasis(), "wrong coordinates" );

      const Coordinate coordinates = entity.getCoordinates();
      const Coordinate dimensions = grid.getDimensions();

      return ( coordinates.z() * ( dimensions.y() + 1 ) + coordinates.y() ) * ( dimensions.x() + 1 ) + coordinates.x();
   }
};

}  // namespace Meshes
}  // namespace TNL
