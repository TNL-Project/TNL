
#pragma once

#include <TNL/Meshes/GridDetails/GridEntityGetter.h>

namespace TNL {
namespace Meshes {

/****
 * 1D grid
 */
template< typename Real, typename Device, typename Index, int EntityDimension >
class GridEntityGetter< Meshes::Grid< 1, Real, Device, Index >, EntityDimension >
{
public:
   static constexpr int entityDimension = EntityDimension;

   using Grid = Meshes::Grid< 1, Real, Device, Index >;
   using Entity = GridEntity< Grid, EntityDimension >;
   using Coordinate = typename Grid::Coordinate;

   __cuda_callable__
   inline static Index
   getEntityIndex( const Grid& grid, const Entity& entity )
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
class GridEntityGetter< Meshes::Grid< 2, Real, Device, Index >, 2 >
{
public:
   static constexpr int entityDimension = 2;

   using Grid = Meshes::Grid< 2, Real, Device, Index >;
   using Entity = GridEntity< Grid, entityDimension >;
   using Coordinate = typename Grid::Coordinate;

   __cuda_callable__
   inline static Index
   getEntityIndex( const Grid& grid, const Entity& entity )
   {
      TNL_ASSERT_GE( entity.getCoordinates(), Coordinate( 0, 0 ), "wrong coordinates" );
      TNL_ASSERT_LT( entity.getCoordinates(), grid.getDimensions() + entity.getBasis(), "wrong coordinates" );

      return entity.getCoordinates().y() * grid.getDimensions().x() + entity.getCoordinates().x();
   }
};

template< typename Real, typename Device, typename Index >
class GridEntityGetter< Meshes::Grid< 2, Real, Device, Index >, 1 >
{
public:
   static constexpr int entityDimension = 1;

   using Grid = Meshes::Grid< 2, Real, Device, Index >;
   using Entity = GridEntity< Grid, entityDimension >;
   using Coordinate = typename Grid::Coordinate;

   __cuda_callable__
   inline static Index
   getEntityIndex( const Grid& grid, const Entity& entity )
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
class GridEntityGetter< Meshes::Grid< 2, Real, Device, Index >, 0 >
{
public:
   static constexpr int entityDimension = 0;

   using Grid = Meshes::Grid< 2, Real, Device, Index >;
   using Entity = GridEntity< Grid, entityDimension >;
   using Coordinate = typename Grid::Coordinate;

   __cuda_callable__
   inline static Index
   getEntityIndex( const Grid& grid, const Entity& entity )
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
class GridEntityGetter< Meshes::Grid< 3, Real, Device, Index >, 3 >
{
public:
   static constexpr int entityDimension = 3;

   using Grid = Meshes::Grid< 3, Real, Device, Index >;
   using Entity = GridEntity< Grid, entityDimension >;
   using Coordinate = typename Grid::Coordinate;

   __cuda_callable__
   inline static Index
   getEntityIndex( const Grid& grid, const Entity& entity )
   {
      TNL_ASSERT_GE( entity.getCoordinates(), Coordinate( 0, 0, 0 ), "wrong coordinates" );
      TNL_ASSERT_LT( entity.getCoordinates(), grid.getDimensions() + entity.getBasis(), "wrong coordinates" );

      const Coordinate coordinates = entity.getCoordinates();
      const Coordinate dimensions = grid.getDimensions();

      return ( coordinates.z() * dimensions.y() + coordinates.y() ) * dimensions.x() + coordinates.x();
   }
};

template< typename Real, typename Device, typename Index >
class GridEntityGetter< Meshes::Grid< 3, Real, Device, Index >, 2 >
{
public:
   static constexpr int entityDimension = 2;

   using Grid = Meshes::Grid< 3, Real, Device, Index >;
   using Entity = GridEntity< Grid, entityDimension >;
   using Coordinate = typename Grid::Coordinate;

   __cuda_callable__
   inline static Index
   getEntityIndex( const Grid& grid, const Entity& entity )
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
class GridEntityGetter< Meshes::Grid< 3, Real, Device, Index >, 1 >
{
public:
   static constexpr int entityDimension = 1;

   using Grid = Meshes::Grid< 3, Real, Device, Index >;
   using Entity = GridEntity< Grid, entityDimension >;
   using Coordinate = typename Grid::Coordinate;

   __cuda_callable__
   inline static Index
   getEntityIndex( const Grid& grid, const Entity& entity )
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
class GridEntityGetter< Meshes::Grid< 3, Real, Device, Index >, 0 >
{
public:
   static constexpr int entityDimension = 0;

   using Grid = Meshes::Grid< 3, Real, Device, Index >;
   using Entity = GridEntity< Grid, entityDimension >;
   using Coordinate = typename Grid::Coordinate;

   __cuda_callable__
   inline static Index
   getEntityIndex( const Grid& grid, const Entity& entity )
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
