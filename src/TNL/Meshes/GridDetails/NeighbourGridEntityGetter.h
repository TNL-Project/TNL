// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend/Macros.h>
#include <TNL/Meshes/GridDetails/NormalsGetter.h>

namespace TNL::Meshes {

template< class, int >
class GridEntity;

template< int GridDimension, int ParentEntityDimension, int NeighbourEntityDimension >
class NeighbourGridEntityGetter
{
public:
   template< class Grid >
   [[nodiscard]] __cuda_callable__
   static GridEntity< Grid, NeighbourEntityDimension >
   getEntity( const GridEntity< Grid, ParentEntityDimension >& entity, const typename Grid::CoordinatesType& offset )
   {
      using CoordinatesType = typename Grid::CoordinatesType;

      constexpr int orientationsCount = combinationsCount( NeighbourEntityDimension, GridDimension );

      const CoordinatesType coordinate = entity.getCoordinates() + offset;
      const int orientation = TNL::min( orientationsCount - 1, entity.getOrientation().getIndex() );
      const CoordinatesType normals =
         orientation == entity.getOrientation().getIndex() && ParentEntityDimension == NeighbourEntityDimension
            ? entity.getNormals()
            : entity.getMesh().template getNormals< NeighbourEntityDimension >( orientation );

      TNL_ASSERT_ALL_GE( coordinate, 0, "wrong coordinate" );
      TNL_ASSERT_ALL_LT( coordinate, entity.getMesh().getDimensions() + normals, "wrong coordinate" );

      return { entity.getMesh(), coordinate, normals, orientation };
   }

   template< class Grid >
   static __cuda_callable__
   inline GridEntity< Grid, NeighbourEntityDimension >
   getEntityIndex( const GridEntity< Grid, ParentEntityDimension >& entity, const typename Grid::CoordinatesType& offset )
   {
      if constexpr( ParentEntityDimension == Grid::getMeshDimension() && ParentEntityDimension == NeighbourEntityDimension )
      {
         TNL_ASSERT_GE( entity.getCoordinates() + offset, Grid::CoordinatesType( 0 ), "Wrong coordinates of neighbour entity." );
         TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "Wrong coordinates of neighbour entity." );

         //Algorithms::staticFor< IndexType, 0,
         return entity.getIndex() + ( offset, entity.getMesh().getDimensions() );
      }
      else
      {
         using NormalsGetterType = NormalsGetter< typename Grid::IndexType, NeighbourEntityDimension, GridDimension >;
         using CoordinatesType = typename Grid::CoordinatesType;

         constexpr int orientationsCount = combinationsCount( NeighbourEntityDimension, GridDimension );

         const CoordinatesType coordinate = entity.getCoordinates() + offset;
         const int orientation = TNL::min( orientationsCount - 1, entity.getOrientation().getIndex() );
         const CoordinatesType normals =
            orientation == entity.getOrientation().getIndex() && ParentEntityDimension == NeighbourEntityDimension
               ? entity.getNormals()
               : entity.getMesh().template getNormals< NeighbourEntityDimension >( orientation );

         TNL_ASSERT_GE( coordinate, CoordinatesType( 0 ), "wrong coordinate" );
         TNL_ASSERT_LT( coordinate, entity.getMesh().getDimensions() + normals, "wrong coordinate" );

         return { entity.getMesh(), coordinate, normals, orientation };
      }
   }


   template< class Grid,
             int Orientation,
             std::enable_if_t< Templates::isInLeftClosedRightOpenInterval(
                                  0,
                                  Orientation,
                                  combinationsCount( NeighbourEntityDimension, GridDimension ) ),
                               bool > = true >
   [[nodiscard]] __cuda_callable__
   static GridEntity< Grid, NeighbourEntityDimension >
   getEntity( const GridEntity< Grid, ParentEntityDimension >& entity, const typename Grid::CoordinatesType& offset )
   {
      using NormalsGetterType = NormalsGetter< typename Grid::IndexType, NeighbourEntityDimension, GridDimension >;
      using CoordinatesType = typename Grid::CoordinatesType;

      const CoordinatesType coordinate = entity.getCoordinates() + offset;
      const CoordinatesType normals = NormalsGetterType::template getNormals< Orientation >();

      TNL_ASSERT_ALL_GE( coordinate, 0, "wrong coordinate" );
      TNL_ASSERT_ALL_LT( coordinate, entity.getMesh().getDimensions() + normals, "wrong coordinate" );

      return { entity.getMesh(), coordinate, normals, Orientation };
   }
};

}  // namespace TNL::Meshes
