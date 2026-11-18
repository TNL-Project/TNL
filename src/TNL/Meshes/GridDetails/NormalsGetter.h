// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

#include <TNL/Assert.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/DiscreteMath.h>

namespace TNL::Meshes {

namespace detail {

/**
 * \brief Returns the normals of the grid entity orientation with the given index.
 *
 * The normals of entities with dimension `entityDimension` in a grid with dimension `GridDimension` are all
 * vectors of length `GridDimension` consisting of `GridDimension - entityDimension` ones and `entityDimension`
 * zeros. The orientation index is the position of the normals in the descending lexicographic order of these
 * vectors. For example, the normals of edges in a 3D grid are ordered as `( 1, 1, 0 )`, `( 1, 0, 1 )` and
 * `( 0, 1, 1 )`.
 *
 * The descending lexicographic order is the order of binary numbers from the largest to the smallest, where
 * the first element of the vector is the most significant bit. Therefore, we go through all binary numbers
 * with `GridDimension` bits starting from the largest one and count those with the right number of ones.
 *
 * \tparam GridDimension is the dimension of the grid.
 * \param entityDimension is the dimension of the grid entity.
 * \param orientation is the orientation index of the grid entity.
 * \return The normals or a vector of zeros if the orientation index is out of range.
 */
template< int GridDimension >
constexpr std::array< int, GridDimension >
getOrientationNormals( int entityDimension, int orientation )
{
   std::array< int, GridDimension > normals{};
   int currentOrientation = 0;
   for( int bits = ( 1 << GridDimension ) - 1; bits >= 0; bits-- ) {
      int onesCount = 0;
      for( int i = 0; i < GridDimension; i++ )
         onesCount += ( bits >> i ) & 1;
      if( onesCount != GridDimension - entityDimension )
         continue;
      if( currentOrientation == orientation ) {
         for( int i = 0; i < GridDimension; i++ )
            normals[ i ] = ( bits >> ( GridDimension - 1 - i ) ) & 1;
         return normals;
      }
      currentOrientation++;
   }
   return normals;
}

/**
 * \brief Returns the orientation index of the grid entity with the given normals.
 *
 * This is the inverse of \ref getOrientationNormals.
 *
 * \tparam GridDimension is the dimension of the grid.
 * \param entityDimension is the dimension of the grid entity.
 * \param normals are the normals of the grid entity.
 * \return The orientation index or -1 if the normals do not belong to an entity with the given dimension.
 */
template< int GridDimension >
constexpr int
getOrientationIndex( int entityDimension, const std::array< int, GridDimension >& normals )
{
   for( int orientation = 0; orientation < combinationsCount( entityDimension, GridDimension ); orientation++ ) {
      const std::array< int, GridDimension > orientationNormals =
         getOrientationNormals< GridDimension >( entityDimension, orientation );
      bool equal = true;
      for( int i = 0; i < GridDimension; i++ )
         equal = equal && orientationNormals[ i ] == normals[ i ];
      if( equal )
         return orientation;
   }
   return -1;
}

}  // namespace detail

template< typename Index, int EntityDimension, int GridDimension >
struct NormalsGetter
{
   static_assert( EntityDimension >= 0 && EntityDimension <= GridDimension, "Wrong entity dimension." );

public:
   constexpr static int
   getOrientationsCount()
   {
      return combinationsCount( EntityDimension, GridDimension );
   }
   using NormalsType = TNL::Containers::StaticVector< GridDimension, short int >;
   using OrientationNormalsContainer =
      TNL::Containers::StaticVector< combinationsCount( EntityDimension, GridDimension ), NormalsType >;

   template< int TotalOrientation >
   [[nodiscard]] constexpr static int
   getEntityDimension()
   {
      static_assert(
         TotalOrientation < getOrientationsCount() || EntityDimension < GridDimension, "Wrong number of total orientation." );
      if constexpr( TotalOrientation < getOrientationsCount() )
         return EntityDimension;
      else if constexpr( EntityDimension < GridDimension )
         return NormalsGetter< int, EntityDimension + 1, GridDimension >::template getEntityDimension<
            TotalOrientation - getOrientationsCount() >();
      else {
         return -1;
      }
   }

   template< int Orientation >
   constexpr static NormalsType
   getNormals()
   {
      static_assert( Orientation >= 0 && Orientation < getOrientationsCount(), "Wrong orientation index." );
      constexpr std::array< int, GridDimension > normals =
         detail::getOrientationNormals< GridDimension >( EntityDimension, Orientation );
      return NormalsType( normals );
   }

   NormalsType
   getNormals( int orientation )
   {
      TNL_ASSERT_GE( orientation, 0, "Wrong index of orientation." );
      TNL_ASSERT_LT( orientation, getOrientationsCount(), "Wrong index of orientation." );
      return NormalsType( detail::getOrientationNormals< GridDimension >( EntityDimension, orientation ) );
   }

   template< int... Normals >
   constexpr static int
   getOrientationIndex()
   {
      static_assert( sizeof...( Normals ) == GridDimension, "The size of normals must be equal to grid dimension." );
      constexpr int orientation =
         detail::getOrientationIndex< GridDimension >( EntityDimension, std::array< int, GridDimension >{ Normals... } );
      return orientation;
   }

   template< int... Normals >
   constexpr static int
   getTotalOrientationIndex()
   {
      static_assert( sizeof...( Normals ) == GridDimension, "The size of normals must be equal to grid dimension." );
      constexpr int idx = NormalsGetter< Index, EntityDimension, GridDimension >::template getOrientationIndex< Normals... >();
      if constexpr( idx < 0 ) {
         if constexpr( EntityDimension < GridDimension )
            return NormalsGetter< Index, EntityDimension + 1, GridDimension >::template getTotalOrientationIndex< Normals... >()
                 + getOrientationsCount();
         else
            return -1;
      }
      return idx;
   }

   template< int TotalOrientation >
   constexpr static NormalsType
   getNormalsByTotalOrientation()
   {
      if constexpr( TotalOrientation < getOrientationsCount() )
         return NormalsGetter< Index, EntityDimension, GridDimension >::getNormals< TotalOrientation >();
      else
         return NormalsGetter< Index, EntityDimension + 1, GridDimension >::template getNormalsByTotalOrientation<
            TotalOrientation - getOrientationsCount() >();
   }
};

}  // namespace TNL::Meshes
