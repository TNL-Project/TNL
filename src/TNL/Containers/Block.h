// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <set>
#include <stdexcept>
#include <vector>

#include "StaticVector.h"

namespace TNL::Containers {

/**
 * \brief Minimal class representing a block of an equidistant D-dimensional
 * lattice.
 */
template< int D = 3, typename idx_ = int >
struct Block
{
   using idx = idx_;
   using CoordinatesType = StaticVector< D, idx >;

   //! \brief Dimension of the lattice.
   static constexpr int dimension = D;

   //! \brief Offset of the block on the global lattice.
   CoordinatesType begin = 0;

   //! \brief Ending point of the block on the global lattice. The ending
   //! point is __not__ included in the block, i.e., the block spans the
   //! multidimensional interval `[begin, end)`.
   CoordinatesType end = 0;

   //! \brief Default constructor.
   Block() = default;

   //! \brief Constructs a block from the given \e begin and \e end.
   Block( const CoordinatesType& begin, const CoordinatesType& end ) : begin( begin ), end( end ) {}
};

/**
 * \brief Writes a textual representation of the given multiindex into the given
 * stream.
 */
template< int D, typename idx >
std::ostream&
operator<<( std::ostream& str, const Block< D, idx >& block )
{
   return str << "( " << block.begin << ", " << block.end << " )";
}

/**
 * \brief Compares two blocks point-wise.
 */
template< int D, typename idx >
bool
operator==( const Block< D, idx >& left, const Block< D, idx >& right )
{
   return left.begin == right.begin && left.end == right.end;
}

/**
 * \brief Compares two blocks point-wise.
 */
template< int D, typename idx >
bool
operator!=( const Block< D, idx >& left, const Block< D, idx >& right )
{
   return ! ( left == right );
}

/**
 * \brief Lexicographically compares two blocks by joining their \e begin and
 * \e end points.
 */
template< int D, typename idx >
bool
operator<( const Block< D, idx >& left, const Block< D, idx >& right )
{
   if( left.begin < right.begin ) {
      return true;
   }
   if( left.begin == right.begin ) {
      return left.end < right.end;
   }
   return false;
}

/**
 * \brief Calculates the volume of a D-dimensional block.
 */
template< int D, typename idx >
idx
getVolume( const Block< D, idx >& block )
{
   return product( abs( block.end - block.begin ) );
}

/**
 * \brief Calculates the volume occupied by blocks in a decomposition.
 */
template< int D, typename idx >
idx
getVolume( const std::vector< Block< D, idx > >& decomposition )
{
   idx result = 0;
   for( const auto& block : decomposition )
      result += getVolume( block );
   return result;
}

/**
 * \brief Calculates the maximum imbalance of blocks in a decomposition.
 *
 * Imbalance is a non-negative quantity that measures how much the block's
 * volume differs from the ideal, average volume. If all blocks have the ideal
 * volume, then all blocks have zero imbalance.
 *
 * \param decomposition Vector of blocks in the decomposition.
 * \param global_volume Optional pre-computed volume of all blocks.
 */
template< typename idx >
double
getMaximumImbalance( const std::vector< Block< 3, idx > >& decomposition, idx global_volume = 0 )
{
   if( global_volume == 0 )
      global_volume = getVolume( decomposition );

   double max_imbalance = 0;
   for( const auto& block : decomposition ) {
      const double imbalance = decomposition.size() * getVolume( block ) / double( global_volume ) - 1.0;
      max_imbalance = std::max( max_imbalance, std::abs( imbalance ) );
   }

   return max_imbalance;
}

/**
 * \brief Calculates the area of a 2D block in 3D space.
 *
 * \param block The block whose area will be calculated.
 * \param axes_weights Optional weights that modify the area based on the block orientation.
 *                     `axes_weights.x()` multiplies the area of blocks normal to the x-axis,
 *                     `axes_weights.y()` multiplies the area of blocks normal to the y-axis, and
 *                     `axes_weights.z()` multiplies the area of blocks normal to the z-axis.
 */
template< typename idx >
idx
getArea( const Block< 3, idx >& block, const StaticVector< 3, idx >& axes_weights = { 1, 1, 1 } )
{
   if( block.begin.x() == block.end.x() ) {
      return std::abs( block.end.y() - block.begin.y() ) * std::abs( block.end.z() - block.begin.z() ) * axes_weights.x();
   }
   if( block.begin.y() == block.end.y() ) {
      return std::abs( block.end.x() - block.begin.x() ) * std::abs( block.end.z() - block.begin.z() ) * axes_weights.y();
   }
   if( block.begin.z() == block.end.z() ) {
      return std::abs( block.end.x() - block.begin.x() ) * std::abs( block.end.y() - block.begin.y() ) * axes_weights.z();
   }
   throw std::logic_error( "3D block passed to the area function is not a 2D "
                           "object (it has a non-zero volume)" );
}

/**
 * \brief Creates the sides of a 3D block and adds them to the output vector.
 */
template< typename idx, typename OutputIterator >
void
createSides( const Block< 3, idx >& block, OutputIterator output )
{
   // each block has 6 sides, we need 8 points to define them
   using idx3d = typename Block< 3, idx >::CoordinatesType;
   const idx3d point_bbb = block.begin;
   const idx3d point_eeb( block.end.x(), block.end.y(), block.begin.z() );
   const idx3d point_ebe( block.end.x(), block.begin.y(), block.end.z() );
   const idx3d point_bee( block.begin.x(), block.end.y(), block.end.z() );
   const idx3d point_ebb( block.end.x(), block.begin.y(), block.begin.z() );
   const idx3d point_beb( block.begin.x(), block.end.y(), block.begin.z() );
   const idx3d point_bbe( block.begin.x(), block.begin.y(), block.end.z() );
   const idx3d point_eee = block.end;

   // bottom
   *output++ = Block< 3, idx >{ point_bbb, point_eeb };
   // front
   *output++ = Block< 3, idx >{ point_bbb, point_ebe };
   // left
   *output++ = Block< 3, idx >{ point_bbb, point_bee };
   // right
   *output++ = Block< 3, idx >{ point_ebb, point_eee };
   // back
   *output++ = Block< 3, idx >{ point_beb, point_eee };
   // top
   *output++ = Block< 3, idx >{ point_bbe, point_eee };
}

/**
 * \brief Creates all unique sides of blocks in a 3D decomposition.
 */
template< typename idx >
std::set< Block< 3, idx > >
createSides( const std::vector< Block< 3, idx > >& decomposition )
{
   // insert directly into a set via std::inserter
   std::set< Block< 3, idx > > sides;
   for( const auto& block : decomposition )
      createSides( block, std::inserter( sides, sides.end() ) );
   return sides;
}

/**
 * \brief Creates all unique sides of blocks in a 3D decomposition.
 */
template< typename idx >
std::set< Block< 3, idx > >
createInteriorSides( const std::vector< Block< 3, idx > >& decomposition )
{
   // first insert all sides into a vector
   std::vector< Block< 3, idx > > sides;
   for( const auto& block : decomposition )
      createSides( block, std::inserter( sides, sides.end() ) );

   // find duplicate sides
   std::set< Block< 3, idx > > visited;
   std::set< Block< 3, idx > > duplicates;
   for( const auto& side : sides ) {
      const auto& [ _, inserted ] = visited.insert( side );
      if( ! inserted )
         duplicates.insert( side );
   }

   return duplicates;
}

/**
 * \brief Calculates the total area of interior sides in a 3D decomposition.
 *
 * \param decomposition A vector of blocks that form a 3D decomposition of a large block.
 * \param axes_weights Optional weights that modify interface areas based on the interior side orientation.
 *                     `axes_weights.x()` multiplies the area of sides normal to the x-axis,
 *                     `axes_weights.y()` multiplies the area of sides normal to the y-axis, and
 *                     `axes_weights.z()` multiplies the area of sides normal to the z-axis.
 */
template< typename idx >
idx
getInterfaceArea( const std::vector< Block< 3, idx > >& decomposition,
                  const StaticVector< 3, idx >& axes_weights = { 1, 1, 1 } )
{
   idx result = 0;
   for( const auto& side : createInteriorSides( decomposition ) )
      result += getArea( side, axes_weights );
   return result;
}

}  // namespace TNL::Containers
