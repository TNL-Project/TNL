// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once

#include <TNL/Containers/StaticVector.h>
#include <TNL/Meshes/GridDetails/NormalsGetter.h>
#include <TNL/Meshes/GridDetails/GridTraits.h>

namespace TNL {
namespace Meshes {

/***
 * \brief Structure holding all grid entities orientations.
 *
 * Grid is a orthogonal regular numerical mesh consisting of various mesh entities having various orientations. For example:
 *
 * 1. 1D-grid consists of:
 *    1. 0D vertexes,
 *    2. 1D cells.
 * 2. 2D-grid consists of:
 *    1. 0D vertexes,
 *    2. 1D faces going along x axis,
 *    3. 1D faces going along y axis,
 *    4. 2D cells.
 * 3. 3D-grid consists of:
 *    1. 0D vertexes,
 *    2. 1D edges going along x axis,
 *    3. 1D edges going along y axis,
 *    4. 1D edges going along z axis,
 *    5. 2D faces spanning along x and y axes,
 *    6. 2D faces spanning along x and z axes,
 *    7. 2D faces spanning along y and z axes,
 *    8. 3D cells.
 *
 * To be able to specify what kind of grid entities we aim to work with we have to encode somehow type and orientation of the entity.
 * Each entity can be given by a subset of vector of [standard basis](https://en.wikipedia.org/wiki/Standard_basis) of \f$R^n\f$ which the entity spans.
 * All the basis vectors generating the grid entity can be packed into one vector having ones at positions where the vectors of the standard basis
 * generating the grid entity have number one. See the following table for examples:
 *
 * | Grid dimension | Entity type              | Entity dimension  | Vectors of standard basis             | Packed vectors of standard basis |
 * |---------------:|-------------------------:|------------------:|--------------------------------------:|---------------------------------:|
 * | 1              | Vertex                   | 0                 | none or ( 0 )                         | ( 0 )                            |
 * | 1              | Cell                     | 1                 | ( 1 )                                 | ( 1 )                            |
 * |---------------:|-------------------------:|------------------:|--------------------------------------:|---------------------------------:|
 * | 2              | Vertex                   | 0                 | none or ( 0, 0 )                      | ( 0, 0 )                         |
 * | 2              | Face along x axis        | 1                 | ( 1, 0 )                              | ( 1, 0 )                         |
 * | 2              | Face along x axis        | 1                 | ( 0, 1 )                              | ( 0, 1 )                         |
 * | 2              | Cell                     | 2                 | ( 1, 0 ), ( 0, 1 )                    | ( 1, 1 )                         |
 * |---------------:|-------------------------:|------------------:|--------------------------------------:|---------------------------------:|
 * | 3              | Vertexes                 | 0                 | none or ( 0, 0, 0 )                   | ( 0, 0, 0 )                      |
 * | 3              | Edges along x axis       | 1                 | ( 1, 0, 0 )                           | ( 1, 0, 0 )                      |
 * | 3              | Edges along y axis       | 1                 | ( 0, 1, 0 )                           | ( 0, 1, 0 )                      |
 * | 3              | Edges along z axis       | 1                 | ( 0, 0, 1 )                           | ( 0, 0, 1 )                      |
 * | 3              | Faces along x and y axes | 2                 | ( 1, 0, 0 ), ( 0, 1, 0 )              | ( 1, 1, 0 )                      |
 * | 3              | Faces along x and z axes | 2                 | ( 1, 0, 0 ), ( 0, 0, 1 )              | ( 1, 0, 1 )                      |
 * | 3              | Faces along y and z axes | 2                 | ( 0, 1, 0 ), ( 0, 0, 1 )              | ( 0, 1, 1 )                      |
 * | 3              | Cells                    | 3                 | ( 1, 0, 0 ), ( 0, 1, 0 ), ( 0, 0, 1 ) | ( 1, 1, 1 )                      |
 *
 * Another way to encode the orientation is by vectors of the standard basis which are normal (or orthogonal) to standard basis vectors which the
 * grid entity spans. Clearly the normal vectors are complement to the basis vectors which the entity spans. Even the normal vectors can be packed
 * into one vector. See the following table for examples:
 *
 * | Grid dimension | Entity type              | Entity dimension  | Normal vectors of standard basis      | Packed  normal vectors  |
 * |---------------:|-------------------------:|------------------:|--------------------------------------:|------------------------:|
 * | 1              | Vertex                   | 0                 | ( 1 )                                 | ( 1 )                   |
 * | 1              | Cell                     | 1                 | none or ( 0 )                         | ( 0 )                   |
 * |---------------:|-------------------------:|------------------:|--------------------------------------:|------------------------:|
 * | 2              | Vertex                   | 0                 | ( 1, 0 ), ( 0, 1 )                    | ( 1, 1 )                |
 * | 2              | Face along x axis        | 1                 | ( 0, 1 )                              | ( 0, 1 )                |
 * | 2              | Face along x axis        | 1                 | ( 1, 0 )                              | ( 1, 0 )                |
 * | 2              | Cell                     | 2                 | none or ( 0, 0 )                      | ( 0, 0 )                |
 * |---------------:|-------------------------:|------------------:|--------------------------------------:|------------------------:|
 * | 3              | Vertexes                 | 0                 | ( 1, 0, 0 ), ( 0, 1, 0 ), ( 0, 0, 1 ) | ( 1, 1, 1 )             |
 * | 3              | Edges along x axis       | 1                 | ( 0, 1, 0 ), ( 0, 0, 1 )              | ( 0, 1, 1 )             |
 * | 3              | Edges along y axis       | 1                 | ( 1, 0, 0 ), ( 0, 0, 1 )              | ( 1, 0, 1 )             |
 * | 3              | Edges along z axis       | 1                 | ( 1, 0, 0 ), ( 0, 1, 0 )              | ( 1, 1, 0 )             |
 * | 3              | Faces along x and y axes | 2                 | ( 0, 0, 1 )                           | ( 0, 0, 1 )             |
 * | 3              | Faces along x and z axes | 2                 | ( 0, 1, 0 )                           | ( 0, 1, 0 )             |
 * | 3              | Faces along y and z axes | 2                 | ( 1, 0, 0 )                           | ( 1, 0, 0 )             |
 * | 3              | Cells                    | 3                 | none or ( 0, 0, 0 )                   | ( 0, 0, 0 )             |
 *
 * While the basis vectors generating the grid entity are useful for computing the center of the grid entity, for example, the normal
 * vectors are more suitable for computations of grid entities indexes - see \ref TNL::Meshes::Grid for more details.
 */
template< int GridDimension >
struct GridEntitiesOrientations
{
   using NormalsType = Containers::StaticVector< GridDimension, short int >;

   using OrientationNormalsContainer = Containers::StaticVector< 1 << GridDimension, NormalsType >;

   GridEntitiesOrientations() { addNormalsToTable< 0, 0 >( 0 ); }

   template< int EntityDimension >
   constexpr static int getOrientationsCount() { return combinationsCount( EntityDimension, GridDimension ); }

   template< int EntityDimension >
   constexpr static int getTotalOrientationsCount() { return cumulativeCombinationsCount( EntityDimension, GridDimension ); }

   template< int TotalOrientation >
   constexpr static int getEntityDimension() { return NormalsGetter< int, 0, GridDimension >::template getEntityDimension< TotalOrientation >(); }

   template< int EntityDimension, int... Normals >
   constexpr static int getOrientationIndex() { return NormalsGetter< int, EntityDimension, GridDimension >::template getOrientationIndex< Normals... >(); }

   template< int... Normals >
   constexpr static int getTotalOrientationIndex() { return NormalsGetter< int, 0, GridDimension >::template getTotalOrientationIndex< Normals... >(); }

   template< int EntityDimension >
   constexpr static int getTotalOrientationIndex( int orientation ) { return getTotalOrientationsCount< EntityDimension - 1 >() + orientation; }

   template< int EntityDimension, int Orientation >
   static NormalsType getNormals() { return NormalsGetter< int, EntityDimension, GridDimension >::template getNormals< Orientation >(); }

   template< int TotalOrientation >
   static NormalsType getNormals() {
      static_assert( TotalOrientation >= 0 && TotalOrientation < ( 1 << GridDimension ), "Wrong index of total orientation." );
      return NormalsGetter< int, 0, GridDimension >::template getNormalsByTotalOrientation< TotalOrientation >();
   };

   template< int EntityDimension >
   NormalsType getNormals( int orientation ) { return normalsTable[ getTotalOrientationIndex< EntityDimension >( orientation ) ]; }


   NormalsType getNormals( int totalOrientation ) { return normalsTable[ totalOrientation ]; }

protected:

   template< int EntityDimension, int Orientation >
   void addNormalsToTable( int offset ) {
      normalsTable[ offset ] = getNormals< EntityDimension, Orientation >();
      if constexpr( Orientation < getOrientationsCount< EntityDimension >() - 1 )
         addNormalsToTable< EntityDimension, Orientation + 1 >( offset + 1 );
      else if constexpr( EntityDimension < GridDimension )
         addNormalsToTable< EntityDimension + 1, 0 >( offset + 1 );
   }

   OrientationNormalsContainer normalsTable;

};

} //namespace Meshes
} //namespace TNL
