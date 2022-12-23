// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once

#include <TNL/Meshes/GridEntitiesOrientations.h>

namespace TNL {
namespace Meshes {


template< int GridDimension >
GridEntitiesOrientations< GridDimension >::
   GridEntitiesOrientations() { addNormalsToTable< 0, 0 >( 0 ); }

template< int GridDimension >
   template< int EntityDimension >
constexpr int
GridEntitiesOrientations< GridDimension >::
getOrientationsCount() {
   return combinationsCount( EntityDimension, GridDimension );
}

template< int GridDimension >
constexpr int
GridEntitiesOrientations< GridDimension >::
getTotalOrientationsCount() {
   return cumulativeCombinationsCount( GridDimension, GridDimension );
}

template< int GridDimension >
   template< int TotalOrientation >
constexpr int
GridEntitiesOrientations< GridDimension >::
getEntityDimension() {
   return NormalsGetter< int, 0, GridDimension >::template getEntityDimension< TotalOrientation >();
}

template< int GridDimension >
   template< int EntityDimension, int... Normals >
constexpr int
GridEntitiesOrientations< GridDimension >::
getOrientationIndex() {
   return NormalsGetter< int, EntityDimension, GridDimension >::template getOrientationIndex< Normals... >();
}

template< int GridDimension >
   template< int... Normals >
constexpr int
GridEntitiesOrientations< GridDimension >::
getTotalOrientationIndex() {
   return NormalsGetter< int, 0, GridDimension >::template getTotalOrientationIndex< Normals... >();
}

template< int GridDimension >
   template< int EntityDimension >
constexpr int
GridEntitiesOrientations< GridDimension >::
getTotalOrientationIndex( int orientation ) {
   return cumulativeCombinationsCount( EntityDimension - 1, GridDimension ) + orientation;
}

template< int GridDimension >
   template< int EntityDimension, int Orientation >
auto
GridEntitiesOrientations< GridDimension >::
getNormals() -> NormalsType {
   return NormalsGetter< int, EntityDimension, GridDimension >::template getNormals< Orientation >();
}

template< int GridDimension >
   template< int TotalOrientation >
auto
GridEntitiesOrientations< GridDimension >::
getNormals() -> NormalsType {
   static_assert( TotalOrientation >= 0 && TotalOrientation < ( 1 << GridDimension ), "Wrong index of total orientation." );
   return NormalsGetter< int, 0, GridDimension >::template getNormalsByTotalOrientation< TotalOrientation >();
};

template< int GridDimension >
   template< int EntityDimension >
auto
GridEntitiesOrientations< GridDimension >::
getNormals( int orientation ) -> NormalsType {
   return normalsTable[ getTotalOrientationIndex< EntityDimension >( orientation ) ];
}

template< int GridDimension >
auto
GridEntitiesOrientations< GridDimension >::
getNormals( int totalOrientation ) -> NormalsType {
   return normalsTable[ totalOrientation ];
}

template< int GridDimension >
   template< int EntityDimension, int Orientation >
void
GridEntitiesOrientations< GridDimension >::
addNormalsToTable( int offset ) {
   normalsTable[ offset ] = getNormals< EntityDimension, Orientation >();
   if constexpr( Orientation < getOrientationsCount< EntityDimension >() - 1 )
      addNormalsToTable< EntityDimension, Orientation + 1 >( offset + 1 );
   else if constexpr( EntityDimension < GridDimension )
      addNormalsToTable< EntityDimension + 1, 0 >( offset + 1 );
}

} //namespace Meshes
} //namespace TNL
