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
