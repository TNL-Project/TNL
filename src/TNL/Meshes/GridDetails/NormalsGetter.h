// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>
#include <TNL/Meshes/GridDetails/Templates/Permutations.h>
#include <TNL/Meshes/GridDetails/Templates/Functions.h>

namespace TNL::Meshes {

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
   using Permutations =
      Templates::make_int_permutations< GridDimension,
                                        Templates::build_ones_pack< GridDimension - EntityDimension, GridDimension > >;

   template< int TotalOrientation >
   [[nodiscard]] constexpr static int
   getEntityDimension()
   {
      static_assert( TotalOrientation < getOrientationsCount() || EntityDimension < GridDimension,
                     "Wrong number of total orientation." );
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
      using Value = Templates::get< Orientation, Permutations >;

      return BuildNormals< Value >::build();
   }

   template< int StaticOrientation = 0 >
   NormalsType
   getNormals( int orientation )
   {
      TNL_ASSERT_LT( orientation, getOrientationsCount(), "Wrong index of orientation." );
      static_assert( StaticOrientation >= 0 && StaticOrientation < getOrientationsCount(), "Wrong orientation index." );
      using Value = Templates::get< StaticOrientation, Permutations >;
      if( orientation == StaticOrientation )
         return BuildNormals< Value >::build();
      if constexpr( StaticOrientation < getOrientationsCount() - 1 )
         return getNormals< StaticOrientation + 1 >( orientation );
      return BuildNormals< Value >::build();  // Just to avoid warning, this should never happen.
   }

   template< int... Normals >  //, std::enable_if_t< sizeof...( Normals ) == GridDimension, bool > = true >
   constexpr static int
   getOrientationIndex()
   {
      static_assert( sizeof...( Normals ) == GridDimension, "The size of normals must be equal to grid dimension." );
      return IndexOfNormalsGetter< getOrientationsCount() - 1, Normals... >::getIndex();
   }

   template< int... Normals >  //, std::enable_if_t< sizeof...( Normals ) == GridDimension, bool > = true >
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

private:
   template< class >
   struct BuildNormals;

   template< int... Values >
   struct BuildNormals< TNL::Meshes::Templates::int_pack< Values... > >
   {
   public:
      using normals_integer_sequence = std::integer_sequence< int, Values... >;
      [[nodiscard]] constexpr static NormalsType
      build()
      {
         return NormalsType( Values... );
      }
   };

   template< int OrientationIdx, int... Normals >
   struct IndexOfNormalsGetter
   {
      constexpr static int
      getIndex()
      {
         static_assert( OrientationIdx >= 0, "Index of entity orientation cannot be negative." );
         static_assert( OrientationIdx < getOrientationsCount(),
                        "Index of entity orientation must be smaller than number of all orientations." );
         using sequence1 = std::integer_sequence< int, Normals... >;
         using sequence2 = typename BuildNormals< Templates::get< OrientationIdx, Permutations > >::normals_integer_sequence;
         if constexpr( std::is_same< sequence1, sequence2 >::value )
            return OrientationIdx;
         else if constexpr( OrientationIdx > 0 )
            return IndexOfNormalsGetter< OrientationIdx - 1, Normals... >::getIndex();
         else
            return -1;
      }
   };
};

}  // namespace TNL::Meshes
