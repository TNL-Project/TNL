// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Algorithms/compress.h>

#pragma once

namespace TNL::Algorithms::Segments::detail {

template< typename Segments >
struct ReducingOperationsBase
{
   using ViewType = typename Segments::ViewType;
   using ConstViewType = typename ViewType::ConstViewType;
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   template< typename IndexBegin,
             typename IndexEnd,
             typename Condition,
             typename Fetch,
             typename Reduction,
             typename ResultStorer,
             typename Value >
   static IndexType
   reduceSegmentsIf( const Segments& segments,
                     IndexBegin begin,
                     IndexEnd end,
                     Condition&& condition,
                     Fetch&& fetch,
                     Reduction&& reduction,
                     ResultStorer&& storer,
                     const Value& identity,
                     LaunchConfiguration launchConfig )
   {
      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;

      VectorType conditions( end - begin );
      conditions.forAllElements(
         [ = ] __cuda_callable__( IndexType i, IndexType & value )
         {
            value = condition( i + begin );
         } );

      auto indexes = compressFast< VectorType >( conditions );
      indexes += begin;
      reduceSegments( segments,
                      indexes,
                      std::forward< Fetch >( fetch ),
                      std::forward< Reduction >( reduction ),
                      std::forward< ResultStorer >( storer ),
                      identity,
                      launchConfig );
      return indexes.getSize();
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Condition,
             typename Fetch,
             typename Reduction,
             typename ResultStorer,
             typename Value >
   static IndexType
   reduceSegmentsWithArgumentIf( const Segments& segments,
                                 IndexBegin begin,
                                 IndexEnd end,
                                 Condition&& condition,
                                 Fetch&& fetch,
                                 Reduction&& reduction,
                                 ResultStorer&& storer,
                                 const Value& identity,
                                 LaunchConfiguration launchConfig )
   {
      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;

      VectorType conditions( end - begin );
      conditions.forAllElements(
         [ = ] __cuda_callable__( IndexType i, IndexType & value )
         {
            value = condition( i + begin );
         } );

      auto indexes = compressFast< VectorType >( conditions );
      indexes += begin;
      reduceSegmentsWithArgument( segments,
                                  indexes,
                                  std::forward< Fetch >( fetch ),
                                  std::forward< Reduction >( reduction ),
                                  std::forward< ResultStorer >( storer ),
                                  identity,
                                  launchConfig );
      return indexes.getSize();
   }
};

}  //namespace TNL::Algorithms::Segments::detail
