// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Algorithms::Segments::detail {

template< typename Segments >
struct TraversingOperationsBaseline
{
   using ViewType = typename Segments::ViewType;
   using ConstViewType = typename ViewType::ConstViewType;
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIfSparse( const ConstViewType& segments,
                        IndexBegin begin,
                        IndexEnd end,
                        Condition condition,
                        Function function,
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
      forElements( segments, indexes, function, launchConfig );
   }
};

}  //namespace TNL::Algorithms::Segments::detail
