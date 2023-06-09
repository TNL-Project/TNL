// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/AtomicVectorView.h>
#include <TNL/Algorithms/Segments/SegmentsPrinting.h>
#include <TNL/Algorithms/Segments/detail/CheckLambdas.h>


namespace TNL {
namespace Algorithms {
namespace Segments {

template< typename Segments, typename SegmentsView_ = typename Segments::ViewType >
struct GrowingSegmentsView : public SegmentsView_
{
   using SegmentsType = Segments;
   using SegmentsView = typename SegmentsType::ViewType;
   using SegmentsConstView = typename SegmentsType::ConstViewType;
   using IndexType = typename SegmentsType::IndexType;
   using DeviceType = typename SegmentsType::DeviceType;
   using FillingVector = Containers::Vector< IndexType, DeviceType, IndexType >;
   using FillingVectorView = typename FillingVector::ViewType;
   using AtomicFillingVectorView = Containers::AtomicVectorView< IndexType, DeviceType, IndexType >;

   GrowingSegmentsView( SegmentsView&& segmentsView, FillingVectorView&& fillingView )
      : SegmentsView_( segmentsView ), segmentsFilling( fillingView ) {}

   __cuda_callable__
   IndexType newSlot( IndexType segmentIdx )
   {
      IndexType localIdx = segmentsFilling.atomicAdd( segmentIdx, ( IndexType) 1 );
      TNL_ASSERT_LT( localIdx, this->getSegmentSize( segmentIdx ), "" );
      return this->getGlobalIndex( segmentIdx, localIdx );
   }

   __cuda_callable__
   IndexType deleteSlot( IndexType segmentIdx )
   {
      IndexType localIdx = segmentsFilling.atomicAdd( segmentIdx, ( IndexType) -1 );
      return this->getGlobalIndex( segmentIdx, localIdx-1 );
   }

   template< typename Function >
   void forElements( IndexType begin, IndexType end, Function&& f )
   {
      auto main_f = [=,*this] __cuda_callable__ ( IndexType segmentIdx ) mutable {
         for( IndexType localIdx = 0; localIdx < segmentsFilling[ segmentIdx ]; localIdx++ ) {
            f( segmentIdx, localIdx, this->getGlobalIndex( segmentIdx, localIdx ) );
         }
      };
      Algorithms::parallelFor< DeviceType >( begin, end, main_f );
   }

   template< typename Function >
   void forAllElements( Function&& f )
   {
      forElements( 0, this->getSegmentsCount(), f );
   }

   /*template< typename Function >
   void
   sequentialForSegments( IndexType begin, IndexType end, Function&& function ) const
   {
      Segments::sequentialForSegments( begin, end, function ); // TODO: we need special SegmentView for growing segemnts.
   }

   template< typename Function >
   void
   sequentialForAllSegments( Function&& f ) const
   {
      Segemnts::sequentialForAllSegments( 0, this->sequentialForSegments(), f );
   }*/

   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
   void reduceSegments( IndexType begin,
                        IndexType end,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        ResultKeeper&& keeper,
                        const Value& identity ) const
   {
      // NVCC does not allow if constexpr inside lambda
      if constexpr( detail::CheckFetchLambda< IndexType, Fetch >::hasAllParameters() ) {
         auto main_fetch_with_all_params = [=,*this] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx, bool compute ) mutable {
            IndexType end = this->segmentsFilling[ segmentIdx ];
            if( localIdx < end  ) {
               if( localIdx == end -1 )
                  compute = false;
               return fetch( segmentIdx, localIdx, globalIdx, compute );
            }
            else return identity;
         };
         SegmentsView_::reduceSegments( begin, end, main_fetch_with_all_params, reduction, keeper, identity );
      }
      else {
         auto main_fetch = [=,*this] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx, bool compute ) mutable {
            IndexType end = this->segmentsFilling[ segmentIdx ];
            if( localIdx < end  ) {
               if( localIdx == end -1 )
                  compute = false;
               return fetch( globalIdx, compute );
            }
            else return identity;
         };
         SegmentsView_::reduceSegments( begin, end, main_fetch, reduction, keeper, identity );
      }
   }

   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
   void reduceAllSegments( Fetch& fetch,
                           const Reduction& reduction,
                           ResultKeeper& keeper,
                           const Value& identity ) const
   {
      this->reduceSegments( 0, this->segments.getSegmentsCount(), fetch, reduction, keeper, identity );
   }

   void clear()
   {
      this->segmentsFilling = 0;
   }

   const FillingVector& getFilling() const
   {
      return this->segmentsFilling;
   }

   template< typename Fetch >
   auto
   print( Fetch&& fetch ) const -> SegmentsPrinter< SegmentsConstView, Fetch >
   {
      return SegmentsPrinter< SegmentsConstView, Fetch >( *this, fetch );
   }

private:
   AtomicFillingVectorView segmentsFilling;
};


} // namespace Segments
} // namespace Algorithms
} // namespace TNL
