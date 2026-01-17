// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/print.h>
#include <TNL/Algorithms/Segments/GrowingSegmentsView.h>

namespace TNL::Algorithms::Segments {

template< typename Segments >
struct GrowingSegments : public Segments
{
   using SegmentsType = Segments;
   using SegmentsConstView = typename SegmentsType::ConstViewType;
   using IndexType = typename SegmentsType::IndexType;
   using DeviceType = typename SegmentsType::DeviceType;
   using FillingVector = Containers::Vector< IndexType, DeviceType, IndexType >;
   using FillingVectorView = typename FillingVector::ViewType;
   using GrowingSegmentsViewType = GrowingSegmentsView< SegmentsType >;
   using SegmentViewType = typename Segments::SegmentViewType;

   template< typename SizesContainer >
   GrowingSegments( const SizesContainer& segmentsSizes )
   : SegmentsType( segmentsSizes ),
     segmentsFilling( segmentsSizes.getSize(), 0 ),
     view( Segments::getView(), segmentsFilling.getView() )
   {}

   template< typename ListIndex >
   GrowingSegments( const std::initializer_list< ListIndex >& segmentsSizes )
   : SegmentsType( segmentsSizes ),
     segmentsFilling( segmentsSizes.size(), 0 ),
     view( SegmentsType::getView(), segmentsFilling.getView() )
   {}

   GrowingSegmentsViewType
   getView()
   {
      return GrowingSegmentsViewType( SegmentsType::getView(), segmentsFilling.getView() );
   }

   //const GrowingSegmentsViewType getView() const
   //{
   //   return GrowingSegmentsViewType( SegmentsType::getConstView(), segmentsFilling.getConstView() );
   //}

   __cuda_callable__
   IndexType
   newSlot( IndexType segmentIdx )
   {
      return this->view.newSlot( segmentIdx );
   }

   __cuda_callable__
   IndexType
   deleteSlot( IndexType segmentIdx )
   {
      return this->view.deleteSlot( segmentIdx );
   }

   template< typename Function >
   void
   forElements( IndexType begin, IndexType end, Function&& f )
   {
      this->view.forElements( begin, end, f );
   }

   template< typename Function >
   void
   forAllElements( Function&& f )
   {
      forElements( 0, this->getSegmentCount(), f );
   }

   /*template< typename Function >
   void
   sequentialForSegments( IndexType begin, IndexType end, Function&& function ) const
   {
      this->view.sequentialForSegments( begin, end, function );
   }

   template< typename Function >
   void
   sequentialForAllSegments( Function&& f ) const
   {
      this->view.sequentialForAllSegments( f );
   }*/

   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
   void
   reduceSegments( IndexType begin,
                   IndexType end,
                   Fetch& fetch,
                   const Reduction& reduction,
                   ResultKeeper& keeper,
                   const Value& identity ) const
   {
      this->view.reduceSegments( begin, end, fetch, reduction, keeper, identity );
   }

   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
   void
   reduceAllSegments( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Value& identity ) const
   {
      this->reduceSegments( (IndexType) 0, this->getSegmentCount(), fetch, reduction, keeper, identity );
   }

   void
   clear()
   {
      this->segmentsFilling = 0;
   }

   [[nodiscard]] const FillingVector&
   getFilling() const
   {
      return this->segmentsFilling;
   }

   /*template< typename Fetch >
   auto
   print( Fetch&& fetch ) const -> SegmentsPrinter< GrowingSegments, Fetch >
   {
      return SegmentsPrinter< GrowingSegments, Fetch >( *this, std::move( fetch ) );
   }*/

private:
   FillingVector segmentsFilling;
   GrowingSegmentsViewType view;
};

}  // namespace TNL::Algorithms::Segments
