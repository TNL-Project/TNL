// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Sorting/DefaultSorter.h>

#include "CSR.h"

namespace TNL::Algorithms::Segments {

template< typename EmbeddedSegments, typename IndexAllocator >
SortedSegments< EmbeddedSegments, IndexAllocator >::SortedSegments( const SortedSegments& segments_ )
: embeddedSegments( segments_.embeddedSegments ),
  segmentsPermutation( segments_.segmentsPermutation ),
  inverseSegmentsPermutation( segments_.inverseSegmentsPermutation )

{
   // update the base
   Base::bind(
      this->embeddedSegments.getView(), this->segmentsPermutation.getView(), this->inverseSegmentsPermutation.getView() );
}

template< typename EmbeddedSegments, typename IndexAllocator >
template< typename SizesContainer, typename T >
SortedSegments< EmbeddedSegments, IndexAllocator >::SortedSegments( const SizesContainer& segmentsSizes )
{
   this->setSegmentsSizes( segmentsSizes );
}

template< typename EmbeddedSegments, typename IndexAllocator >
template< typename ListIndex >
SortedSegments< EmbeddedSegments, IndexAllocator >::SortedSegments( const std::initializer_list< ListIndex >& segmentsSizes )
{
   this->setSegmentsSizes( OffsetsContainer( segmentsSizes ) );
}

template< typename EmbeddedSegments, typename IndexAllocator >
SortedSegments< EmbeddedSegments, IndexAllocator >&
SortedSegments< EmbeddedSegments, IndexAllocator >::operator=( const SortedSegments& segments_ )
{
   this->embeddedSegments = segments_.embeddedSegments;
   this->segmentsPermutation = segments_.segmentsPermutation;
   this->inverseSegmentsPermutation = segments_.inverseSegmentsPermutation;

   // update the base
   Base::bind(
      this->embeddedSegments.getView(), this->segmentsPermutation.getView(), this->inverseSegmentsPermutation.getView() );
   return *this;
}

template< typename EmbeddedSegments, typename IndexAllocator >
SortedSegments< EmbeddedSegments, IndexAllocator >&
SortedSegments< EmbeddedSegments, IndexAllocator >::operator=( SortedSegments&& segments_ ) noexcept( false )
{
   this->embeddedSegments = std::move( segments_.embeddedSegments );
   this->segmentsPermutation = std::move( segments_.segmentsPermutation );
   this->inverseSegmentsPermutation = std::move( segments_.inverseSegmentsPermutation );

   // update the base
   Base::bind(
      this->embeddedSegments.getView(), this->segmentsPermutation.getView(), this->inverseSegmentsPermutation.getView() );
   return *this;
}

template< typename EmbeddedSegments, typename IndexAllocator >
template< typename EmbeddedSegments_, typename IndexAllocator_ >
SortedSegments< EmbeddedSegments, IndexAllocator >&
SortedSegments< EmbeddedSegments, IndexAllocator >::operator=(
   const SortedSegments< EmbeddedSegments_, IndexAllocator_ >& segments_ ) noexcept( false )
{
   this->embeddedSegments = segments_.getEmbeddedSegments();
   this->segmentsPermutation = segments_.getSegmentsPermutation();
   this->inverseSegmentsPermutation = segments_.getInverseSegmentsPermutation();

   // update the base
   Base::bind(
      this->embeddedSegments.getView(), this->segmentsPermutation.getView(), this->inverseSegmentsPermutation.getView() );
   return *this;
}

template< typename EmbeddedSegments, typename IndexAllocator >
template< typename EmbeddedSegments_, typename IndexAllocator_ >
SortedSegments< EmbeddedSegments, IndexAllocator >&
SortedSegments< EmbeddedSegments, IndexAllocator >::operator=(
   SortedSegments< EmbeddedSegments_, IndexAllocator_ >&& segments_ ) noexcept( false )
{
   this->embeddedSegments = std::move( segments_.getEmbeddedSegments() );
   this->segmentsPermutation = std::move( segments_.getSegmentsPermutation() );
   this->inverseSegmentsPermutation = std::move( segments_.getInverseSegmentsPermutation() );

   // update the base
   Base::bind(
      this->embeddedSegments.getView(), this->segmentsPermutation.getView(), this->inverseSegmentsPermutation.getView() );
   return *this;
}

template< typename EmbeddedSegments, typename IndexAllocator >
auto
SortedSegments< EmbeddedSegments, IndexAllocator >::getView() -> ViewType
{
   return { this->embeddedSegmentsView.getView(),
            this->segmentsPermutationView.getView(),
            this->inverseSegmentsPermutationView.getView() };
}

template< typename EmbeddedSegments, typename IndexAllocator >
auto
SortedSegments< EmbeddedSegments, IndexAllocator >::getConstView() const -> ConstViewType
{
   EmbeddedSegmentsConstView embeddedSegmentsView;
   embeddedSegmentsView.bind( this->getEmbeddedSegmentsView() );
   ConstPermutationView segmentsPermutation =
      const_cast< std::remove_const_t< decltype( this ) > >( this )->getSegmentsPermutationView();
   ConstPermutationView inverseSegmentsPermutation =
      const_cast< std::remove_const_t< decltype( this ) > >( this )->getInverseSegmentsPermutationView();
   ViewType view;
   view.bind( embeddedSegmentsView, segmentsPermutation, inverseSegmentsPermutation );
   return *(ConstViewType*) &view;
}

template< typename EmbeddedSegments, typename IndexAllocator >
template< typename SizesHolder >
void
SortedSegments< EmbeddedSegments, IndexAllocator >::setSegmentsSizes( const SizesHolder& sizes )
{
   // TODO: Reimplement the following when TNL sorters can create the permutation vector itself
   using Tuple = Containers::StaticArray< 2, IndexType >;
   if( sizes.getSize() == 0 ) {
      this->reset();
      return;
   }

   // Sort the segments sizes in descending order
   Containers::Array< Tuple, DeviceType, IndexType > aux( sizes.getSize() );
   auto sizesView = sizes.getConstView();
   aux.forAllElements(
      [ = ] __cuda_callable__( IndexType i, Tuple & tuple )
      {
         tuple[ 0 ] = sizesView[ i ];
         tuple[ 1 ] = i;
      } );

   //std::cout << "aux before sorting: " << aux << std::endl;
   auto aux_view = aux.getView();
   typename Algorithms::Sorting::DefaultSorter< DeviceType >::SorterType sorter;
   sorter.sort( aux_view,
                [] __cuda_callable__( const Tuple& a, const Tuple& b )
                {
                   return a[ 0 ] > b[ 0 ];  // sort in descending order
                } );

   // Initialize the embedded segments with the sorted sizes
   auto auxView = aux.getConstView();
   SizesHolder sortedSizes( sizes.getSize() );
   sortedSizes.forAllElements(
      [ = ] __cuda_callable__( IndexType i, IndexType & value )
      {
         value = auxView[ i ][ 0 ];
      } );
   //std::cout << "sorted sizes: " << sortedSizes << std::endl;
   this->embeddedSegments.setSegmentsSizes( sortedSizes );

   // Create the inverse segments permutation and the segments permutation
   this->inverseSegmentsPermutation.setSize( sizes.getSize() );
   this->inverseSegmentsPermutation.forAllElements(
      [ = ] __cuda_callable__( IndexType i, IndexType & value )
      {
         value = auxView[ i ][ 1 ];
      } );
   //std::cout << "inverse segments permutation: " << this->inverseSegmentsPermutation << std::endl;

   this->segmentsPermutation.setSize( sizes.getSize() );
   auto inverseSegmentsPermutationView = this->inverseSegmentsPermutation.getView();
   auto segmentsPermutationView = this->segmentsPermutation.getView();
   Algorithms::parallelFor< DeviceType >( 0,
                                          this->segmentsPermutation.getSize(),
                                          [ = ] __cuda_callable__( IndexType i ) mutable
                                          {
                                             TNL_ASSERT_LT( i, inverseSegmentsPermutationView.getSize(), "" );
                                             TNL_ASSERT_LT(
                                                inverseSegmentsPermutationView[ i ], segmentsPermutationView.getSize(), "" );
                                             segmentsPermutationView[ inverseSegmentsPermutationView[ i ] ] = i;
                                          } );

   //std::cout << "segments permutation: " << this->segmentsPermutation << std::endl;
   // update the base
   Base::bind(
      this->embeddedSegments.getView(), this->segmentsPermutation.getView(), this->inverseSegmentsPermutation.getView() );
}

template< typename EmbeddedSegments, typename IndexAllocator >
void
SortedSegments< EmbeddedSegments, IndexAllocator >::reset()
{
   this->embeddedSegments.reset();
   this->segmentsPermutation.setSize( 0 );
   this->inverseSegmentsPermutation.setSize( 0 );

   // update the base
   Base::bind(
      this->embeddedSegments.getView(), this->segmentsPermutation.getView(), this->inverseSegmentsPermutation.getView() );
}

template< typename EmbeddedSegments, typename IndexAllocator >
auto
SortedSegments< EmbeddedSegments, IndexAllocator >::getEmbeddedSegments() const -> const EmbeddedSegments&
{
   return this->embeddedSegments;
}

template< typename EmbeddedSegments, typename IndexAllocator >
auto
SortedSegments< EmbeddedSegments, IndexAllocator >::getSegmentsPermutation() const -> const PermutationContainer&
{
   return this->segmentsPermutation;
}

template< typename EmbeddedSegments, typename IndexAllocator >
auto
SortedSegments< EmbeddedSegments, IndexAllocator >::getInverseSegmentsPermutation() const -> const PermutationContainer&
{
   return this->inverseSegmentsPermutation;
}

template< typename EmbeddedSegments, typename IndexAllocator >
void
SortedSegments< EmbeddedSegments, IndexAllocator >::save( File& file ) const
{
   this->embeddedSegments.save( file );
   file << this->segmentsPermutation << this->inverseSegmentsPermutation;
}

template< typename EmbeddedSegments, typename IndexAllocator >
void
SortedSegments< EmbeddedSegments, IndexAllocator >::load( File& file )
{
   this->embeddedSegments.load( file );
   file >> this->segmentsPermutation >> this->inverseSegmentsPermutation;

   // update the base
   Base::bind(
      this->embeddedSegments.getView(), this->segmentsPermutation.getView(), this->inverseSegmentsPermutation.getView() );
}

}  // namespace TNL::Algorithms::Segments
