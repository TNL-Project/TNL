// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "AdaptiveCSR.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, typename IndexAllocator >
AdaptiveCSR< Device, Index, IndexAllocator >::AdaptiveCSR( const AdaptiveCSR& segments ) : Base( segments )
{
   for( int i = 0; i < MaxValueSizeLog(); i++ )
      this->blocksArray[ i ] = segments.blocksArray[ i ];
   this->view.bind( Base::getOffsets().getView(), this->blocksArray );
}

template< typename Device, typename Index, typename IndexAllocator >
template< typename SizesContainer >
AdaptiveCSR< Device, Index, IndexAllocator >::AdaptiveCSR( const SizesContainer& segmentsSizes )
{
   this->setSegmentsSizes( segmentsSizes );
}

template< typename Device, typename Index, typename IndexAllocator >
template< typename ListIndex >
AdaptiveCSR< Device, Index, IndexAllocator >::AdaptiveCSR( const std::initializer_list< ListIndex >& segmentsSizes )
: Base( segmentsSizes )
{
   this->setSegmentsSizes( OffsetsContainer( segmentsSizes ) );
}

template< typename Device, typename Index, typename IndexAllocator >
AdaptiveCSR< Device, Index, IndexAllocator >&
AdaptiveCSR< Device, Index, IndexAllocator >::operator=( const AdaptiveCSR& segments )
{
   Base::operator=( segments );
   for( int i = 0; i < MaxValueSizeLog(); i++ )
      this->blocksArray[ i ] = segments.blocksArray[ i ];
   this->view.bind( Base::getOffsets(), this->blocksArray );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator >
AdaptiveCSR< Device, Index, IndexAllocator >&
AdaptiveCSR< Device, Index, IndexAllocator >::operator=( AdaptiveCSR&& segments ) noexcept( false )
{
   this->offsets = std::move( segments.offsets );
   for( int i = 0; i < MaxValueSizeLog(); i++ )
      this->blocksArray[ i ] = std::move( segments.blocksArray[ i ] );
   this->view.bind( this->offsets, this->blocksArray );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator >
template< typename Device_, typename Index_, typename IndexAllocator_ >
AdaptiveCSR< Device, Index, IndexAllocator >&
AdaptiveCSR< Device, Index, IndexAllocator >::operator=( const AdaptiveCSR< Device_, Index_, IndexAllocator_ >& segments )
{
   Base::operator=( segments );
   for( int i = 0; i < MaxValueSizeLog(); i++ )
      this->blocksArray[ i ] = segments.blocksArray[ i ];
   this->view.bind( Base::getOffsets(), this->blocksArray );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator >
__cuda_callable__
auto
AdaptiveCSR< Device, Index, IndexAllocator >::getView() -> ViewType
{
   return this->view;
}

template< typename Device, typename Index, typename IndexAllocator >
__cuda_callable__
auto
AdaptiveCSR< Device, Index, IndexAllocator >::getConstView() const -> ConstViewType
{
   return this->view.getConstView();
}

template< typename Device, typename Index, typename IndexAllocator >
template< typename SizesContainer >
void
AdaptiveCSR< Device, Index, IndexAllocator >::setSegmentsSizes( const SizesContainer& segmentsSizes )
{
   Base::setSegmentsSizes( segmentsSizes );
   const auto& offsets = this->getOffsets();

   if( max( offsets ) == 0 ) {
      for( int i = 0; i < MaxValueSizeLog(); i++ ) {
         this->blocksArray[ i ].reset();
         this->view.setBlocks( this->blocksArray[ i ], i );
      }
   }
   else {
      this->template initValueSize< 1 >( offsets );
      this->template initValueSize< 2 >( offsets );
      this->template initValueSize< 4 >( offsets );
      this->template initValueSize< 8 >( offsets );
      this->template initValueSize< 16 >( offsets );
      this->template initValueSize< 32 >( offsets );
      for( int i = 0; i < MaxValueSizeLog(); i++ )
         this->view.setBlocks( this->blocksArray[ i ], i );
   }
   this->view.bind( Base::getOffsets(), this->blocksArray );
}

template< typename Device, typename Index, typename IndexAllocator >
void
AdaptiveCSR< Device, Index, IndexAllocator >::reset()
{
   Base::reset();
   for( int i = 0; i < MaxValueSizeLog(); i++ ) {
      this->blocksArray[ i ].reset();
      this->view.setBlocks( this->blocksArray[ i ], i );
   }
}

template< typename Device, typename Index, typename IndexAllocator >
auto
AdaptiveCSR< Device, Index, IndexAllocator >::getBlocks() const -> const BlocksView*
{
   return this->view.getBlocks();
}

template< typename Device, typename Index, typename IndexAllocator >
void
AdaptiveCSR< Device, Index, IndexAllocator >::save( File& file ) const
{
   Base::save( file );
   file << this->blocksArray;
}

template< typename Device, typename Index, typename IndexAllocator >
void
AdaptiveCSR< Device, Index, IndexAllocator >::load( File& file )
{
   Base::load( file );
   file >> this->blocksArray;
   this->view.bind( Base::getOffsets(), this->blocksArray );
}

template< typename Device, typename Index, typename IndexAllocator >
template< int SizeOfValue, typename Offsets >
Index
AdaptiveCSR< Device, Index, IndexAllocator >::findLimit( Index start, const Offsets& offsets, Index size, detail::Type& type )
{
   std::size_t sum = 0;
   for( Index current = start; current < size - 1; current++ ) {
      Index elements = offsets[ current + 1 ] - offsets[ current ];
      sum += elements;
      if( sum > detail::CSRAdaptiveKernelParameters< SizeOfValue >::StreamedSharedElementsPerWarp() ) {
         if( current - start > 0 ) {
            // extra row
            type = detail::Type::STREAM;
            return current;
         }
         else {
            // one long row
            if( sum <= 2 * detail::CSRAdaptiveKernelParameters< SizeOfValue >::MaxAdaptiveElementsPerWarp() )
               type = detail::Type::VECTOR;
            else
               type = detail::Type::LONG;
            return current + 1;
         }
      }
   }
   type = detail::Type::STREAM;
   return size - 1;  // return last row pointer
}

template< typename Device, typename Index, typename IndexAllocator >
template< int SizeOfValue, typename Offsets >
void
AdaptiveCSR< Device, Index, IndexAllocator >::initValueSize( const Offsets& offsets )
{
   using HostOffsetsType = Containers::Vector< typename Offsets::IndexType, Devices::Host, typename Offsets::IndexType >;
   HostOffsetsType hostOffsets;
   hostOffsets = offsets;
   const Index rows = offsets.getSize();
   Index start = 0;
   Index nextStart = 0;

   // Fill blocks
   std::vector< detail::CSRAdaptiveKernelBlockDescriptor< Index > > inBlocks;
   inBlocks.reserve( rows );

   while( nextStart != rows - 1 ) {
      detail::Type type;
      nextStart = findLimit< SizeOfValue >( start, hostOffsets, rows, type );
      if( type == detail::Type::LONG ) {
         const Index blocksCount = inBlocks.size();
         const Index warpsPerCudaBlock =
            detail::CSRAdaptiveKernelParameters< SizeOfValue >::CudaBlockSize() / Backend::getWarpSize();
         Index warpsLeft = roundUpDivision( blocksCount, warpsPerCudaBlock ) * warpsPerCudaBlock - blocksCount;
         if( warpsLeft == 0 )
            warpsLeft = warpsPerCudaBlock;
         for( Index index = 0; index < warpsLeft; index++ )
            inBlocks.emplace_back( start, detail::Type::LONG, index, warpsLeft );
      }
      else {
         inBlocks.emplace_back( start, type, nextStart, offsets.getElement( nextStart ), offsets.getElement( start ) );
      }
      start = nextStart;
   }
   inBlocks.emplace_back( nextStart );
   this->blocksArray[ getSizeValueLog( SizeOfValue ) ] = inBlocks;
}

}  // namespace TNL::Algorithms::Segments