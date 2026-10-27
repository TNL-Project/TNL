// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Devices/GPU.h>

#include "AdaptiveCSR.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, typename IndexAllocator >
AdaptiveCSR< Device, Index, IndexAllocator >::AdaptiveCSR()
{
   this->view.bind( Base::getOffsets().getView(), this->blocksArray );
}

template< typename Device, typename Index, typename IndexAllocator >
AdaptiveCSR< Device, Index, IndexAllocator >::AdaptiveCSR( const AdaptiveCSR& segments )
: Base( segments )
{
   for( int i = 0; i < MaxValueSizeLog(); i++ )
      this->blocksArray[ i ] = segments.blocksArray[ i ];
   this->view.bind( Base::getOffsets().getView(), this->blocksArray );
}

template< typename Device, typename Index, typename IndexAllocator >
template< typename SizesContainer, std::enable_if_t< IsArrayType< SizesContainer >::value, bool > >
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
   blocksArray = std::move( segments.blocksArray );
   this->view.bind( this->offsets, this->blocksArray );
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator >
template< typename Device_, typename Index_, typename IndexAllocator_ >
AdaptiveCSR< Device, Index, IndexAllocator >&
AdaptiveCSR< Device, Index, IndexAllocator >::operator=( const AdaptiveCSR< Device_, Index_, IndexAllocator_ >& segments )
{
   Base::operator=( segments );
   // The blocks exist only on GPUs, so they are copied only between GPUs
   if constexpr( std::is_same_v< Device, Devices::GPU > && std::is_same_v< Device_, Devices::GPU > ) {
      for( int i = 0; i < MaxValueSizeLog(); i++ )
         this->blocksArray[ i ] = segments.getBlocks()[ i ];
      this->view.bind( Base::getOffsets(), this->blocksArray );
   }
   else
      this->initBlocks();
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator >
std::string
AdaptiveCSR< Device, Index, IndexAllocator >::getSerializationType()
{
   return "AdaptiveCSR< " + TNL::getSerializationType< Index >() + " >";
}

template< typename Device, typename Index, typename IndexAllocator >
std::string
AdaptiveCSR< Device, Index, IndexAllocator >::getSegmentsType()
{
   return "Adaptive CSR";
}

template< typename Device, typename Index, typename IndexAllocator >
auto
AdaptiveCSR< Device, Index, IndexAllocator >::getView() -> ViewType
{
   return this->view;
}

template< typename Device, typename Index, typename IndexAllocator >
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
   this->initBlocks();
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
AdaptiveCSR< Device, Index, IndexAllocator >::getBlocks() const -> const BlocksViewArray&
{
   return this->view.getBlocks();
}

template< typename Device, typename Index, typename IndexAllocator >
void
AdaptiveCSR< Device, Index, IndexAllocator >::save( File& file ) const
{
   Base::save( file );
   for( int i = 0; i < MaxValueSizeLog(); i++ )
      file << this->blocksArray[ i ];
}

template< typename Device, typename Index, typename IndexAllocator >
void
AdaptiveCSR< Device, Index, IndexAllocator >::load( File& file )
{
   Base::load( file );
   for( int i = 0; i < MaxValueSizeLog(); i++ )
      file >> this->blocksArray[ i ];
   // The file may come from segments on another device, which have no blocks
   this->initBlocks();
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
         const Index warpsPerBlock =
            detail::CSRAdaptiveKernelParameters< SizeOfValue >::BlockSize() / Backend::getWarpSize( Backend::getDevice() );
         Index warpsLeft = roundUpDivision( blocksCount, warpsPerBlock ) * warpsPerBlock - blocksCount;
         if( warpsLeft == 0 )
            warpsLeft = warpsPerBlock;
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

template< typename Device, typename Index, typename IndexAllocator >
void
AdaptiveCSR< Device, Index, IndexAllocator >::initBlocks()
{
   // The blocks are used only by the reduction on GPUs. On the host and the
   // sequential device, the reduction falls back to the CSR kernels, so the
   // sequential construction of the blocks is skipped.
   bool hasBlocks = false;
   if constexpr( std::is_same_v< Device, Devices::GPU > ) {
      const auto& offsets = this->getOffsets();
      hasBlocks = max( offsets ) != 0;
      if( hasBlocks ) {
         this->template initValueSize< 1 >( offsets );
         this->template initValueSize< 2 >( offsets );
         this->template initValueSize< 4 >( offsets );
         this->template initValueSize< 8 >( offsets );
         this->template initValueSize< 16 >( offsets );
         this->template initValueSize< 32 >( offsets );
      }
   }
   if( ! hasBlocks )
      for( int i = 0; i < MaxValueSizeLog(); i++ )
         this->blocksArray[ i ].reset();
   this->view.bind( Base::getOffsets(), this->blocksArray );
}

}  // namespace TNL::Algorithms::Segments
