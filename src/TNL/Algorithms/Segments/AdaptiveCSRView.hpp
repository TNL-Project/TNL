// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "AdaptiveCSRView.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index >
__cuda_callable__
AdaptiveCSRView< Device, Index >::AdaptiveCSRView( const AdaptiveCSRView& view )
{
   auto* ptr = const_cast< AdaptiveCSRView* >( &view );
   bind( ptr->getOffsets(), ptr->blocksArray );
}

template< typename Device, typename Index >
__cuda_callable__
AdaptiveCSRView< Device, Index >::AdaptiveCSRView( AdaptiveCSRView&& view ) noexcept
{
   Base::bind( view.getOffsets() );
   for( int i = 0; i < MaxValueSizeLog(); i++ )
      this->blocksArray[ i ].bind( std::move( view.blocksArray[ i ] ) );
}

template< typename Device, typename Index >
__cuda_callable__
AdaptiveCSRView< Device, Index >::AdaptiveCSRView( const CSRView< Device, Index >& csrView, const BlocksViewArray& blocks )
{
   Base::bind( csrView.getOffsets() );
   for( int i = 0; i < MaxValueSizeLog(); i++ )
      this->blocksArray[ i ].bind( blocks[ i ] );
}

template< typename Device, typename Index >
__cuda_callable__
AdaptiveCSRView< Device, Index >::AdaptiveCSRView( const CSRView< Device, Index >& csrView, BlocksArray& blocks )
{
   Base::bind( csrView.getOffsets() );
   for( int i = 0; i < MaxValueSizeLog(); i++ )
      this->blocksArray[ i ].bind( blocks[ i ].getView() );
}

template< typename Device, typename Index >
__cuda_callable__
void
AdaptiveCSRView< Device, Index >::bind( AdaptiveCSRView view )
{
   Base::bind( view.getOffsets() );
   for( int i = 0; i < MaxValueSizeLog(); i++ )
      this->blocksArray[ i ].bind( std::move( view.blocksArray[ i ] ) );
}

template< typename Device, typename Index >
__cuda_callable__
void
AdaptiveCSRView< Device, Index >::bind( OffsetsView offsets, const BlocksViewArray& blocks )
{
   Base::bind( std::move( offsets ) );
   for( int i = 0; i < MaxValueSizeLog(); i++ )
      this->blocksArray[ i ].bind( blocks[ i ] );
}

template< typename Device, typename Index >
__cuda_callable__
void
AdaptiveCSRView< Device, Index >::bind( OffsetsView offsets, BlocksArray& blocks )
{
   Base::bind( std::move( offsets ) );
   for( int i = 0; i < MaxValueSizeLog(); i++ )
      this->blocksArray[ i ].bind( blocks[ i ].getView() );
}

template< typename Index, typename Device >
void
AdaptiveCSRView< Index, Device >::setBlocks( BlocksType& blocks, const int idx )
{
   this->blocksArray[ idx ].bind( blocks );
}

template< typename Device, typename Index >
std::string
AdaptiveCSRView< Device, Index >::getSerializationType()
{
   return "AdaptiveCSR< " + TNL::getSerializationType< Index >() + " >";
}

template< typename Device, typename Index >
std::string
AdaptiveCSRView< Device, Index >::getSegmentsType()
{
   return "Adaptive CSR";
}

template< typename Device, typename Index >
[[nodiscard]] __cuda_callable__
auto
AdaptiveCSRView< Device, Index >::getView() -> ViewType
{
   return *this;
}

template< typename Device, typename Index >
[[nodiscard]] __cuda_callable__
auto
AdaptiveCSRView< Device, Index >::getConstView() const -> ConstViewType
{
   using BaseConstViewType = typename Base::ConstViewType;
   return ConstViewType( BaseConstViewType( this->getOffsets().getConstView() ), this->blocksArray );
}

template< typename Device, typename Index >
auto __cuda_callable__
AdaptiveCSRView< Device, Index >::getBlocks() const->const BlocksViewArray&
{
   return this->blocksArray;
}

template< typename Device, typename Index >
void
AdaptiveCSRView< Device, Index >::save( File& file ) const
{
   Base::save( file );
   for( int i = 0; i < MaxValueSizeLog(); i++ )
      file << this->blocksArray[ i ];
}

template< typename Device, typename Index >
void
AdaptiveCSRView< Device, Index >::load( File& file )
{
   Base::load( file );
   for( int i = 0; i < MaxValueSizeLog(); i++ )
      file >> this->blocksArray[ i ];
}

template< typename Device, typename Index >
void
AdaptiveCSRView< Device, Index >::printBlocks( int idx, std::ostream& os ) const
{
   if( idx == -1 ) {
      for( int i = 0; i < MaxValueSizeLog(); i++ )
         printBlocks( i, os );
      return;
   }
   os << "Blocks for sizeof( Value ) == 2^" << idx << '\n';
   auto blocks = this->getBlocks()[ idx ];
   for( int i = 0; i < blocks.getSize(); i++ ) {
      os << "Block " << i << " : " << blocks.getElement( i ) << '\n';
   }
}

}  // namespace TNL::Algorithms::Segments
