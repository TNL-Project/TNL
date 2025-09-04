// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "AdaptiveCSRView.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index >
__cuda_callable__
AdaptiveCSRView< Device, Index >::AdaptiveCSRView( const CSRView< Device, Index >& csrView, BlocksView* blocksView )
{
   using NonConstBlockView = typename AdaptiveCSRView< Device, std::remove_const_t< Index > >::BlocksView;
   Base::bind( csrView.getOffsets() );
   for( int i = 0; i < MaxValueSizeLog(); i++ )
      ( (NonConstBlockView*) &this->blocksArray[ i ] )
         ->bind( *(NonConstBlockView*) &blocksView[ i ] );  // TODO: rewrite without cast
}

template< typename Device, typename Index >
__cuda_callable__
AdaptiveCSRView< Device, Index >::AdaptiveCSRView( const CSRView< Device, Index >& csrView, BlocksType* blocksView )
{
   Base::bind( std::move( csrView.getOffsets() ) );
   for( int i = 0; i < MaxValueSizeLog(); i++ )
      this->blocksArray[ i ].bind( blocksView[ i ].getView() );
}

template< typename Device, typename Index >
__cuda_callable__
void
AdaptiveCSRView< Device, Index >::bind( AdaptiveCSRView view )
{
   Base::bind( std::move( view.offsets ) );
   for( int i = 0; i < MaxValueSizeLog(); i++ )
      this->blocksArray[ i ].bind( std::move( view.blocksArray[ i ] ) );
}

template< typename Device, typename Index >
__cuda_callable__
void
AdaptiveCSRView< Device, Index >::bind( OffsetsView offsets, BlocksView* blocks )
{
   Base::bind( std::move( offsets ) );
   for( int i = 0; i < MaxValueSizeLog(); i++ )
      this->blocksArray[ i ].bind( blocks[ i ] );
}

template< typename Device, typename Index >
__cuda_callable__
void
AdaptiveCSRView< Device, Index >::bind( OffsetsView offsets, BlocksType* blocks )
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
   using ConstBlocksView = typename AdaptiveCSRView< Device, std::add_const_t< Index > >::BlocksView;
   return ConstViewType( BaseConstViewType( this->getOffsets().getConstView() ),
                         (ConstBlocksView*) &this->blocksArray[ 0 ] );  // TODO: rewrite without cast
}

template< typename Device, typename Index >
auto __cuda_callable__
AdaptiveCSRView< Device, Index >::getBlocks() const->const BlocksView*
{
   return &this->blocksArray[ 0 ];
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
AdaptiveCSRView< Device, Index >::printBlocks( int idx ) const
{
   if( idx == -1 ) {
      for( int i = 0; i < MaxValueSizeLog(); i++ )
         printBlocks( i );
      return;
   }
   std::cout << "Blocks for sizeof( Value ) == 2^" << idx << '\n';
   auto blocks = this->getBlocks()[ idx ];
   for( int i = 0; i < blocks.getSize(); i++ ) {
      std::cout << "Block " << i << " : " << blocks.getElement( i ) << std::endl;
   }
}

}  // namespace TNL::Algorithms::Segments
