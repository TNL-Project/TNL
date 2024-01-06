// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "SlicedEllpackView.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
SlicedEllpackView< Device, Index, Organization, SliceSize >::SlicedEllpackView( Index size,
                                                                                Index alignedSize,
                                                                                Index segmentsCount,
                                                                                typename Base::OffsetsView sliceOffsets,
                                                                                typename Base::OffsetsView sliceSegmentSizes )
: Base( size, alignedSize, segmentsCount, std::move( sliceOffsets ), std::move( sliceSegmentSizes ) )
{}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
void
SlicedEllpackView< Device, Index, Organization, SliceSize >::bind( SlicedEllpackView view )
{
   Base::bind( view.getSize(),
               view.getStorageSize(),
               view.getSegmentsCount(),
               view.getSliceOffsetsView(),
               view.getSliceSegmentSizesView() );
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
typename SlicedEllpackView< Device, Index, Organization, SliceSize >::ViewType
SlicedEllpackView< Device, Index, Organization, SliceSize >::getView()
{
   return { this->getSize(),
            this->getStorageSize(),
            this->getSegmentsCount(),
            this->getSliceOffsetsView(),
            this->getSliceSegmentSizesView() };
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
__cuda_callable__
auto
SlicedEllpackView< Device, Index, Organization, SliceSize >::getConstView() const -> ConstViewType
{
   return { this->getSize(),
            this->getStorageSize(),
            this->getSegmentsCount(),
            this->getSliceOffsetsView(),
            this->getSliceSegmentSizesView() };
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
void
SlicedEllpackView< Device, Index, Organization, SliceSize >::save( File& file ) const
{
   file.save( &this->size );
   file.save( &this->storageSize );
   file.save( &this->segmentsCount );
   file << this->sliceOffsets;
   file << this->sliceSegmentSizes;
}

template< typename Device, typename Index, ElementsOrganization Organization, int SliceSize >
void
SlicedEllpackView< Device, Index, Organization, SliceSize >::load( File& file )
{
   file.load( &this->size );
   file.load( &this->storageSize );
   file.load( &this->segmentsCount );
   file >> this->sliceOffsets;
   file >> this->sliceSegmentSizes;
}

}  // namespace TNL::Algorithms::Segments
