// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>

#include "ChunkedEllpackView.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
ChunkedEllpackView< Device, Index, Organization >::ChunkedEllpackView( Index size,
                                                                       Index storageSize,
                                                                       Index numberOfSlices,
                                                                       Index chunksInSlice,
                                                                       Index desiredChunkSize,
                                                                       typename Base::OffsetsView rowToChunkMapping,
                                                                       typename Base::OffsetsView rowToSliceMapping,
                                                                       typename Base::OffsetsView chunksToSegmentsMapping,
                                                                       typename Base::OffsetsView rowPointers,
                                                                       typename Base::SliceInfoContainerView slices )
: Base( size,
        storageSize,
        numberOfSlices,
        chunksInSlice,
        desiredChunkSize,
        std::move( rowToChunkMapping ),
        std::move( rowToSliceMapping ),
        std::move( chunksToSegmentsMapping ),
        std::move( rowPointers ),
        std::move( slices ) )
{}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
void
ChunkedEllpackView< Device, Index, Organization >::bind( ChunkedEllpackView view )
{
   Base::bind( view.size,
               view.storageSize,
               view.numberOfSlices,
               view.chunksInSlice,
               view.desiredChunkSize,
               std::move( view.rowToChunkMapping ),
               std::move( view.rowToSliceMapping ),
               std::move( view.chunksToSegmentsMapping ),
               std::move( view.rowPointers ),
               std::move( view.slices ) );
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
typename ChunkedEllpackView< Device, Index, Organization >::ViewType
ChunkedEllpackView< Device, Index, Organization >::getView()
{
   return { this->getSize(),
            this->getStorageSize(),
            this->getNumberOfSlices(),
            this->getChunksInSlice(),
            this->getDesiredChunkSize(),
            this->getRowToChunkMappingView(),
            this->getRowToSliceMappingView(),
            this->getChunksToSegmentsMappingView(),
            this->getRowPointersView(),
            this->getSlicesView() };
}

template< typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
ChunkedEllpackView< Device, Index, Organization >::getConstView() const -> ConstViewType
{
   return { this->getSize(),
            this->getStorageSize(),
            this->getNumberOfSlices(),
            this->getChunksInSlice(),
            this->getDesiredChunkSize(),
            this->getRowToChunkMappingView(),
            this->getRowToSliceMappingView(),
            this->getChunksToSegmentsMappingView(),
            this->getRowPointersView(),
            this->getSlicesView() };
}

template< typename Device, typename Index, ElementsOrganization Organization >
void
ChunkedEllpackView< Device, Index, Organization >::save( File& file ) const
{
   file.save( &this->size );
   file.save( &this->storageSize );
   file.save( &this->numberOfSlices );
   file.save( &this->chunksInSlice );
   file.save( &this->desiredChunkSize );
   file << this->rowToChunkMapping << this->rowToSliceMapping << this->chunksToSegmentsMapping << this->rowPointers
        << this->slices;
}

template< typename Device, typename Index, ElementsOrganization Organization >
void
ChunkedEllpackView< Device, Index, Organization >::load( File& file )
{
   file.load( &this->size );
   file.load( &this->storageSize );
   file.load( &this->numberOfSlices );
   file.load( &this->chunksInSlice );
   file.load( &this->desiredChunkSize );
   file >> this->rowToChunkMapping >> this->rowToSliceMapping >> this->chunksToSegmentsMapping >> this->rowPointers
      >> this->slices;
}

}  // namespace TNL::Algorithms::Segments
