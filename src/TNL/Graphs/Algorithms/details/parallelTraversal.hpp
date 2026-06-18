// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/scan.h>
#include <TNL/Containers/Vector.h>

namespace TNL::Graphs::Algorithms::detail {

/**
 * \brief Compacts newly discovered vertices into the next frontier.
 *
 * Given a 0/1 \e marks vector (1 = vertex was discovered/improved in this
 * iteration), computes the inclusive prefix sum into \e marksScan and uses it
 * to write the indices of marked vertices densely into \e frontier.
 *
 * \param marks      – 0/1 vector indicating newly discovered vertices
 * \param marksScan  – scratch vector for the inclusive scan (resized automatically)
 * \param frontier   – output vector receiving the compacted vertex indices
 * \return           – the number of vertices in the new frontier
 */
template< typename DeviceType, typename IndexType >
IndexType
compactFrontier(
   Containers::Vector< IndexType, DeviceType, IndexType >& marks,
   Containers::Vector< IndexType, DeviceType, IndexType >& marksScan,
   Containers::Vector< IndexType, DeviceType, IndexType >& frontier )
{
   TNL::Algorithms::inclusiveScan( marks, marksScan );
   const IndexType n = marks.getSize();
   const IndexType frontierSize = marksScan.getElement( n - 1 );
   if( frontierSize == 0 )
      return 0;
   frontier = 0;
   auto frontierView = frontier.getView();
   auto marksScanView = marksScan.getView();
   auto f = [ = ] __cuda_callable__( const IndexType idx, const IndexType value ) mutable
   {
      if( idx == 0 ) {
         if( marksScanView[ 0 ] == 1 )
            frontierView[ 0 ] = idx;
      }
      else if( marksScanView[ idx ] - marksScanView[ idx - 1 ] == 1 )
         frontierView[ marksScanView[ idx ] - 1 ] = idx;
   };
   marksScan.forAllElements( f );
   return frontierSize;
}

}  // namespace TNL::Graphs::Algorithms::detail
