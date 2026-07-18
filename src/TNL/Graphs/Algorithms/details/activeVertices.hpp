// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/uncompress.h>

namespace TNL::Graphs::Algorithms::detail {

/**
 * \brief Builds a boolean mask from a list of vertex indices.
 *
 * Sets \p activeVertices to a vector of size \c graph.getVertexCount() where
 * every position listed in \p vertexIndexes is 1 and all others are 0.
 *
 * Delegates to \ref TNL::Algorithms::uncompress with the graph vertex count
 * as the explicit mask size so that indices are validated against the actual
 * number of vertices.
 *
 * \throws std::invalid_argument if any index is outside [0, getVertexCount()).
 */
template< typename Graph, typename VertexIndexes, typename ActiveVector >
void
activateIndexedVertices( const Graph& graph, const VertexIndexes& vertexIndexes, ActiveVector& activeVertices )
{
   TNL::Algorithms::uncompress( vertexIndexes, activeVertices, graph.getVertexCount() );
}

}  // namespace TNL::Graphs::Algorithms::detail
