// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Graphs/Graph.h>

namespace TNL::Graphs {

//! \brief Computes the total weight of all edges in the graph.
template< typename Graph >
[[nodiscard]] typename Graph::ValueType
getTotalWeight( const Graph& graph );

}  // namespace TNL::Graphs

#include "GraphOperations.hpp"
