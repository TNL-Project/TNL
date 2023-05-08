// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#include <iostream>
#include <vector>
#include <algorithm>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/AtomicVectorView.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Algorithms/Graphs/Graph.h>
#include <TNL/Algorithms/Segments/GrowingSegments.h>

namespace TNL {
namespace Algorithms {
namespace Graphs {

// TODO: replace with std::touple
template< typename Real = double,
          typename Index = int >
struct Aux {

   Aux( int ) {}

   Aux( const Real& weight, Index source, Index target  )
      : first( weight ), second( source ), third( target ) {}

   Real first;
   Index second, third;
};

template< typename Index, typename Real >
bool compareEdges( const Edge< Index, Real > &a, const Edge< Index, Real > &b) {
    return a < b;
}

template< typename Real, typename Index >
struct Forest
{
   Forest( Index size) {
      parent.resize(size);
      rank.resize(size, 0);
      for (Index i = 0; i < size; ++i) {
         parent[i] = i;
      }
   }

   Index getRoot( Index u ) {
      if( parent[ u ] != u )
         parent[ u ] = getRoot( parent[u] );
      return parent[ u ];
   }

   void mergeTrees( Index u, Index v) {
      Index root_u = getRoot( u );
      Index root_v = getRoot( v );

      if( root_u == root_v ) {
            return;
      }

      if( rank[ root_u ] > rank[ root_v ] ) {
         parent[ root_v ] = root_u;
      } else {
         parent[ root_u ] = root_v;
         if( rank[ root_u ] == rank[ root_v ] ) {
            rank[ root_v ]++;
         }
      }
   }

private:
    std::vector< Index > parent;
    std::vector< Index > rank;
};

template< typename Matrix,
          typename Real = typename Matrix::RealType,
          typename Index = typename Matrix::IndexType >
void kruskal(const Matrix& graph, Matrix& minimum_spanning_tree )
   //Containers::Array< Edge< Real, Index >, TNL::Devices::Sequential, Index >& minimumSpanningTree )
{
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   Index n = graph.getRows();
   std::vector< Edge< Real, Index > > edges;
   for( Index i = 0; i < n; i++ )
   {
      const auto row = graph.getRow(i);
      for( Index j = 0; j < row.getSize(); j++ ) {
         const Index& col = row.getColumnIndex( j );
         if( i < col && col != graph.getPaddingIndex() )
            edges.emplace_back(i, col, row.getValue(j) );
      }
   }

   std::sort(edges.begin(), edges.end(), compareEdges< Real, Index >);

   Forest< Real, Index > forest(n);
   //std::vector< Edge< Real, Index > > minSpanningTree;
   IndexVector rowCapacities( n ), tree_filling( n, 0 );
   graph.getRowCapacities( rowCapacities );
   minimum_spanning_tree.setDimensions( n, n );
   minimum_spanning_tree.setRowCapacities( rowCapacities );


   for( const auto& edge : edges ) {
      auto u = edge.getSource();
      auto v = edge.getTarget();
      if( forest.getRoot( u ) != forest.getRoot( v ) ) {
         IndexType localIdx = tree_filling[ edge.getSource() ]++;
         minimum_spanning_tree.getRow( edge.getSource() ).setElement( localIdx, edge.getTarget(), edge.getWeight() );
         forest.mergeTrees( u, v );
      }
   }
}

template< typename Matrix,
          typename Real = typename Matrix::RealType,
          typename Index = typename Matrix::IndexType >
void parallelMST(const Matrix& graph, Matrix& tree )
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;
   using RealVector = Containers::Vector< RealType, DeviceType, IndexType >;
   using RowView = typename Matrix::ConstRowView;
   using SegmentsType = typename Matrix::SegmentsType;
   using GrowingSegmentsType = Segments::GrowingSegments< SegmentsType >;

   Index n = graph.getRows();
   IndexVector starRootsSlots;
   graph.getRowCapacities( starRootsSlots );
   starRootsSlots = n;
   GrowingSegmentsType hook_candidates( starRootsSlots );
   tree.setDimensions( n, n );
   tree.setRowCapacities( starRootsSlots );

   IndexVector p( n ), p_old( n, 0 ), q( n ), new_links_target( n, -1 );
   RealVector new_links_weight( n, 0.0 );
   RealVector hook_candidates_weights( hook_candidates.getStorageSize(), std::numeric_limits< Real >::max() ),
              hook_weights( n, std::numeric_limits< Real >::max() );
   IndexVector hook_candidates_sources( hook_candidates.getStorageSize(), ( IndexType ) 0 ),
               hook_candidates_targets( hook_candidates.getStorageSize(), ( IndexType ) 0 ),
               hook_targets( n, ( IndexType ) -1 ),
               hook_sources( n, ( IndexType ) -1 );
   p.forAllElements( [] __cuda_callable__ ( Index i, Index& value ) { value = i; } );
   IndexVector treeFilling( n, 0 );
   Containers::AtomicVectorView< IndexType, DeviceType, IndexType > treeFillingView( treeFilling.getView() );

   auto hook_candidates_view = hook_candidates.getView();
   auto hook_candidates_weights_view = hook_candidates_weights.getView();
   auto hook_candidates_targets_view = hook_candidates_targets.getView();
   auto hook_candidates_sources_view = hook_candidates_sources.getView();
   auto hook_weights_view = hook_weights.getView();
   auto hook_targets_view = hook_targets.getView();
   auto hook_sources_view = hook_sources.getView();
   auto new_links_target_view = new_links_target.getView();
   auto new_links_weight_view = new_links_weight.getView();
   auto tree_view = tree.getView();

   IndexType iter( 0 );
   Real sum( 0.0 );
   while( p != p_old )
   {
      std::cout << "Iteration " << ++iter << std::endl;
      p_old = p;
      auto p_view = p.getView();
      auto p_old_view = p_old.getView();

      // Erase segments for finding minimum weight hook candidates
      // TODO: hook_candidates_sources_view = -1 might be also a good idea
      /*hook_candidates.forAllElements( [=] __cuda_callable__ ( Index segmentIdx, Index localIdx, Index globalIdx ) mutable {
         hook_candidates_sources_view[ globalIdx] = 0;
         hook_candidates_weights_view[ globalIdx ] = std::numeric_limits< Real >::max();
      } );*/
      //hook_candidates.setSegmentsSizes( starRootsSlots );
      hook_candidates_targets_view = -1;
      hook_candidates_sources_view = -1;
      hook_candidates_weights_view = std::numeric_limits< Real >::max();
      hook_candidates.clear();

      // Find hook candidates
      auto hooking = [=] __cuda_callable__ ( RowView& row ) mutable {
         const Index& source_node = row.getRowIndex();
         Index minEdgeTarget = -1;
         Real minEdgeWeight = std::numeric_limits< Real >::max();
         //std::cout << "Source node " << source_node << std::endl;
         for( Index j = 0; j < row.getSize(); j++ ) {
            const Index& target_node = row.getColumnIndex( j );
            /*std::cout << " target_node = " << target_node
                      << " p_old[ target_node ] = " << p_old[ target_node ]
                      << " p_old[ source_node ] = " << p_old[ source_node ]
                      << " row.getValue( j ) = " << row.getValue( j )
                      << " minEdgeValue = " << minEdgeWeight
                      << std::endl;*/
            if( target_node != graph.getPaddingIndex() && p_view[ source_node ] != p_view[ target_node ] && row.getValue( j ) < minEdgeWeight ) {
               minEdgeWeight = row.getValue( j );
               minEdgeTarget = target_node;
            }
         }
         //std::cout  << "Min. edge " << source_node << " -> " << minEdgeTarget << " weight " << minEdgeWeight << std::endl;
         if( minEdgeTarget != -1 ) {
            std::cout << " Adding candidate edge " << source_node << " -> " << minEdgeTarget << " weight " << minEdgeWeight
                      << " to star root " << p_view[ minEdgeTarget ];
            IndexType idx = hook_candidates_view.newSlot( p_view[ minEdgeTarget ] );
            std::cout <<  " at position idx = " << idx << " / " << hook_candidates_view.getSize() << std::endl;
            hook_candidates_weights_view[ idx ] = minEdgeWeight;
            hook_candidates_targets_view[ idx ] = minEdgeTarget;
            hook_candidates_sources_view[ idx ] = source_node;
         }
      };
      graph.forAllRows( hooking );

      using SegmentView = typename GrowingSegmentsType::SegmentViewType;;
      hook_candidates.sequentialForAllSegments( [=] __cuda_callable__ ( const SegmentView& segment ) {
         auto segmentIdx = segment.getSegmentIndex();
         std::cout << " Hook candidate for node " << segmentIdx << " : ";
         for( Index localIdx = 0; localIdx < segment.getSize(); localIdx++ )
         {
            auto globalIdx = segment.getGlobalIndex( localIdx );
            if( hook_candidates_sources_view[ globalIdx ] != -1 )
               std::cout << hook_candidates_targets_view[ globalIdx ] << " -> "
                         << hook_candidates_sources_view[ globalIdx ] << " @ "
                         << hook_candidates_weights_view[ globalIdx ] << ", ";
         }
         std::cout << std::endl;
      } );


      // Find minimum weight hook candidates
      using AuxType = Aux< RealType, IndexType >;
      auto hook_candidates_fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx, bool& compute ) {
         return AuxType{ hook_candidates_weights_view[ globalIdx ],
                         hook_candidates_sources_view[ globalIdx ],
                         hook_candidates_targets_view[ globalIdx ] };
      };
      auto hook_candidates_reduction = [=] __cuda_callable__ ( const AuxType& a, const AuxType& b ) {
         if( a.first == b.first )
            return a.second < b.second ? a : b;
         else return a.first < b.first ? a : b;
      };
      auto hook_candidates_keep = [=] __cuda_callable__ ( IndexType segmentIdx, const AuxType& value ) mutable {
         hook_weights_view[ segmentIdx ] = value.first;
         hook_sources_view[ segmentIdx ] = value.second;
         hook_targets_view[ segmentIdx ] = value.third;
      };
      hook_candidates.reduceAllSegments( hook_candidates_fetch, hook_candidates_reduction, hook_candidates_keep, AuxType( std::numeric_limits< RealType >::max(), -1, -1 ) );
      /*for( Index i = 0; i < hook_sources.getSize(); i++ ) {
         if( hook_sources_view[ i ] != -1 )
            std::cout << " Best hook: " << hook_sources[ i ] << " -> " << i << " @ " << hook_weights[ i ] << std::endl;
      }*/

      // Perform the hooking
      new_links_target_view = -1;
      new_links_weight_view = 0.0;
      auto hooking_fetch = [=] __cuda_callable__ ( Index i ) mutable {
         auto source = hook_sources_view[ i ];
         auto target = hook_targets_view[ i ];
         if( source != -1 ) {
            std::cout << " Hooking " << source << " -> " << i << " parent node is " << p_old_view[ source ] << " weight " << hook_weights_view[ i ] << std::endl;
            p_view[ i ] = p_old_view[ source ];
            new_links_target_view[ target ] = source; //source; //hook_sources_view[ i ];
            new_links_weight_view[ target ] = hook_weights_view[ i ];
            return hook_weights_view[ i ];
         }
         else
            return 0.0;
      };
      sum += Algorithms::reduce< DeviceType >( 0, p.getSize(), hooking_fetch, TNL::Plus{} );
      //Algorithms::parallelFor< DeviceType >( 0, p.getSize(), hooking_fetch );
      std::cout << " After hooking: p = " << p << "                         sum = " << sum << std::endl;

      // Find cycles
      auto cycles_fetch = [=] __cuda_callable__ ( Index i ) mutable {
         auto& p_i = p_view[ i ];
         if( i == p_old_view[ i ] &&    // i was a star root before the hooking
             i < p_i &&                 // we cancel only one edge of the cycle
             i == p_view[ p_i ] )       // i == p_p_i <=> there is a cycle
         {
            std::cout << " Cycle detected: " << i << " -> " << p_i << ". Avoiding edge with weight " << hook_weights_view[ i ] << std::endl;
            p_i = i;
            new_links_target_view[ i ] = -1;
            return hook_weights_view[ i ];
         }
         else return 0.0;
      };
      auto add = TNL::Algorithms::reduce< DeviceType >( 0, p.getSize(), cycles_fetch, TNL::Plus{} );
      sum -= add;

      // Adding edges to the graph of the spanning tree
      Algorithms::parallelFor< DeviceType >( 0, n,
      [=] __cuda_callable__ ( Index i ) mutable {
         if( new_links_target_view[ i ] != -1 ) {
            IndexType localIdx = treeFillingView.atomicAdd( i, 1 );
            tree_view.getRow( new_links_target_view[ i ] ).setElement( localIdx, i, new_links_weight_view[ i ] );
            std::cout <<    " Adding edge " << i << " -> " << new_links_target_view[ i ] << " with weight " << new_links_weight_view[ i ]
                      << " to the output tree." << std::endl;
         }
      } );
      std::cout << "  Adding " << add << " to sum " << sum << std::endl;
      std::cout << " After cycles removing:    p = " << p << "                 sum = " << sum << std::endl;

      // Perform shortcutting
      p.forAllElements( [=] __cuda_callable__ ( Index i, Index& p_i ) mutable {
         if( p_i != p_view[ p_i ] ) {
            std::cout << " Shortcutting " << i << " to " << p_view[ p_i ] << std::endl;
            p_i = p_view[ p_i ];
         }
      } );

      std::cout << " After shortcutting:       p = " << p << std::endl;

      // Updating star roots slots after star roots merging
      /*std::cout << "Star roots slots:              " << starRootsSlots << std::endl;
      auto slots_view = starRootsSlots.getView();
      starRootsSlots.forAllElements( [=] __cuda_callable__ ( Index i, Index& slot ) mutable {
         const Index& p_old_i = p_old_view[ i ];
         const Index& p_i = p_view[ i ];
         if( p_i != p_old_i ) {
            slots_view[ p_i ] += slots_view[ p_old_i ]; // ?????? TODO ATOMIC asi
            slots_view[ p_old_i ] = 0;
         }
      } );
      std::cout << "Star roots slots after update: " << starRootsSlots << std::endl;*/
      getchar();
   }
}

template< typename Matrix,
          typename Real = typename Matrix::RealType,
          typename Index = typename Matrix::IndexType >
void minimumSpanningTree( const Matrix& adjacencyMatrix, Matrix& spanning_tree )
   //Containers::Array< Edge< Real, Index >, typename Matrix::DeviceType >& spanning_tree )
{
   TNL_ASSERT_TRUE( adjacencyMatrix.getRows() == adjacencyMatrix.getColumns(), "Adjacency matrix must be square matrix." );

   using Device = typename Matrix::DeviceType;

   //if constexpr( std::is_same< Device, TNL::Devices::Sequential >::value )
   //   kruskal( adjacencyMatrix, spanning_tree );
   //else
      parallelMST( adjacencyMatrix, spanning_tree );
}
} // namespace Graphs
} // namespace Algorithms
} // namespace TNL
