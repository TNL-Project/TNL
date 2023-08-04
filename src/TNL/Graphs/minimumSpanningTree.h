// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#include <iostream>
#include <vector>
#include <algorithm>
#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/Edge.h>
#include <TNL/Algorithms/Segments/GrowingSegments.h>

namespace TNL::Graphs {

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

   void mergeTrees( Index source, Index target ) {
      Index root_source = getRoot( source );
      Index root_target = getRoot( target );

      TNL_ASSERT_NE( root_source, root_target, "Roots must be different at this point." );
      if( rank[ root_source ] > rank[ root_target ] ) {
         parent[ root_target ] = root_source;
      } else {
         parent[ root_source ] = root_target;
         if( rank[ root_source ] == rank[ root_target ] ) {
            rank[ root_target ]++;
         }
      }
   }

   void getRoots( std::vector< Index >& roots ) const {
      for( Index u = 0; u < ( Index ) parent.size(); u++ ) {
         if( u == parent[ u ] )
            roots.push_back( u );
      }
   }

private:
   std::vector< Index > parent;
   std::vector< Index > rank;
};

template< typename Vector, typename Index >
Index getRoot( const Vector& parent, Index source ) {
   if( parent[ source ] != source )
      return getRoot( parent, parent[source] );
   return source;
}

template< typename InGraph,
          typename OutGraph = InGraph,
          typename RootsVector = Containers::Vector< typename InGraph::IndexType >,
          typename Real = typename InGraph::ValueType,
          typename Index = typename InGraph::IndexType >
void kruskal( const InGraph& graph, OutGraph& minimum_spanning_tree, RootsVector& roots )
{
   static_assert( InGraph::isUndirected(), "Both input and output graph must be undirected." );
   static_assert( OutGraph::isUndirected(), "Both input and output graph must be undirected." );

   using DeviceType = typename InGraph::DeviceType;
   using IndexType = typename InGraph::IndexType;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;

   Index n = graph.getNodeCount();
   std::vector< Edge< Real, Index > > edges;
   for( Index i = 0; i < n; i++ )
   {
      const auto row = graph.getAdjacencyMatrix().getRow(i);
      for( Index j = 0; j < row.getSize(); j++ ) {
         const Index& col = row.getColumnIndex( j );
         if( col < i && col != Matrices::paddingIndex< Index > )
            edges.emplace_back(i, col, row.getValue(j) );
      }
   }

   std::sort(edges.begin(), edges.end(), compareEdges< Real, Index >);

   Forest< Real, Index > forest(n);
   IndexVector nodeCapacities( n ), tree_filling( n, 0 );
   graph.getAdjacencyMatrix().getRowCapacities( nodeCapacities );
   minimum_spanning_tree.setNodeCount( n );
   minimum_spanning_tree.setNodeCapacities( nodeCapacities );

   for( const auto& edge : edges ) {
      auto source = edge.getSource();
      auto target = edge.getTarget();
      auto source_root = forest.getRoot( source );
      auto target_root = forest.getRoot( target );
      if( source_root != target_root ) {
         minimum_spanning_tree.getAdjacencyMatrix().getRow( source ).setElement( tree_filling[ source ]++, target, edge.getWeight() );
         if constexpr( OutGraph::isUndirected() && ! OutGraph::MatrixType::isSymmetric() )
            minimum_spanning_tree.getAdjacencyMatrix().getRow( target ).setElement( tree_filling[ target ]++, source, edge.getWeight() );
         forest.mergeTrees( source, target );
      }
   }
   std::vector< Index > roots_;
   forest.getRoots( roots_ );
   roots = RootsVector( roots_ );
}

template< typename InGraph,
          typename OutGraph = InGraph,
          typename Real = typename InGraph::ValueType,
          typename Index = typename InGraph::IndexType >
void parallelMST(const InGraph& graph, OutGraph& tree )
{
   using RealType = Real;
   using DeviceType = typename InGraph::DeviceType;
   using IndexType = Index;
   using IndexVector = Containers::Vector< IndexType, DeviceType, IndexType >;
   using RealVector = Containers::Vector< RealType, DeviceType, IndexType >;
   using InMatrixType = typename InGraph::MatrixType;
   using RowView = typename InMatrixType::ConstRowView;
   using SegmentsType = typename InMatrixType::SegmentsType;
   using GrowingSegmentsType = Algorithms::Segments::GrowingSegments< SegmentsType >;

   Index n = graph.getNodeCount();
   IndexVector starRootsSlots;
   graph.getAdjacencyMatrix().getRowCapacities( starRootsSlots );
   starRootsSlots = n;
   GrowingSegmentsType hook_candidates( starRootsSlots );
   tree.setNodeCount( n );
   tree.getAdjacencyMatrix().setRowCapacities( starRootsSlots );

   IndexVector p( n ), p_old( n, 0 ), q( n ), new_links_target( n, -1 ), star_link_source( n, -1 );
   RealVector new_links_weight( n, 0.0 );
   RealVector hook_candidates_weights( hook_candidates.getStorageSize(), std::numeric_limits< Real >::max() ),
              star_hook_candidates_weights( n, std::numeric_limits< Real >::max() ),
              star_hook_weights( n, std::numeric_limits< Real >::max() ),
              hook_weights( n, std::numeric_limits< Real >::max() );
   IndexVector hook_candidates_sources( hook_candidates.getStorageSize(), ( IndexType ) 0 ),
               hook_candidates_targets( hook_candidates.getStorageSize(), ( IndexType ) 0 ),
               star_hook_candidates_sources( n, ( IndexType ) 0 ),
               star_hook_candidates_targets( n, ( IndexType ) 0 ),
               star_hook_sources( n, ( IndexType ) 0 ),
               star_hook_targets( n, ( IndexType ) 0 ),
               hook_targets( n, ( IndexType ) -1 ),
               hook_sources( n, ( IndexType ) -1 );
   p.forAllElements( [] __cuda_callable__ ( Index i, Index& value ) { value = i; } );
   IndexVector treeFilling( n, 0 );
   Containers::VectorView< IndexType, DeviceType, IndexType > treeFillingView( treeFilling.getView() );

   auto hook_candidates_view = hook_candidates.getView();
   auto hook_candidates_weights_view = hook_candidates_weights.getView();
   auto hook_candidates_targets_view = hook_candidates_targets.getView();
   auto hook_candidates_sources_view = hook_candidates_sources.getView();
   auto star_hook_candidates_weights_view = star_hook_candidates_weights.getView();
   auto star_hook_candidates_targets_view = star_hook_candidates_targets.getView();
   auto star_hook_candidates_sources_view = star_hook_candidates_sources.getView();
   auto star_hook_weights_view = star_hook_weights.getView();
   auto star_hook_targets_view = star_hook_targets.getView();
   auto star_hook_sources_view = star_hook_sources.getView();
   auto hook_weights_view = hook_weights.getView();
   auto hook_targets_view = hook_targets.getView();
   auto hook_sources_view = hook_sources.getView();
   auto new_links_target_view = new_links_target.getView();
   auto new_links_weight_view = new_links_weight.getView();
   auto star_link_source_view = star_link_source.getView();
   auto tree_view = tree.getAdjacencyMatrix().getView();

   const IndexType paddingIndex = graph.getAdjacencyMatrix().getPaddingIndex();
   IndexType iter( 0 );
   Real sum( 0.0 );
   //std::cout << graph.getAdjacencyMatrix() << std::endl;
   while( p != p_old )
   {
      //std::cout << "====================  Iteration " << ++iter << " ========================== " << std::endl;
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
         //std::cout << "Row = " << source_node << " row size = " << row.getSize() << std::endl;
         for( Index j = 0; j < row.getSize(); j++ ) {
            const Index& target_node = row.getColumnIndex( j );
            //std::cout << "   Checking edge " << source_node << " -> " << target_node << " weight " << row.getValue( j ) << " min. weight " << minEdgeWeight << std::endl;
            if( target_node != paddingIndex && p_view[ source_node ] != p_view[ target_node ] && row.getValue( j ) < minEdgeWeight ) {
               minEdgeWeight = row.getValue( j );
               minEdgeTarget = target_node;
            }
         }
         //std::cout  << "Min. edge " << source_node << " -> " << minEdgeTarget << " weight " << minEdgeWeight << std::endl;
         if( minEdgeTarget != -1 ) {
            //std::cout << " Adding candidate edge " << source_node << " -> " << minEdgeTarget << " weight " << minEdgeWeight
            //          << " to star root " << p_view[ minEdgeTarget ];
            IndexType idx = hook_candidates_view.newSlot( p_view[ minEdgeTarget ] );
            hook_candidates_weights_view[ idx ] = minEdgeWeight;
            hook_candidates_targets_view[ idx ] = minEdgeTarget;
            hook_candidates_sources_view[ idx ] = source_node;
            //std::cout << " Adding candidate edge " << minEdgeTarget << " -> " << source_node << " weight " << minEdgeWeight
            //          << " to star root " << p_view[ minEdgeTarget ];
            idx = hook_candidates_view.newSlot( p_view[ source_node ] );
            hook_candidates_weights_view[ idx ] = minEdgeWeight;
            hook_candidates_targets_view[ idx ] = source_node;
            hook_candidates_sources_view[ idx ] = minEdgeTarget;

         }
      };
      graph.getAdjacencyMatrix().forAllRows( hooking );

      /*using SegmentView = typename GrowingSegmentsType::SegmentViewType;;
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
      } );*/


      // Find minimum weight hook candidates for each node
      using AuxType = Aux< RealType, IndexType >;
      //IndexVector star_slots( n, 0 );
      //auto star_slots_view = star_slots.getView();
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
         //star_slots_view[ p_old_view[ value.second ] ]++; //TODO: makes this atomic
         //star_slots_view[ p_old_view[ value.third ] ]++; //TODO: makes this atomic
      };
      hook_candidates.reduceAllSegments( hook_candidates_fetch, hook_candidates_reduction, hook_candidates_keep, AuxType( std::numeric_limits< RealType >::max(), -1, -1 ) );
      /*for( Index i = 0; i < hook_sources.getSize(); i++ ) {
         if( hook_sources_view[ i ] != -1 )
            std::cout << " > Best hook: " << hook_sources[ i ] << " -> " << i << " @ " << hook_weights[ i ] << std::endl;
      }*/

      // Find the minimum hook candidates for each star root using a sparse matrix
      // TODO: the following must be done in atomic way to run in parallel
      /*InMatrixType star_link_matrix( n, n );
      star_link_matrix.setRowCapacities( star_slots );
      auto star_matrix_view = star_link_matrix.getView();

      IndexVector star_segments_ptrs( n, 0 );
      auto star_segments_ptrs_view = star_segments_ptrs.getView();
      auto star_segments_storage_size = star_link_matrix.getSegments().getStorageSize();
      IndexVector star_link_sources( star_segments_storage_size, -1 ), star_link_targets( star_segments_storage_size, -1 );
      auto star_link_sources_view = star_link_sources.getView();
      auto star_link_targets_view = star_link_targets.getView();
      Algorithms::parallelFor< DeviceType >( ( IndexType ) 0, n,
         [=] __cuda_callable__ ( IndexType i ) mutable {
            if( hook_sources_view[ i ] != -1 ) {
               auto source_star = p_old_view[ hook_sources_view[ i ] ];
               auto target_star = p_old_view[ hook_targets_view[ i ] ];
               auto weight = hook_weights_view[ i ];
               auto source_local_idx = star_segments_ptrs_view[ source_star ]++; // TODO: make this atomic
               auto target_local_idx = star_segments_ptrs_view[ target_star ]++; // TODO: make this atomic
               auto source_row = star_matrix_view.getRow( source_star );
               auto target_row = star_matrix_view.getRow( target_star );
               source_row.setElement( source_local_idx, target_star, weight );
               target_row.setElement( target_local_idx, source_star, weight );
               auto source_global_idx = source_row.getGlobalIndex( source_local_idx );
               auto target_global_idx = target_row.getGlobalIndex( target_local_idx );
               star_link_sources_view[ source_row.getGlobalIndex( source_local_idx ) ] = hook_sources_view[ i ];
               star_link_targets_view[ source_row.getGlobalIndex( source_local_idx ) ] = hook_targets_view[ i ];
               star_link_sources_view[ target_row.getGlobalIndex( target_local_idx ) ] = hook_targets_view[ i ];
               star_link_targets_view[ target_row.getGlobalIndex( target_local_idx ) ] = hook_sources_view[ i ];
            }
         } );

      std::cout << "Star link matrix: " << std::endl << star_link_matrix << std::endl;

      star_link_matrix.forAllRows( [=] __cuda_callable__ ( const typename InMatrixType::RowView& row ) {
         auto min_weight = std::numeric_limits< RealType >::max();
         IndexType source = -1, target = -1;
         for( IndexType localIdx = 0; localIdx < row.getSize(); localIdx++ ) {
            if( row.getValue( localIdx ) < min_weight ) {
               min_weight = row.getValue( localIdx );
               auto globalIdx = row.getGlobalIndex( localIdx );
               source = star_link_sources_view[ globalIdx ];
               target = star_link_targets_view[ globalIdx ];
            }
         }
         if( source != -1 ) {
            std::cout << " > Best star link: " << source << " -> " << target << " @ " << min_weight << std::endl;
         }
      } );*/

      // The following is alternative to the above without using the matrix - but not working yet
      star_hook_candidates_sources_view = -1;
      star_hook_candidates_weights_view = std::numeric_limits< Real >::max();
      Algorithms::parallelFor< DeviceType >( ( IndexType ) 0, n,
         [=] __cuda_callable__ ( Index i ) mutable {
            if( hook_sources_view[ i ] != -1 ) {
               const auto weight = hook_weights_view[ i ];
               const auto source_star = p_old_view[ hook_sources_view[ i ] ];
               const auto target_star = p_old_view[ hook_targets_view[ i ] ];
               if( weight < star_hook_candidates_weights_view[ source_star ] ) {
                  star_hook_candidates_weights_view[ source_star ] = weight;                    // TODO: make this atomic
                  star_hook_candidates_sources_view[ source_star ] = hook_sources_view[ i ];
                  star_hook_candidates_targets_view[ source_star ] = hook_targets_view[ i ];
               }
            }
         } );

      /*std::cout << " Star hook candidates: " << std::endl;

      for( Index i = 0; i < n; i++ ) {
         if( star_hook_candidates_sources_view[ i ] != -1 )
            std::cout << " >> Best star hook: " << star_hook_candidates_sources_view[ i ] << " -> " << star_hook_candidates_targets_view[ i ]
                      << " @ " << star_hook_candidates_weights_view[ i ] << std::endl;
      }*/

      // TODO: the following must be done in atomic way to run in parallel
      star_hook_sources_view = -1;
      star_hook_weights_view = std::numeric_limits< Real >::max();
      Algorithms::parallelFor< DeviceType >( ( IndexType ) 0, n,
         [=] __cuda_callable__ ( Index i ) mutable {
            if( star_hook_candidates_sources_view[ i ] != -1 ) {
               const auto weight = star_hook_candidates_weights_view[ i ];
               const auto source_star = p_view[ star_hook_candidates_sources_view[ i ] ];
               const auto target_star = p_view[ star_hook_candidates_targets_view[ i ] ];
               if( weight < star_hook_weights_view[ target_star ] ) {
                  star_hook_weights_view[ target_star ] = weight;                                      // TODO: make this atomic
                  star_hook_sources_view[ target_star ] = star_hook_candidates_targets_view[ i ];
                  star_hook_targets_view[ target_star ] = star_hook_candidates_sources_view[ i ];
               }
            }
         } );

      /*std::cout << " Star hook candidates: " << std::endl;
      for( Index i = 0; i < n; i++ ) {
         if( star_hook_sources_view[ i ] != -1 )
            std::cout << " >>> Best star hook: " << star_hook_sources_view[ i ] << " -> " << star_hook_targets_view[ i ]
                      << " @ " << star_hook_weights_view[ i ] << std::endl;
      }*/

      //std::cout << "p_old= " << p_old << std::endl;
      Algorithms::parallelFor< DeviceType >( ( IndexType ) 0, n,
         [=] __cuda_callable__ ( Index i ) mutable {
            if( star_hook_sources_view[ i ] != -1 ) {
               TNL_ASSERT_EQ( i, p_old_view[ star_hook_sources_view[ i ] ], "" );
               const auto source_star = p_old_view[ star_hook_sources_view[ i ] ];
               const auto target_star = p_old_view[ star_hook_targets_view[ i ] ];
               //std::cout << " >>>> Star hook " << i << ": " << star_hook_sources_view[ i ] << " -> " << star_hook_targets_view[ i ]
               //          << " source star: " << source_star << " target star: " << target_star << " @ " << star_hook_weights_view[ i ] << std::endl;
               if( star_hook_sources_view[ target_star ] != -1 ) {
                  const auto next_star = p_view[ star_hook_targets_view[ target_star ] ];
                  if( next_star != i ) {
                     //std::cerr << " Triple star hooking found: " << star_hook_sources_view[ i ]
                     //          << " -> " << star_hook_targets_view[ i ] << " -> " << star_hook_targets_view[ target_star ] << std::endl;
                     const auto first_weight = star_hook_weights_view[ i ];
                     const auto second_weight = star_hook_weights_view[ target_star ];
                     if( first_weight > second_weight ) {
                     //   std::cout << " Erasing hook: " << star_hook_sources_view[ i ] << " -> " << star_hook_targets_view[ i ] << std::endl;
                        star_hook_sources_view[ i ] = -1;
                     }
                     else {
                     //   std::cout << " Erasing hook: " << star_hook_sources_view[ target_star ] << " -> " << star_hook_targets_view[ target_star ] << std::endl;
                        star_hook_sources_view[ target_star ] = -1;
                     }
                  }
               }
            }
         } );

      //getchar();


      // Perform the hooking
      new_links_target_view = -1;
      new_links_weight_view = 0.0;
      star_link_source_view = -1; // TODO: this is not necessary

      auto hooking_fetch = [=] __cuda_callable__ ( Index i ) mutable -> RealType {
         auto source = star_hook_sources_view[ i ]; //hook_sources_view[ i ];
         auto target = star_hook_targets_view[ i ]; //hook_targets_view[ i ];
         if( source != -1 ) {
            //std::cout << " Hooking " << source << " -> " << target << " parent node is " << p_old_view[ source ] << " weight " << hook_weights_view[ i ] << std::endl;
            p_view[ p_old_view[ target ] ] = p_old_view[ source ];
            new_links_target_view[ target ] = source; //source; //hook_sources_view[ i ];
            star_link_source_view[ p_old_view[ target ] ] = target;
            new_links_weight_view[ target ] = hook_weights_view[ i ];
            return hook_weights_view[ i ];
         }
         else
            return 0;
      };
      sum += Algorithms::reduce< DeviceType >( 0, p.getSize(), hooking_fetch, TNL::Plus{} );
      //Algorithms::parallelFor< DeviceType >( 0, p.getSize(), hooking_fetch );
      /*std::cout << " After hooking: p     = " << p     << "                         sum = " << sum << std::endl;
      std::cout << " After hooking: p_old = " << p_old << "                         sum = " << sum << std::endl;
      std::cout << " New links target     = " << new_links_target_view << std::endl;
      std::cout << " Star link source     = " << star_link_source_view << std::endl;*/

      // Find cycles
      auto cycles_fetch = [=] __cuda_callable__ ( Index i ) mutable -> RealType {
         /*auto& new_link_i = new_links_target_view[ i ];
         if( new_link_i != -1 && i < new_link_i && i == new_links_target_view[ new_link_i ] ) {
            std::cout << " Found cycle " << i << " -> " << new_link_i << " -> " << i << std::endl;
            new_links_target_view[ i ] = -1;
         }*/
         auto& p_i = p_view[ i ];
         if( i == p_old_view[ i ] &&    // i was a star root before the hooking
             i < p_i &&                 // we cancel only one edge of the cycle
             i == p_view[ p_i ] )       // i == p_p_i <=> there is a cycle
         {
            //std::cout << " Cycle detected: " << i << " -> " << p_i << ". Avoiding edge with weight " << hook_weights_view[ i ] << std::endl;
            TNL_ASSERT_NE( star_link_source_view[ p_i ], -1, "" );
            //std::cout << "Erasing new links target at postion: star_link_source_view[ " << p_i << " ]  = " << star_link_source_view[ p_i ] << std::endl;
            new_links_target_view[ star_link_source_view[ p_i ] ] = -1;
            p_i = i;
            return hook_weights_view[ i ];
         }
         else return 0;
      };
      auto add = TNL::Algorithms::reduce< DeviceType >( 0, p.getSize(), cycles_fetch, TNL::Plus{} );
      sum -= add;
      /*std::cout << " After cycles: p      = " << p     << "                         sum = " << sum << std::endl;
      std::cout << " After cycles: p_old  = " << p_old << "                         sum = " << sum << std::endl;
      std::cout << " New links target     = " << new_links_target_view << std::endl;
      std::cout << " Star link source     = " << star_link_source_view << std::endl;*/

      // Adding edges to the graph of the spanning tree
      Algorithms::parallelFor< DeviceType >( 0, n,
      [=] __cuda_callable__ ( Index i ) mutable {
         const IndexType& target = new_links_target_view[ i ];
         if( target != -1 ) {
            IndexType row = max( i, target );
            IndexType col = min( i, target );
            IndexType localIdx = Algorithms::AtomicOperations< DeviceType >::add( treeFillingView[ row ], 1 );
            tree_view.getRow( row ).setElement( localIdx, col, new_links_weight_view[ i ] );
            if constexpr( ! OutGraph::MatrixType::isSymmetric() ) {
               IndexType localIdx = Algorithms::AtomicOperations< DeviceType >::add( treeFillingView[ row ], 1 );
               tree_view.getRow( col ).setElement( localIdx, row, new_links_weight_view[ i ] );
            }
            //std::cout <<    " Adding edge " << row << " -> " << col << " with weight " << new_links_weight_view[ i ]
            //          << " to the output tree." << std::endl;
         }
      } );
      /*std::cout << "  Adding " << add << " to sum " << sum << std::endl;
      std::cout << "Tree:" << std::endl << tree << std::endl;
      std::cout << " After cycles removing:    p = " << p << "                 sum = " << sum << std::endl;*/

      // Perform shortcutting
      while( Algorithms::reduce< DeviceType >(
         0, p.getSize(),
         [=] __cuda_callable__ ( Index i ) mutable {
            auto& p_i = p_view[ i ];
            if( p_i != p_view[ p_i ] ) {
               //std::cout << " Shortcutting " << i << " to " << p_view[ p_i ] << std::endl;
               p_i = p_view[ p_i ];
               return 1;
            }
            else return 0; },
         TNL::Plus{} ) );

      //std::cout << " After shortcutting:       p = " << p << std::endl;

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
      //getchar();
   }
}

/**
 * \brief Computes minimum spanning tree of a graph.
 *
 * The input graph must be undirected. The output graph representing the minimum spanning tree must
 * be of the same type in this sense. If the input graph is not connected, the output graph will be a forest and the
 * \e roots vector will contain the roots of the trees in the forest.
 *
 * \tparam InGraph is the type of the input graph.
 * \tparam OutGraph is the type of the output graph.
 * \tparam RootsVector is the type of the vector containing the roots of the
 * \tparam Value is the type of the values of the input graph.
 * \tparam Index is the type of the indices of the input graph.
 *
 * \param graph is the input graph
 * \param spanning_tree is the output graph representing the minimum spanning tree.
 * \param roots is the vector containing the roots of the trees in the forest.
 */
template< typename InGraph,
          typename OutGraph = InGraph,
          typename RootsVector = Containers::Vector< typename InGraph::IndexType >,
          typename Value = typename InGraph::ValueType,
          typename Index = typename InGraph::IndexType >
void minimumSpanningTree( const InGraph& graph, OutGraph& spanning_tree, RootsVector& roots )
{
   static_assert( InGraph::isUndirected(), "The input graph must be undirected." );
   static_assert( OutGraph::isUndirected(), "The output graph must be undirected." );

   //using Device = typename InGraph::DeviceType;

   //if constexpr( std::is_same< Device, TNL::Devices::Sequential >::value )
      kruskal( graph, spanning_tree, roots );
   //else
   // TODO: The parallel MST algorithm is not working properly. It works only for sequential devices and in debug build.
   //   parallelMST( graph, spanning_tree );
}

} // namespace TNL::Graphs
