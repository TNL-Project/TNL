// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <TNL/Algorithms/reduce.h>
#include <TNL/Benchmarks/BenchmarkResult.h>
#include <TNL/Devices/Host.h>

namespace TNL::Benchmarks::Graphs {

/**
 * \brief Picks the source vertices for single-source kernels (BFS, SSSP) in the same way as the GAP reference code.
 *
 * GAP prescribes randomly selected vertices with non-zero (out-)degree and recommends the deterministic vertex selector
 * of its reference code (`SourcePicker` in `benchmark.h` and `UniDist` in `generator.h` of gapbs). This function
 * reproduces it exactly, including its 32-bit arithmetic, so the sources are identical to the ones used by gapbs
 * and in the published GAP results.
 *
 * \param vertexCount is the number of vertices of the graph, it must fit in 32 bits like in gapbs.
 * \param count is the number of sources to pick.
 * \param outDegree is a callable returning the out-degree of a vertex.
 */
template< typename Index, typename DegreeFunction >
[[nodiscard]] std::vector< Index >
pickGapSources( Index vertexCount, int count, DegreeFunction&& outDegree )
{
   if( vertexCount <= 0 || count <= 0 )
      return {};
   if( vertexCount > std::numeric_limits< std::int32_t >::max() )
      throw std::invalid_argument( "pickGapSources: the vertex count must fit in 32 bits like in gapbs." );

   bool hasEdges = false;
   for( Index v = 0; v < vertexCount && ! hasEdges; v++ )
      hasEdges = outDegree( v ) > 0;
   if( ! hasEdges )
      return {};

   // The following mirrors UniDist< NodeID = int32_t, std::mt19937_64 > of gapbs including the truncations to 32 bits.
   constexpr std::int64_t gapRandomSeed = 27491095;
   std::mt19937_64 rng( gapRandomSeed );
   using UnsignedNodeID = std::uint32_t;
   const std::int32_t maxValue = static_cast< std::int32_t >( vertexCount - 1 );
   const bool noMod = rng.max() == static_cast< UnsignedNodeID >( maxValue );
   const UnsignedNodeID mod = maxValue + 1;
   const UnsignedNodeID remainderSub1 = rng.max() % mod;
   const UnsignedNodeID cutoff = remainderSub1 == mod - 1 ? 0 : static_cast< UnsignedNodeID >( rng.max() - remainderSub1 );
   const auto uniform = [ & ]() -> std::int32_t
   {
      UnsignedNodeID randomNumber = rng();
      if( noMod )
         return randomNumber;
      if( cutoff != 0 ) {
         while( randomNumber >= cutoff )
            randomNumber = rng();
      }
      return randomNumber % mod;
   };

   std::vector< Index > sources;
   sources.reserve( count );
   while( static_cast< int >( sources.size() ) < count ) {
      std::int32_t source = 0;
      do {
         source = uniform();
      } while( outDegree( source ) == 0 );
      sources.push_back( source );
   }
   return sources;
}

/**
 * \brief Benchmark result extended by the number of traversed edges and the traversal rate.
 *
 * For the single-source kernels, `traversed edges` is the number of edges traversed from the source, i.e. the sum of the
 * out-degrees of all vertices reached from the source (the stored edges, so an undirected edge counts twice).
 * GTEPS is this number divided by the time, in billions of edges per second. It replaces the bandwidth, which
 * is meaningless for traversals that touch only the reached part of the graph.
 */
struct GraphBenchmarkResult : public BenchmarkResult
{
   double edges = std::numeric_limits< double >::quiet_NaN();
   double gteps = std::numeric_limits< double >::quiet_NaN();

   void
   setDerivedResults( std::size_t datasetSize, std::size_t operationsPerLoop ) override
   {
      BenchmarkResult::setDerivedResults( datasetSize, operationsPerLoop );
      gteps = edges / time / 1e9;
   }

   [[nodiscard]] HeaderElements
   getTableHeader() const override
   {
      HeaderElements header = BenchmarkResult::getTableHeader();
      // "edges" is the metadata column with the number of edges of the graph
      header.push_back( "traversed edges" );
      header.push_back( "GTEPS" );
      return header;
   }

   [[nodiscard]] RowElements
   getRowElements() const override
   {
      RowElements elements = BenchmarkResult::getRowElements();
      if( edges == edges ) {
         elements << static_cast< std::int64_t >( edges );
         elements << gteps;
      }
      else
         elements << "N/A"
                  << "N/A";
      return elements;
   }
};

/**
 * \brief Result of a verification of a graph algorithm output.
 */
struct GraphVerificationResult
{
   //! \brief Number of violated conditions; zero means that the output is correct.
   std::int64_t violations = 0;
   //! \brief Number of vertices reached from the source (single-source kernels only).
   std::int64_t reachedVertices = 0;
   //! \brief Sum of the out-degrees of the reached vertices (single-source kernels only).
   std::int64_t traversedEdges = 0;
   //! \brief Description of the first detected problem.
   std::string message;
};

namespace detail {

// Provides direct access to the CSR arrays of a host adjacency matrix.
template< typename Matrix >
struct HostCSR
{
   using IndexType = typename Matrix::IndexType;
   using RealType = typename Matrix::RealType;

   explicit HostCSR( const Matrix& matrix )
   : vertices( matrix.getRows() ),
     offsets( matrix.getSegments().getOffsets().getData() ),
     columns( matrix.getColumnIndexes().getData() )
   {
      if constexpr( ! Matrix::isBinary() )
         values = matrix.getValues().getData();
   }

   [[nodiscard]] RealType
   weight( IndexType k ) const
   {
      if constexpr( Matrix::isBinary() )
         return 1;
      else
         return values[ k ];
   }

   IndexType vertices;
   const IndexType* offsets;
   const IndexType* columns;
   const RealType* values = nullptr;
};

}  // namespace detail

/**
 * \brief Verifies the output of BFS by checking the properties of a correct solution.
 *
 * The conditions are those of the GAP specification extended by the distances, which together certify the result
 * without a reference solution:
 *
 * - `distances[ source ] == 0` and `predecessors[ source ] == source` (GAP) or -1 (TNL),
 * - unreached vertices have the distance and the predecessor -1,
 * - for each edge `u -> v` from a reached vertex `u`, `v` is reached and `distances[ v ] <= distances[ u ] + 1`,
 * - each reached vertex `v` other than the source has an edge `u -> v` with `distances[ v ] == distances[ u ] + 1`
 *   and, if the predecessors are given, `u == predecessors[ v ]`.
 *
 * The first two edge conditions imply that the distances are the exact BFS depths, so every vertex reachable
 * from the source is reached and the predecessors form a valid BFS tree.
 *
 * \param matrix is the CSR adjacency matrix on the host, the edges are its rows.
 * \param source is the source vertex.
 * \param distances are the computed distances (-1 for unreached vertices).
 * \param predecessors are the computed predecessors, or `nullptr` if only the distances are verified.
 */
template< typename Matrix, typename Index >
[[nodiscard]] GraphVerificationResult
verifyBFS( const Matrix& matrix, Index source, const Index* distances, const Index* predecessors = nullptr )
{
   using IndexType = typename Matrix::IndexType;
   const detail::HostCSR< Matrix > csr( matrix );
   const IndexType n = csr.vertices;
   GraphVerificationResult result;
   // GAP requires predecessors[ source ] == source, TNL uses -1 for the source
   if( distances[ source ] != 0
       || ( predecessors != nullptr && predecessors[ source ] != source && predecessors[ source ] != -1 ) )
   {
      result.violations++;
      result.message = "the source has a nonzero distance or is not its own predecessor";
   }

   std::unique_ptr< std::atomic< std::uint8_t >[] > hasParent( new std::atomic< std::uint8_t >[ n ] );
   for( IndexType v = 0; v < n; v++ )
      hasParent[ v ].store( 0, std::memory_order_relaxed );

   std::atomic< std::int64_t > traversedEdges = 0;
   const auto edgeViolations = TNL::Algorithms::reduce< TNL::Devices::Host >(
      IndexType( 0 ),
      n,
      [ & ]( IndexType u ) -> std::int64_t
      {
         const Index du = distances[ u ];
         if( du < 0 ) {
            const bool valid = du == -1 && ( predecessors == nullptr || predecessors[ u ] == -1 );
            return valid ? 0 : 1;
         }
         std::int64_t violations = 0;
         for( IndexType k = csr.offsets[ u ]; k < csr.offsets[ u + 1 ]; k++ ) {
            const IndexType v = csr.columns[ k ];
            const Index dv = distances[ v ];
            if( dv < 0 || dv > du + 1 )
               violations++;
            else if( dv == du + 1 && ( predecessors == nullptr || predecessors[ v ] == u ) )
               hasParent[ v ].store( 1, std::memory_order_relaxed );
         }
         traversedEdges.fetch_add( csr.offsets[ u + 1 ] - csr.offsets[ u ], std::memory_order_relaxed );
         return violations;
      },
      std::plus<>{},
      std::int64_t( 0 ) );
   if( edgeViolations > 0 && result.message.empty() )
      result.message = "an edge leaves the reached vertices or skips a BFS level, or an unreached vertex is not -1";
   result.violations += edgeViolations;

   std::int64_t missingParents = 0;
   for( IndexType v = 0; v < n; v++ ) {
      if( distances[ v ] >= 0 )
         result.reachedVertices++;
      if( v != source && distances[ v ] >= 0 && hasParent[ v ].load( std::memory_order_relaxed ) == 0 )
         missingParents++;
   }
   if( missingParents > 0 && result.message.empty() )
      result.message = "a reached vertex has no valid predecessor in the previous BFS level";
   result.violations += missingParents;
   result.traversedEdges = traversedEdges.load();
   return result;
}

/**
 * \brief Verifies a BFS output consisting of the parents only, as computed by the GAP reference code.
 *
 * The depths are computed by following the parents; a parent out of range or a cycle is a violation. Then the
 * parents and the depths are verified by \ref verifyBFS.
 *
 * \param matrix is the CSR adjacency matrix on the host.
 * \param source is the source vertex.
 * \param parents are the computed parents, negative for unreached vertices.
 */
template< typename Matrix, typename Parent >
[[nodiscard]] GraphVerificationResult
verifyBFSParents( const Matrix& matrix, typename Matrix::IndexType source, const Parent* parents )
{
   using IndexType = typename Matrix::IndexType;
   const IndexType n = matrix.getRows();
   constexpr IndexType unknown = -2;
   std::vector< IndexType > depths( n, unknown );
   std::vector< IndexType > normalizedParents( n );
   std::vector< IndexType > path;
   GraphVerificationResult result;
   for( IndexType v = 0; v < n; v++ ) {
      normalizedParents[ v ] = parents[ v ] < 0 ? -1 : parents[ v ];
      if( parents[ v ] >= n ) {
         result.violations++;
         result.message = "a parent is not a vertex index";
         return result;
      }
   }
   for( IndexType v = 0; v < n; v++ ) {
      // follow the parents until a vertex with a known depth, the source, or an unreached vertex
      IndexType u = v;
      path.clear();
      while( depths[ u ] == unknown ) {
         if( u == source ) {
            depths[ u ] = 0;
            break;
         }
         if( normalizedParents[ u ] < 0 ) {
            depths[ u ] = -1;
            break;
         }
         path.push_back( u );
         depths[ u ] = -3;  // on the current path
         u = normalizedParents[ u ];
         if( depths[ u ] == -3 ) {
            result.violations++;
            result.message = "the parents contain a cycle";
            return result;
         }
      }
      for( auto it = path.rbegin(); it != path.rend(); ++it ) {
         const IndexType parentDepth = depths[ normalizedParents[ *it ] ];
         depths[ *it ] = parentDepth < 0 ? -1 : parentDepth + 1;
         if( parentDepth < 0 ) {
            // a vertex with a parent that is not connected to the source
            result.violations++;
            if( result.message.empty() )
               result.message = "a vertex has a parent that is not reached from the source";
         }
      }
   }
   if( result.violations > 0 )
      return result;
   return verifyBFS( matrix, source, depths.data(), normalizedParents.data() );
}

/**
 * \brief Verifies the output of SSSP by checking the optimality conditions of the distances.
 *
 * The distances are correct if and only if
 *
 * - `distances[ source ] == 0`,
 * - for each edge `u -> v` with weight `w` from a reached vertex `u`, `v` is reached and
 *   `distances[ v ] <= distances[ u ] + w`,
 * - each reached vertex `v` other than the source has an edge `u -> v` with `distances[ v ] == distances[ u ] + w`.
 *
 * The weights must be non-negative. The sums are computed in the type of the distances, i.e. in the same way as in
 * the verified algorithm, so the conditions hold exactly also in floating-point arithmetic.
 *
 * \param matrix is the weighted CSR adjacency matrix on the host.
 * \param source is the source vertex.
 * \param distances are the computed distances, unreached vertices must have a negative or infinite distance.
 */
template< typename Matrix, typename Real >
[[nodiscard]] GraphVerificationResult
verifySSSP( const Matrix& matrix, typename Matrix::IndexType source, const Real* distances )
{
   using IndexType = typename Matrix::IndexType;
   const detail::HostCSR< Matrix > csr( matrix );
   const IndexType n = csr.vertices;
   const auto reached = [ & ]( IndexType v )
   {
      return distances[ v ] >= 0 && distances[ v ] < std::numeric_limits< Real >::infinity();
   };
   GraphVerificationResult result;
   if( distances[ source ] != 0 ) {
      result.violations++;
      result.message = "the source has a nonzero distance";
   }

   std::unique_ptr< std::atomic< std::uint8_t >[] > hasParent( new std::atomic< std::uint8_t >[ n ] );
   for( IndexType v = 0; v < n; v++ )
      hasParent[ v ].store( 0, std::memory_order_relaxed );

   std::atomic< std::int64_t > traversedEdges = 0;
   const auto edgeViolations = TNL::Algorithms::reduce< TNL::Devices::Host >(
      IndexType( 0 ),
      n,
      [ & ]( IndexType u ) -> std::int64_t
      {
         if( ! reached( u ) )
            return 0;
         std::int64_t violations = 0;
         for( IndexType k = csr.offsets[ u ]; k < csr.offsets[ u + 1 ]; k++ ) {
            const IndexType v = csr.columns[ k ];
            const Real w = csr.weight( k );
            if( w < 0 ) {
               violations++;
               continue;
            }
            const Real candidate = distances[ u ] + w;
            if( ! reached( v ) || distances[ v ] > candidate )
               violations++;
            else if( distances[ v ] == candidate )
               hasParent[ v ].store( 1, std::memory_order_relaxed );
         }
         traversedEdges.fetch_add( csr.offsets[ u + 1 ] - csr.offsets[ u ], std::memory_order_relaxed );
         return violations;
      },
      std::plus<>{},
      std::int64_t( 0 ) );
   if( edgeViolations > 0 && result.message.empty() )
      result.message = "a distance can be shortened by an edge, an edge leaves the reached vertices, or a weight is negative";
   result.violations += edgeViolations;

   std::int64_t missingParents = 0;
   for( IndexType v = 0; v < n; v++ ) {
      if( reached( v ) )
         result.reachedVertices++;
      if( v != source && reached( v ) && hasParent[ v ].load( std::memory_order_relaxed ) == 0 )
         missingParents++;
   }
   if( missingParents > 0 && result.message.empty() )
      result.message = "the distance of a reached vertex is not attained by any edge";
   result.violations += missingParents;
   result.traversedEdges = traversedEdges.load();
   return result;
}

/**
 * \brief Counts the connected components of an undirected graph with a sequential union-find.
 *
 * Vertices of zero degree are counted as separate components, as required by GAP.
 */
template< typename Matrix >
[[nodiscard]] std::int64_t
countConnectedComponents( const Matrix& matrix )
{
   using IndexType = typename Matrix::IndexType;
   const detail::HostCSR< Matrix > csr( matrix );
   const IndexType n = csr.vertices;
   std::vector< IndexType > parent( n );
   std::iota( parent.begin(), parent.end(), IndexType( 0 ) );
   const auto find = [ & ]( IndexType v )
   {
      while( parent[ v ] != v ) {
         parent[ v ] = parent[ parent[ v ] ];
         v = parent[ v ];
      }
      return v;
   };
   std::int64_t components = n;
   for( IndexType u = 0; u < n; u++ )
      for( IndexType k = csr.offsets[ u ]; k < csr.offsets[ u + 1 ]; k++ ) {
         IndexType a = find( u );
         IndexType b = find( csr.columns[ k ] );
         if( a != b ) {
            if( a < b )
               std::swap( a, b );
            parent[ a ] = b;
            components--;
         }
      }
   return components;
}

/**
 * \brief Verifies connected component labels.
 *
 * The labels are correct if and only if the end vertices of each edge have the same label (so each component has
 * a single label) and the number of distinct labels equals the number of components (so different components have
 * different labels). The labels must be vertex indexes, which holds for all benchmarked implementations.
 *
 * \param matrix is the CSR adjacency matrix of the undirected graph on the host.
 * \param labels are the computed component labels.
 * \param components is the number of components, see \ref countConnectedComponents.
 */
template< typename Matrix, typename Label >
[[nodiscard]] GraphVerificationResult
verifyCC( const Matrix& matrix, const Label* labels, std::int64_t components )
{
   using IndexType = typename Matrix::IndexType;
   const detail::HostCSR< Matrix > csr( matrix );
   const IndexType n = csr.vertices;
   GraphVerificationResult result;
   for( IndexType v = 0; v < n; v++ )
      if( labels[ v ] < 0 || labels[ v ] >= n ) {
         result.violations++;
         result.message = "a label is not a vertex index";
         return result;
      }

   result.violations = TNL::Algorithms::reduce< TNL::Devices::Host >(
      IndexType( 0 ),
      n,
      [ & ]( IndexType u ) -> std::int64_t
      {
         std::int64_t violations = 0;
         for( IndexType k = csr.offsets[ u ]; k < csr.offsets[ u + 1 ]; k++ )
            if( labels[ csr.columns[ k ] ] != labels[ u ] )
               violations++;
         return violations;
      },
      std::plus<>{},
      std::int64_t( 0 ) );
   if( result.violations > 0 )
      result.message = "the end vertices of an edge have different labels";

   std::vector< bool > used( n, false );
   std::int64_t distinct = 0;
   for( IndexType v = 0; v < n; v++ )
      if( ! used[ labels[ v ] ] ) {
         used[ labels[ v ] ] = true;
         distinct++;
      }
   if( distinct != components ) {
      result.violations++;
      if( result.message.empty() )
         result.message = "the number of labels " + std::to_string( distinct ) + " differs from the number of components "
                        + std::to_string( components );
   }
   return result;
}

}  // namespace TNL::Benchmarks::Graphs
