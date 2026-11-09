// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/SparseMatrix.h>

namespace TNL::Benchmarks::Graphs {

/**
 * \brief Header of a graph in the serialized format of the GAP Benchmark Suite reference code (gapbs).
 *
 * The files are written by the gapbs converter (see `get-gap-graphs.py`).
 * The `.sg` files store unweighted and the `.wsg` files weighted graphs.
 * The layout of a file is:
 *
 * - `bool directed`,
 * - `int64_t edges` (the number of stored edges, i.e. both directions of an undirected edge are counted),
 * - `int64_t vertices`,
 * - `int64_t offsets[ vertices + 1 ]` of the outgoing edges,
 * - outgoing edges as `int32_t` targets (`.sg`) or pairs of `int32_t` targets and weights (`.wsg`),
 * - for directed graphs only, the offsets and the edges of the transposed graph in the same layout.
 */
struct GapGraphHeader
{
   bool directed = false;
   bool weighted = false;
   std::int64_t vertices = 0;
   std::int64_t edges = 0;
};

[[nodiscard]] inline bool
isGapGraphFile( const std::string& fileName )
{
   const auto endsWith = [ & ]( const std::string& suffix )
   {
      return fileName.size() >= suffix.size()
          && fileName.compare( fileName.size() - suffix.size(), suffix.size(), suffix ) == 0;
   };
   return endsWith( ".sg" ) || endsWith( ".wsg" );
}

/**
 * \brief Returns the name of the file with the symmetrized graph, i.e. `<name>U.sg` for `<name>.sg`.
 *
 * This is the naming convention of the GAP benchmark Makefile and of `get-gap-graphs.py`.
 */
[[nodiscard]] inline std::string
getGapUndirectedFileName( const std::string& fileName )
{
   const auto dot = fileName.find_last_of( '.' );
   if( dot == std::string::npos )
      return fileName + "U";
   return fileName.substr( 0, dot ) + "U" + fileName.substr( dot );
}

inline GapGraphHeader
readGapGraphHeader( std::ifstream& file, const std::string& fileName )
{
   GapGraphHeader header;
   const auto dot = fileName.find_last_of( '.' );
   header.weighted = dot != std::string::npos && fileName.substr( dot ) == ".wsg";
   file.read( reinterpret_cast< char* >( &header.directed ), sizeof( bool ) );
   file.read( reinterpret_cast< char* >( &header.edges ), sizeof( std::int64_t ) );
   file.read( reinterpret_cast< char* >( &header.vertices ), sizeof( std::int64_t ) );
   if( ! file || header.vertices < 0 || header.edges < 0 )
      throw std::runtime_error( "Invalid header of the GAP graph file " + fileName );
   return header;
}

[[nodiscard]] inline GapGraphHeader
readGapGraphHeader( const std::string& fileName )
{
   std::ifstream file( fileName, std::ios::binary );
   if( ! file )
      throw std::runtime_error( "Cannot open the GAP graph file " + fileName );
   return readGapGraphHeader( file, fileName );
}

/**
 * \brief Reads the outgoing edges of a graph in the gapbs serialized format into a CSR adjacency matrix.
 *
 * The transposed graph stored in the files of directed graphs is not read.
 * The weights of `.wsg` files are stored as the matrix values. For `.sg` files, all values are set to one.
 * If the matrix is binary, the weights are ignored.
 *
 * \param fileName is the name of the `.sg` or `.wsg` file.
 * \param matrix is the output adjacency matrix.
 * \return the header of the file.
 */
template< typename Real, typename Index >
GapGraphHeader
readGapGraph( const std::string& fileName, TNL::Matrices::SparseMatrix< Real, TNL::Devices::Host, Index >& matrix )
{
   using MatrixType = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Host, Index >;
   std::ifstream file( fileName, std::ios::binary );
   if( ! file )
      throw std::runtime_error( "Cannot open the GAP graph file " + fileName );
   const GapGraphHeader header = readGapGraphHeader( file, fileName );
   if( header.vertices > static_cast< std::int64_t >( std::numeric_limits< Index >::max() )
       || header.edges > static_cast< std::int64_t >( std::numeric_limits< Index >::max() ) )
      throw std::runtime_error(
         "The graph in " + fileName + " with " + std::to_string( header.vertices ) + " vertices and "
         + std::to_string( header.edges ) + " edges does not fit in the index type." );
   const Index vertices = header.vertices;

   std::vector< std::int64_t > offsets( header.vertices + 1 );
   file.read( reinterpret_cast< char* >( offsets.data() ), offsets.size() * sizeof( std::int64_t ) );
   if( ! file || offsets.front() != 0 || offsets.back() != header.edges )
      throw std::runtime_error( "Invalid offsets in the GAP graph file " + fileName );

   TNL::Containers::Vector< Index, TNL::Devices::Host, Index > capacities( vertices );
   for( Index i = 0; i < vertices; i++ )
      capacities[ i ] = offsets[ i + 1 ] - offsets[ i ];
   matrix.setDimensions( vertices, vertices );
   matrix.setRowCapacities( capacities );

   // The CSR format stores the column indexes in the order of the rows, i.e. in the same order as the file.
   Index* columns = matrix.getColumnIndexes().getData();
   Real* values = nullptr;
   if constexpr( ! MatrixType::isBinary() )
      values = matrix.getValues().getData();

   const std::size_t edgeSize = header.weighted ? 2 : 1;
   const std::int64_t chunkSize = 1 << 24;
   std::vector< std::int32_t > buffer( chunkSize * edgeSize );
   for( std::int64_t begin = 0; begin < header.edges; begin += chunkSize ) {
      const std::int64_t size = std::min( chunkSize, header.edges - begin );
      file.read( reinterpret_cast< char* >( buffer.data() ), size * edgeSize * sizeof( std::int32_t ) );
      if( ! file )
         throw std::runtime_error( "Unexpected end of the GAP graph file " + fileName );
      for( std::int64_t i = 0; i < size; i++ ) {
         const std::int32_t target = buffer[ i * edgeSize ];
         if( target < 0 || target >= header.vertices )
            throw std::runtime_error( "Invalid vertex index in the GAP graph file " + fileName );
         columns[ begin + i ] = target;
         if constexpr( ! MatrixType::isBinary() )
            values[ begin + i ] = header.weighted ? static_cast< Real >( buffer[ i * edgeSize + 1 ] ) : Real( 1 );
      }
   }
   return header;
}

}  // namespace TNL::Benchmarks::Graphs
