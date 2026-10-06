// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <TNL/Timer.h>
#include <TNL/Matrices/MatrixBase.h>
#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Matrices/detail/MatrixElements.h>

namespace TNL::Matrices {

namespace detail {

// Split a line of an MTX file into words separated by any whitespace, which also drops the '\r' of Windows line endings.
inline std::vector< std::string >
splitMtxLine( const std::string& line )
{
   std::vector< std::string > words;
   std::istringstream stream( line );
   for( std::string word; stream >> word; )
      words.push_back( std::move( word ) );
   return words;
}

// Split a line of an MTX file into at most maxWords words without allocating memory. Returns the number of words, which is
// greater than maxWords if there are more words in the line.
template< std::size_t maxWords >
std::size_t
splitMtxLine( std::string_view line, std::array< std::string_view, maxWords >& words )
{
   std::size_t count = 0;
   std::size_t position = 0;
   while( true ) {
      while( position < line.size() && std::isspace( static_cast< unsigned char >( line[ position ] ) ) )
         position++;
      if( position == line.size() )
         return count;
      const std::size_t begin = position;
      while( position < line.size() && ! std::isspace( static_cast< unsigned char >( line[ position ] ) ) )
         position++;
      if( count == maxWords )
         return count + 1;
      words[ count++ ] = line.substr( begin, position - begin );
   }
}

inline std::string
mtxToLowerCase( std::string word )
{
   std::transform(
      word.begin(),
      word.end(),
      word.begin(),
      []( unsigned char c )
      {
         return std::tolower( c );
      } );
   return word;
}

[[noreturn]] inline void
throwMtxError( long lineNumber, const std::string& message )
{
   throw std::runtime_error( "Error in the MTX file at line " + std::to_string( lineNumber ) + ": " + message );
}

inline long long
parseMtxInteger( std::string_view word, long lineNumber )
{
   long long value = 0;
   const auto [ end, error ] = std::from_chars( word.data(), word.data() + word.size(), value );
   if( error != std::errc{} || end != word.data() + word.size() )
      throwMtxError( lineNumber, "'" + std::string( word ) + "' is not a valid integer." );
   return value;
}

inline double
parseMtxReal( std::string_view word, long lineNumber )
{
   // std::from_chars does not accept the plus sign
   std::string_view number = word;
   if( number.size() > 1 && number[ 0 ] == '+' && number[ 1 ] != '-' && number[ 1 ] != '+' )
      number.remove_prefix( 1 );
   double value = 0;
#if defined( __cpp_lib_to_chars ) && __cpp_lib_to_chars >= 201611L
   const auto [ end, error ] = std::from_chars( number.data(), number.data() + number.size(), value );
   const bool valid = error == std::errc{} && end == number.data() + number.size();
#else
   // older standard libraries do not support std::from_chars for floating-point numbers
   const std::string text( number );
   char* end = nullptr;
   errno = 0;
   value = std::strtod( text.c_str(), &end );
   const bool valid = ! text.empty() && *end == '\0' && errno != ERANGE;
#endif
   if( ! valid )
      throwMtxError( lineNumber, "'" + std::string( word ) + "' is not a valid number." );
   return value;
}

template< typename Index >
Index
parseMtxSize( std::string_view word, long lineNumber )
{
   const long long value = parseMtxInteger( word, lineNumber );
   if( value < 0 || value > static_cast< long long >( std::numeric_limits< Index >::max() ) )
      throwMtxError( lineNumber, "the size " + std::string( word ) + " is out of the range of the index type." );
   return static_cast< Index >( value );
}

// Parse a 1-based row or column index and return the 0-based one.
template< typename Index >
Index
parseMtxIndex( std::string_view word, Index size, long lineNumber )
{
   const long long value = parseMtxInteger( word, lineNumber );
   if( value < 1 || value > static_cast< long long >( size ) )
      throwMtxError(
         lineNumber, "the index " + std::string( word ) + " is out of the range 1 to " + std::to_string( size ) + "." );
   return static_cast< Index >( value - 1 );
}

}  // namespace detail

template< typename Matrix, typename Device >
void
MatrixReader< Matrix, Device >::readMtx( const std::string& fileName, Matrix& matrix, bool verbose )
{
   MtxReaderOptions options;
   options.verbose = verbose;
   readMtx( fileName, matrix, options );
}

template< typename Matrix, typename Device >
void
MatrixReader< Matrix, Device >::readMtx( std::istream& str, Matrix& matrix, bool verbose )
{
   MtxReaderOptions options;
   options.verbose = verbose;
   readMtx( str, matrix, options );
}

template< typename Matrix, typename Device >
void
MatrixReader< Matrix, Device >::readMtx( const std::string& fileName, Matrix& matrix, const MtxReaderOptions& options )
{
   HostMatrix hostMatrix;
   MatrixReader< HostMatrix >::readMtx( fileName, hostMatrix, options );
   matrix = hostMatrix;
}

template< typename Matrix, typename Device >
void
MatrixReader< Matrix, Device >::readMtx( std::istream& file, Matrix& matrix, const MtxReaderOptions& options )
{
   HostMatrix hostMatrix;
   MatrixReader< HostMatrix >::readMtx( file, hostMatrix, options );
   matrix = hostMatrix;
}

template< typename Matrix, typename Device >
bool
MatrixReader< Matrix, Device >::isSymmetric( const std::string& fileName )
{
   return MatrixReader< HostMatrix >::isSymmetric( fileName );
}

template< typename Matrix, typename Device >
bool
MatrixReader< Matrix, Device >::isSymmetric( std::istream& file )
{
   return MatrixReader< HostMatrix >::isSymmetric( file );
}

// MatrixReader specialization for TNL::Devices::Host.

// This is to prevent Doxygen warnings due to hidden class.
/// \cond
template< typename Matrix >
void
MatrixReader< Matrix, TNL::Devices::Host >::readMtx( const std::string& fileName, Matrix& matrix, bool verbose )
{
   MtxReaderOptions options;
   options.verbose = verbose;
   readMtx( fileName, matrix, options );
}

template< typename Matrix >
void
MatrixReader< Matrix, TNL::Devices::Host >::readMtx( std::istream& file, Matrix& matrix, bool verbose )
{
   MtxReaderOptions options;
   options.verbose = verbose;
   readMtx( file, matrix, options );
}

template< typename Matrix >
void
MatrixReader< Matrix, TNL::Devices::Host >::readMtx(
   const std::string& fileName,
   Matrix& matrix,
   const MtxReaderOptions& options )
{
   std::ifstream file( fileName );
   if( ! file )
      throw std::runtime_error( std::string( "I am not able to open the file " ) + fileName );
   readMtx( file, matrix, options );
}

template< typename Matrix >
void
MatrixReader< Matrix, TNL::Devices::Host >::readMtx( std::istream& file, Matrix& matrix, const MtxReaderOptions& options )
{
   long lineNumber = 0;
   const MtxHeader header = readMtxHeader( file, lineNumber );

   if( Matrix::isSymmetric() && ! header.symmetric )
      throw std::runtime_error( "Matrix is not symmetric, but flag for symmetric matrix is given. Aborting." );

   if( options.verbose )
      std::cout << "Matrix dimensions are " << header.rows << " x " << header.columns << '\n';
   matrix.setDimensions( header.rows, header.columns );

   readMatrixElements( file, header, lineNumber, matrix, options );
}

template< typename Matrix >
bool
MatrixReader< Matrix, TNL::Devices::Host >::isSymmetric( const std::string& fileName )
{
   std::ifstream file( fileName );
   if( ! file )
      throw std::runtime_error( std::string( "I am not able to open the file " ) + fileName );
   return isSymmetric( file );
}

template< typename Matrix >
bool
MatrixReader< Matrix, TNL::Devices::Host >::isSymmetric( std::istream& file )
{
   long lineNumber = 0;
   return readMtxHeader( file, lineNumber ).symmetric;
}

template< typename Matrix >
auto
MatrixReader< Matrix, TNL::Devices::Host >::readMtxHeader( std::istream& file, long& lineNumber ) -> MtxHeader
{
   file.clear();
   file.seekg( 0, std::ios::beg );
   lineNumber = 0;
   MtxHeader header;

   // The banner, e.g. "%%MatrixMarket matrix coordinate real general". The keywords are case-insensitive.
   std::string line;
   std::getline( file, line );
   lineNumber++;
   std::vector< std::string > words = detail::splitMtxLine( line );
   for( auto& word : words )
      word = detail::mtxToLowerCase( word );
   if( words.size() < 5 || words[ 0 ] != "%%matrixmarket" )
      throw std::runtime_error(
         "Unknown format of the source file. We expect line like this: %%MatrixMarket matrix coordinate real general" );
   if( words[ 1 ] != "matrix" )
      throw std::runtime_error( "Keyword 'matrix' is expected in the header line: " + line );
   if( words[ 2 ] != "coordinates" && words[ 2 ] != "coordinate" )
      throw std::runtime_error( "Error: Only 'coordinates' format is supported now, not " + words[ 2 ] );
   // TODO: add support for complex matrices
   if( words[ 3 ] != "real" && words[ 3 ] != "integer" && words[ 3 ] != "pattern" )
      throw std::runtime_error( "Only 'real', 'integer' and 'pattern' matrices are supported, not " + words[ 3 ] );
   header.pattern = words[ 3 ] == "pattern";
   if( words[ 4 ] == "symmetric" )
      header.symmetric = true;
   else if( words[ 4 ] != "general" )
      throw std::runtime_error( "Only 'general' and 'symmetric' matrices are supported, not " + words[ 4 ] );

   // The size line follows after comments and blank lines.
   while( std::getline( file, line ) ) {
      lineNumber++;
      words = detail::splitMtxLine( line );
      if( words.empty() || words[ 0 ][ 0 ] == '%' )
         continue;
      if( words.size() != 3 )
         throw std::runtime_error( "Wrong number of parameters in the matrix header - should be 3." );
      header.rows = detail::parseMtxSize< IndexType >( words[ 0 ], lineNumber );
      header.columns = detail::parseMtxSize< IndexType >( words[ 1 ], lineNumber );
      header.elements = detail::parseMtxSize< IndexType >( words[ 2 ], lineNumber );
      if( header.rows == 0 || header.columns == 0 )
         detail::throwMtxError( lineNumber, "the numbers of rows and columns must be positive." );
      if( header.symmetric && header.rows != header.columns )
         detail::throwMtxError( lineNumber, "a symmetric matrix must be square." );
      return header;
   }
   throw std::runtime_error( "The size line of the matrix is missing in the MTX file." );
}

template< typename Matrix >
bool
MatrixReader< Matrix, TNL::Devices::Host >::readMatrixElement(
   std::istream& file,
   const MtxHeader& header,
   std::string& line,
   long& lineNumber,
   IndexType& row,
   IndexType& column,
   RealType& value )
{
   std::array< std::string_view, 3 > words;
   while( std::getline( file, line ) ) {
      lineNumber++;
      const std::size_t count = detail::splitMtxLine( line, words );
      if( count == 0 || words[ 0 ][ 0 ] == '%' )
         continue;
      if( count != 3 - static_cast< std::size_t >( header.pattern ) )
         detail::throwMtxError( lineNumber, "wrong number of parameters in the matrix element line: " + line );
      row = detail::parseMtxIndex( words[ 0 ], header.rows, lineNumber );
      column = detail::parseMtxIndex( words[ 1 ], header.columns, lineNumber );
      // If the MTX file stores only the matrix pattern, there is no value in the file.
      value = header.pattern ? RealType{ 1 } : static_cast< RealType >( detail::parseMtxReal( words[ 2 ], lineNumber ) );
      // The elements of a symmetric matrix are stored below the diagonal, so that
      // the elements (i,j) and (j,i) are recognized as the same element.
      if( header.symmetric && row < column )
         std::swap( row, column );
      return true;
   }
   return false;
}

template< typename Matrix >
void
MatrixReader< Matrix, TNL::Devices::Host >::throwDuplicateElementError( std::istream& file, IndexType row, IndexType column )
{
   // The lines of the elements are not stored to save memory, the file is read again to find them.
   long lineNumber = 0;
   const MtxHeader header = readMtxHeader( file, lineNumber );
   std::string line;
   std::string lines;
   IndexType elementRow = 0;
   IndexType elementColumn = 0;
   RealType value;
   while( readMatrixElement( file, header, line, lineNumber, elementRow, elementColumn, value ) )
      if( elementRow == row && elementColumn == column )
         lines += ( lines.empty() ? "" : ", " ) + std::to_string( lineNumber );
   throw std::runtime_error(
      "The matrix element at row " + std::to_string( row + 1 ) + " and column " + std::to_string( column + 1 )
      + " appears more than once in the MTX file, at lines " + lines + "." );
}

template< typename Matrix >
void
MatrixReader< Matrix, TNL::Devices::Host >::readMatrixElements(
   std::istream& file,
   const MtxHeader& header,
   long& lineNumber,
   Matrix& matrix,
   const MtxReaderOptions& options )
{
   Timer timer;
   timer.start();

   // The elements are read into a vector, which needs much less memory than std::map
   // and sorting it at once is much faster than inserting the elements into std::map.
   using Element = std::tuple< IndexType, IndexType, RealType >;
   std::vector< Element > elements;
   elements.reserve( header.elements );
   std::string line;
   IndexType row = 0;
   IndexType column = 0;
   RealType value;
   while( readMatrixElement( file, header, line, lineNumber, row, column, value ) ) {
      elements.emplace_back( row, column, value );
      if( options.verbose && elements.size() % 1000000 == 0 )
         std::cout << " Reading the matrix elements ... " << elements.size() / 1000000 << " millions      \r" << std::flush;
   }
   const std::size_t elementsInFile = elements.size();

   if( options.checkElementsCount && elementsInFile != static_cast< std::size_t >( header.elements ) )
      throw std::runtime_error(
         "The size line of the MTX file declares " + std::to_string( header.elements )
         + " matrix elements, but the file contains " + std::to_string( elementsInFile ) + " elements." );

   // The stable sort keeps the elements at the same position in the order of the file.
   std::stable_sort( elements.begin(), elements.end(), detail::MatrixElementsPositionLess{} );

   // Resolve the elements appearing more than once.
   std::size_t last = 0;
   for( std::size_t i = 1; i < elements.size(); i++ ) {
      Element& current = elements[ last ];
      const Element& next = elements[ i ];
      if( std::get< 0 >( current ) != std::get< 0 >( next ) || std::get< 1 >( current ) != std::get< 1 >( next ) ) {
         elements[ ++last ] = next;
         continue;
      }
      switch( options.duplicates ) {
         case MtxDuplicateElements::Error:
            throwDuplicateElementError( file, std::get< 0 >( next ), std::get< 1 >( next ) );
         case MtxDuplicateElements::KeepLast:
            std::get< 2 >( current ) = std::get< 2 >( next );
            break;
         case MtxDuplicateElements::Sum:
            if constexpr( std::is_same_v< RealType, bool > )
               std::get< 2 >( current ) = std::get< 2 >( current ) || std::get< 2 >( next );
            else
               std::get< 2 >( current ) += std::get< 2 >( next );
            break;
      }
   }
   if( ! elements.empty() )
      elements.resize( last + 1 );

   // A general matrix needs also the elements above the diagonal.
   if( header.symmetric && ! Matrix::isSymmetric() ) {
      const std::size_t lowerElements = elements.size();
      for( std::size_t i = 0; i < lowerElements; i++ ) {
         const auto [ elementRow, elementColumn, elementValue ] = elements[ i ];
         if( elementRow != elementColumn )
            elements.emplace_back( elementColumn, elementRow, elementValue );
      }
   }

   timer.stop();
   if( options.verbose )
      std::cout << " Reading the matrix elements ... " << elementsInFile << " -> " << timer.getRealTime() << " sec.\n";

   timer.reset();
   timer.start();
   if( options.verbose )
      std::cout << " Copying matrix elements to the matrix ... \r" << std::flush;
   const std::size_t elementsCount = elements.size();
   // The elements of a symmetric matrix are below the diagonal.
   matrix.setElements(
      std::move( elements ),
      Matrix::isSymmetric() ? MatrixElementsEncoding::SymmetricLower : MatrixElementsEncoding::Complete );
   timer.stop();
   const double dataSize = static_cast< double >( elementsCount ) * ( sizeof( RealType ) + sizeof( IndexType ) );
   if( options.verbose )
      std::cout << " Copying matrix elements to the matrix ... -> " << timer.getRealTime() << " sec. i.e. "
                << dataSize / ( timer.getRealTime() * ( 1 << 20 ) ) << "MB/s.\n";
}
/// \endcond

}  // namespace TNL::Matrices
