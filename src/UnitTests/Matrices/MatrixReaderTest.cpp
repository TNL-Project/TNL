#include "gtest/gtest.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Matrices/MatrixType.h>
#include <TNL/Matrices/MatrixWriter.h>
#include <TNL/Matrices/SparseMatrix.h>

using Matrix = TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int >;
using SymmetricMatrix = TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int, TNL::Matrices::SymmetricMatrix >;
using DenseMatrix = TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, int >;
using Reader = TNL::Matrices::MatrixReader< Matrix >;

static const std::string generalMtx =
   "%%MatrixMarket matrix coordinate real general\n"
   "3 3 4\n"
   "1 1 1.0\n"
   "2 1 2.0\n"
   "2 2 3.0\n"
   "3 3 4.0\n";

static const std::string symmetricMtx =
   "%%MatrixMarket matrix coordinate real symmetric\n"
   "% a comment line\n"
   "3 3 4\n"
   "1 1 1.0\n"
   "2 1 2.0\n"
   "2 2 3.0\n"
   "3 3 4.0\n";

static const std::string skewSymmetricMtx =
   "%%MatrixMarket matrix coordinate real skew-symmetric\n"
   "3 3 2\n"
   "2 1 2.0\n"
   "3 2 -3.0\n";

template< typename MatrixType = Matrix >
MatrixType
readFromString( const std::string& text )
{
   std::istringstream stream( text );
   MatrixType matrix;
   TNL::Matrices::MatrixReader< MatrixType >::readMtx( stream, matrix );
   return matrix;
}

template< typename MatrixType = Matrix >
MatrixType
readFromString( const std::string& text, const TNL::Matrices::MtxReaderOptions& options )
{
   std::istringstream stream( text );
   MatrixType matrix;
   TNL::Matrices::MatrixReader< MatrixType >::readMtx( stream, matrix, options );
   return matrix;
}

// Compare all elements of the matrix with the expected dense matrix given row by row
template< typename MatrixType >
void
expectElements( const MatrixType& matrix, const std::vector< std::vector< double > >& expected )
{
   ASSERT_EQ( matrix.getRows(), static_cast< int >( expected.size() ) );
   for( int row = 0; row < matrix.getRows(); row++ ) {
      ASSERT_EQ( matrix.getColumns(), static_cast< int >( expected[ row ].size() ) );
      for( int column = 0; column < matrix.getColumns(); column++ )
         EXPECT_EQ( matrix.getElement( row, column ), expected[ row ][ column ] ) << "row = " << row << ", column = " << column;
   }
}

// readMtx - valid files

TEST( MatrixReaderTest, readMtx_general )
{
   const Matrix matrix = readFromString( generalMtx );
   expectElements( matrix, { { 1, 0, 0 }, { 2, 3, 0 }, { 0, 0, 4 } } );
   EXPECT_EQ( matrix.getNonzeroElementsCount(), 4 );
}

TEST( MatrixReaderTest, readMtx_symmetricIntoGeneralMatrix )
{
   // the elements below the diagonal are mirrored above it
   const Matrix matrix = readFromString( symmetricMtx );
   expectElements( matrix, { { 1, 2, 0 }, { 2, 3, 0 }, { 0, 0, 4 } } );
   EXPECT_EQ( matrix.getNonzeroElementsCount(), 5 );
}

TEST( MatrixReaderTest, readMtx_symmetricIntoSymmetricMatrix )
{
   const SymmetricMatrix matrix = readFromString< SymmetricMatrix >( symmetricMtx );
   expectElements( matrix, { { 1, 2, 0 }, { 2, 3, 0 }, { 0, 0, 4 } } );
}

TEST( MatrixReaderTest, readMtx_generalIntoSymmetricMatrix )
{
   EXPECT_THROW( readFromString< SymmetricMatrix >( generalMtx ), std::runtime_error );
}

TEST( MatrixReaderTest, readMtx_denseMatrix )
{
   const DenseMatrix matrix = readFromString< DenseMatrix >( generalMtx );
   expectElements( matrix, { { 1, 0, 0 }, { 2, 3, 0 }, { 0, 0, 4 } } );
}

TEST( MatrixReaderTest, readMtx_skewSymmetric )
{
   // the elements below the diagonal are mirrored above it with the opposite sign
   const Matrix matrix = readFromString( skewSymmetricMtx );
   expectElements( matrix, { { 0, -2, 0 }, { 2, 0, 3 }, { 0, -3, 0 } } );
   EXPECT_EQ( matrix.getNonzeroElementsCount(), 4 );
}

TEST( MatrixReaderTest, readMtx_skewSymmetricIntoDenseMatrix )
{
   const DenseMatrix matrix = readFromString< DenseMatrix >( skewSymmetricMtx );
   expectElements( matrix, { { 0, -2, 0 }, { 2, 0, 3 }, { 0, -3, 0 } } );
}

TEST( MatrixReaderTest, readMtx_skewSymmetricIntoSymmetricMatrix )
{
   EXPECT_THROW( readFromString< SymmetricMatrix >( skewSymmetricMtx ), std::runtime_error );
}

TEST( MatrixReaderTest, readMtx_skewSymmetricUpperTriangle )
{
   // the element (1,2) is the element (2,1) with the opposite sign
   const Matrix matrix = readFromString(
      "%%MatrixMarket matrix coordinate real skew-symmetric\n"
      "2 2 1\n"
      "1 2 2.0\n" );
   expectElements( matrix, { { 0, 2 }, { -2, 0 } } );
}

TEST( MatrixReaderTest, readMtx_skewSymmetricBothTriangles )
{
   // the elements (i,j) and (j,i) of a skew-symmetric matrix are the same element
   const std::string mtx =
      "%%MatrixMarket matrix coordinate real skew-symmetric\n"
      "2 2 2\n"
      "2 1 2.0\n"
      "1 2 3.0\n";
   EXPECT_THROW( readFromString( mtx ), std::runtime_error );

   TNL::Matrices::MtxReaderOptions options;
   options.duplicates = TNL::Matrices::MtxDuplicateElements::Sum;
   expectElements( readFromString( mtx, options ), { { 0, 1 }, { -1, 0 } } );
}

TEST( MatrixReaderTest, readMtx_skewSymmetricInvalid )
{
   // element on the diagonal
   EXPECT_THROW(
      readFromString(
         "%%MatrixMarket matrix coordinate real skew-symmetric\n"
         "2 2 1\n"
         "1 1 1.0\n" ),
      std::runtime_error );
   // non-square matrix
   EXPECT_THROW(
      readFromString(
         "%%MatrixMarket matrix coordinate real skew-symmetric\n"
         "3 2 1\n"
         "2 1 1.0\n" ),
      std::runtime_error );
   // pattern matrix
   EXPECT_THROW(
      readFromString(
         "%%MatrixMarket matrix coordinate pattern skew-symmetric\n"
         "2 2 1\n"
         "2 1\n" ),
      std::runtime_error );
}

TEST( MatrixReaderTest, readMtx_pattern )
{
   const Matrix matrix = readFromString(
      "%%MatrixMarket matrix coordinate pattern general\n"
      "2 3 3\n"
      "1 1\n"
      "1 3\n"
      "2 2\n" );
   expectElements( matrix, { { 1, 0, 1 }, { 0, 1, 0 } } );
}

TEST( MatrixReaderTest, readMtx_patternSymmetric )
{
   const Matrix matrix = readFromString(
      "%%MatrixMarket matrix coordinate pattern symmetric\n"
      "2 2 2\n"
      "1 1\n"
      "2 1\n" );
   expectElements( matrix, { { 1, 1 }, { 1, 0 } } );
}

TEST( MatrixReaderTest, readMtx_integer )
{
   const Matrix matrix = readFromString(
      "%%MatrixMarket matrix coordinate integer general\n"
      "2 2 2\n"
      "1 2 -7\n"
      "2 1 42\n" );
   expectElements( matrix, { { 0, -7 }, { 42, 0 } } );
}

TEST( MatrixReaderTest, readMtx_valueFormats )
{
   const Matrix matrix = readFromString(
      "%%MatrixMarket matrix coordinate real general\n"
      "1 5 5\n"
      "1 1 -2.5\n"
      "1 2 1e3\n"
      "1 3 1.5E-2\n"
      "1 4 +3\n"
      "1 5 .25\n" );
   expectElements( matrix, { { -2.5, 1000, 0.015, 3, 0.25 } } );
}

TEST( MatrixReaderTest, readMtx_rectangular )
{
   // there are an empty row and an empty column
   const Matrix matrix = readFromString(
      "%%MatrixMarket matrix coordinate real general\n"
      "3 4 2\n"
      "1 4 1.0\n"
      "3 1 2.0\n" );
   expectElements( matrix, { { 0, 0, 0, 1 }, { 0, 0, 0, 0 }, { 2, 0, 0, 0 } } );
}

TEST( MatrixReaderTest, readMtx_noElements )
{
   const Matrix matrix = readFromString(
      "%%MatrixMarket matrix coordinate real general\n"
      "2 2 0\n" );
   expectElements( matrix, { { 0, 0 }, { 0, 0 } } );
   EXPECT_EQ( matrix.getNonzeroElementsCount(), 0 );
}

TEST( MatrixReaderTest, readMtx_unsortedElements )
{
   const Matrix matrix = readFromString(
      "%%MatrixMarket matrix coordinate real general\n"
      "2 2 3\n"
      "2 2 4.0\n"
      "1 2 2.0\n"
      "2 1 3.0\n" );
   expectElements( matrix, { { 0, 2 }, { 3, 4 } } );
}

TEST( MatrixReaderTest, readMtx_symmetricUpperTriangle )
{
   // a symmetric file should store the lower triangle, but the elements above the diagonal are accepted too
   const Matrix matrix = readFromString(
      "%%MatrixMarket matrix coordinate real symmetric\n"
      "2 2 2\n"
      "1 2 2.0\n"
      "2 2 3.0\n" );
   expectElements( matrix, { { 0, 2 }, { 2, 3 } } );
}

TEST( MatrixReaderTest, readMtx_commentsAndBlankLines )
{
   const Matrix matrix = readFromString(
      "%%MatrixMarket matrix coordinate real general\n"
      "% comment\n"
      "%\n"
      "\n"
      "2 2 2\n"
      "% comment between the elements\n"
      "1 1 1.0\n"
      "\n"
      "2 2 2.0\n"
      "\n" );
   expectElements( matrix, { { 1, 0 }, { 0, 2 } } );
}

TEST( MatrixReaderTest, readMtx_extraSpaces )
{
   const Matrix matrix = readFromString(
      "%%MatrixMarket  matrix coordinate   real general  \n"
      "  2   2  2  \n"
      "   1  1   1.0   \n"
      "2 2 2.0 \n" );
   expectElements( matrix, { { 1, 0 }, { 0, 2 } } );
}

TEST( MatrixReaderTest, readMtx_tabs )
{
   const Matrix matrix = readFromString(
      "%%MatrixMarket matrix coordinate real general\n"
      "2\t2\t2\n"
      "1\t1\t1.0\n"
      "2 \t 2\t2.0\n" );
   expectElements( matrix, { { 1, 0 }, { 0, 2 } } );
}

TEST( MatrixReaderTest, readMtx_windowsLineEndings )
{
   const Matrix matrix = readFromString(
      "%%MatrixMarket matrix coordinate real general\r\n"
      "% comment\r\n"
      "2 2 2\r\n"
      "1 1 1.0\r\n"
      "2 2 2.0\r\n" );
   expectElements( matrix, { { 1, 0 }, { 0, 2 } } );
}

TEST( MatrixReaderTest, readMtx_caseInsensitiveHeader )
{
   // the keywords of the header are case-insensitive by the Matrix Market specification
   const Matrix matrix = readFromString(
      "%%MatrixMarket MATRIX Coordinate Real General\n"
      "2 2 1\n"
      "2 1 1.0\n" );
   expectElements( matrix, { { 0, 0 }, { 1, 0 } } );
}

TEST( MatrixReaderTest, readMtx_noFinalNewline )
{
   const Matrix matrix = readFromString(
      "%%MatrixMarket matrix coordinate real general\n"
      "2 2 1\n"
      "2 1 1.0" );
   expectElements( matrix, { { 0, 0 }, { 1, 0 } } );
}

TEST( MatrixReaderTest, readMtx_writerRoundTrip )
{
   Matrix matrix( 4, 5 );
   matrix.setElements( { { 0, 0, 1.5 }, { 0, 4, -2.0 }, { 1, 2, 3.25 }, { 3, 1, 1e-10 }, { 3, 3, 4e10 } } );
   std::stringstream stream;
   TNL::Matrices::MatrixWriter< Matrix >::writeMtx( stream, matrix );
   Matrix result;
   Reader::readMtx( stream, result );
   EXPECT_EQ( result, matrix );
}

TEST( MatrixReaderTest, readMtx_file )
{
   const std::filesystem::path fileName = std::filesystem::temp_directory_path() / "tnl-matrix-reader-test-read.mtx";
   {
      std::ofstream file( fileName );
      file << generalMtx;
   }
   Matrix matrix;
   Reader::readMtx( fileName.string(), matrix );
   std::filesystem::remove( fileName );
   expectElements( matrix, { { 1, 0, 0 }, { 2, 3, 0 }, { 0, 0, 4 } } );

   EXPECT_THROW( Reader::readMtx( fileName.string(), matrix ), std::runtime_error );
}

// readMtx - invalid files

TEST( MatrixReaderTest, readMtx_emptyFile )
{
   EXPECT_THROW( readFromString( "" ), std::runtime_error );
}

TEST( MatrixReaderTest, readMtx_missingBanner )
{
   EXPECT_THROW(
      readFromString(
         "2 2 1\n"
         "1 1 1.0\n" ),
      std::runtime_error );
}

TEST( MatrixReaderTest, readMtx_unsupportedFormats )
{
   // dense (array) format
   EXPECT_THROW(
      readFromString(
         "%%MatrixMarket matrix array real general\n"
         "2 2\n"
         "1.0\n2.0\n3.0\n4.0\n" ),
      std::runtime_error );
   // complex values
   EXPECT_THROW(
      readFromString(
         "%%MatrixMarket matrix coordinate complex general\n"
         "1 1 1\n"
         "1 1 1.0 2.0\n" ),
      std::runtime_error );
   // hermitian matrices
   EXPECT_THROW(
      readFromString(
         "%%MatrixMarket matrix coordinate real hermitian\n"
         "2 2 1\n"
         "2 1 1.0\n" ),
      std::runtime_error );
}

TEST( MatrixReaderTest, readMtx_missingSizeLine )
{
   EXPECT_THROW( readFromString( "%%MatrixMarket matrix coordinate real general\n" ), std::runtime_error );
   EXPECT_THROW(
      readFromString(
         "%%MatrixMarket matrix coordinate real general\n"
         "% only a comment\n" ),
      std::runtime_error );
}

TEST( MatrixReaderTest, readMtx_wrongSizeLine )
{
   EXPECT_THROW(
      readFromString(
         "%%MatrixMarket matrix coordinate real general\n"
         "2 2\n" ),
      std::runtime_error );
   EXPECT_THROW(
      readFromString(
         "%%MatrixMarket matrix coordinate real general\n"
         "0 2 0\n" ),
      std::runtime_error );
   EXPECT_THROW(
      readFromString(
         "%%MatrixMarket matrix coordinate real general\n"
         "2 -1 0\n" ),
      std::runtime_error );
}

TEST( MatrixReaderTest, readMtx_symmetricNonSquare )
{
   EXPECT_THROW(
      readFromString(
         "%%MatrixMarket matrix coordinate real symmetric\n"
         "3 2 1\n"
         "2 1 1.0\n" ),
      std::runtime_error );
}

TEST( MatrixReaderTest, readMtx_wrongElementLine )
{
   // missing value
   EXPECT_THROW(
      readFromString(
         "%%MatrixMarket matrix coordinate real general\n"
         "2 2 1\n"
         "1 1\n" ),
      std::runtime_error );
   // value in a pattern file
   EXPECT_THROW(
      readFromString(
         "%%MatrixMarket matrix coordinate pattern general\n"
         "2 2 1\n"
         "1 1 1.0\n" ),
      std::runtime_error );
   // not a number
   EXPECT_THROW(
      readFromString(
         "%%MatrixMarket matrix coordinate real general\n"
         "2 2 1\n"
         "1 x 1.0\n" ),
      std::runtime_error );
}

TEST( MatrixReaderTest, readMtx_elementOutOfRange )
{
   EXPECT_THROW(
      readFromString(
         "%%MatrixMarket matrix coordinate real general\n"
         "2 2 1\n"
         "3 1 1.0\n" ),
      std::runtime_error );
   EXPECT_THROW(
      readFromString(
         "%%MatrixMarket matrix coordinate real general\n"
         "2 2 1\n"
         "1 3 1.0\n" ),
      std::runtime_error );
   // the indexes are 1-based
   EXPECT_THROW(
      readFromString(
         "%%MatrixMarket matrix coordinate real general\n"
         "2 2 1\n"
         "0 1 1.0\n" ),
      std::runtime_error );
   EXPECT_THROW(
      readFromString(
         "%%MatrixMarket matrix coordinate real general\n"
         "2 2 1\n"
         "1 -1 1.0\n" ),
      std::runtime_error );
}

TEST( MatrixReaderTest, readMtx_wrongNumberOfElements )
{
   // fewer elements than declared, e.g. a truncated file
   EXPECT_THROW(
      readFromString(
         "%%MatrixMarket matrix coordinate real general\n"
         "2 2 3\n"
         "1 1 1.0\n"
         "2 2 2.0\n" ),
      std::runtime_error );
   // more elements than declared
   EXPECT_THROW(
      readFromString(
         "%%MatrixMarket matrix coordinate real general\n"
         "2 2 1\n"
         "1 1 1.0\n"
         "2 2 2.0\n" ),
      std::runtime_error );
}

TEST( MatrixReaderTest, readMtx_errorLineNumber )
{
   try {
      readFromString(
         "%%MatrixMarket matrix coordinate real general\n"
         "% comment\n"
         "2 2 2\n"
         "1 1 1.0\n"
         "3 1 2.0\n" );
      FAIL() << "an exception was expected";
   }
   catch( const std::runtime_error& e ) {
      EXPECT_NE( std::string( e.what() ).find( "line 5" ), std::string::npos ) << e.what();
   }
}

// readMtx - options

static const std::string duplicatesMtx =
   "%%MatrixMarket matrix coordinate real general\n"
   "2 2 3\n"
   "1 2 1.0\n"
   "2 2 5.0\n"
   "1 2 2.0\n";

TEST( MatrixReaderTest, readMtx_duplicatesError )
{
   try {
      readFromString( duplicatesMtx );
      FAIL() << "an exception was expected";
   }
   catch( const std::runtime_error& e ) {
      // the message gives the lines of the element
      EXPECT_NE( std::string( e.what() ).find( "lines 3, 5" ), std::string::npos ) << e.what();
   }
}

TEST( MatrixReaderTest, readMtx_duplicatesKeepLast )
{
   TNL::Matrices::MtxReaderOptions options;
   options.duplicates = TNL::Matrices::MtxDuplicateElements::KeepLast;
   expectElements( readFromString( duplicatesMtx, options ), { { 0, 2 }, { 0, 5 } } );
}

TEST( MatrixReaderTest, readMtx_duplicatesSum )
{
   TNL::Matrices::MtxReaderOptions options;
   options.duplicates = TNL::Matrices::MtxDuplicateElements::Sum;
   expectElements( readFromString( duplicatesMtx, options ), { { 0, 3 }, { 0, 5 } } );
}

TEST( MatrixReaderTest, readMtx_symmetricBothTriangles )
{
   // the elements (i,j) and (j,i) of a symmetric matrix are the same element
   const std::string mtx =
      "%%MatrixMarket matrix coordinate real symmetric\n"
      "2 2 2\n"
      "2 1 2.0\n"
      "1 2 3.0\n";
   EXPECT_THROW( readFromString( mtx ), std::runtime_error );

   TNL::Matrices::MtxReaderOptions options;
   options.duplicates = TNL::Matrices::MtxDuplicateElements::Sum;
   expectElements( readFromString( mtx, options ), { { 0, 5 }, { 5, 0 } } );
   expectElements( readFromString< SymmetricMatrix >( mtx, options ), { { 0, 5 }, { 5, 0 } } );
}

TEST( MatrixReaderTest, readMtx_uncheckedNumberOfElements )
{
   TNL::Matrices::MtxReaderOptions options;
   options.checkElementsCount = false;
   const Matrix fewer = readFromString(
      "%%MatrixMarket matrix coordinate real general\n"
      "2 2 3\n"
      "1 1 1.0\n"
      "2 2 2.0\n",
      options );
   expectElements( fewer, { { 1, 0 }, { 0, 2 } } );
   const Matrix more = readFromString(
      "%%MatrixMarket matrix coordinate real general\n"
      "2 2 1\n"
      "1 1 1.0\n"
      "2 2 2.0\n",
      options );
   expectElements( more, { { 1, 0 }, { 0, 2 } } );
}

TEST( MatrixReaderTest, readMtx_fileWithOptions )
{
   const std::filesystem::path fileName = std::filesystem::temp_directory_path() / "tnl-matrix-reader-test-options.mtx";
   {
      std::ofstream file( fileName );
      file << duplicatesMtx;
   }
   TNL::Matrices::MtxReaderOptions options;
   options.duplicates = TNL::Matrices::MtxDuplicateElements::Sum;
   Matrix matrix;
   Reader::readMtx( fileName.string(), matrix, options );
   std::filesystem::remove( fileName );
   expectElements( matrix, { { 0, 3 }, { 0, 5 } } );
}

// isSymmetric

TEST( MatrixReaderTest, isSymmetric_general )
{
   std::istringstream stream( generalMtx );
   EXPECT_FALSE( Reader::isSymmetric( stream ) );
}

TEST( MatrixReaderTest, isSymmetric_symmetric )
{
   std::istringstream stream( symmetricMtx );
   EXPECT_TRUE( Reader::isSymmetric( stream ) );
}

TEST( MatrixReaderTest, isSymmetric_pattern )
{
   std::istringstream stream(
      "%%MatrixMarket matrix coordinate pattern symmetric\n"
      "2 2 2\n"
      "1 1\n"
      "2 1\n" );
   EXPECT_TRUE( Reader::isSymmetric( stream ) );
}

TEST( MatrixReaderTest, isSymmetric_skewSymmetric )
{
   std::istringstream stream( skewSymmetricMtx );
   EXPECT_FALSE( Reader::isSymmetric( stream ) );
}

TEST( MatrixReaderTest, isSymmetric_unsupportedSymmetry )
{
   std::istringstream stream(
      "%%MatrixMarket matrix coordinate real hermitian\n"
      "2 2 1\n"
      "2 1 1.0\n" );
   EXPECT_THROW( (void) Reader::isSymmetric( stream ), std::runtime_error );
}

TEST( MatrixReaderTest, isSymmetric_thenReadMtx )
{
   // isSymmetric reads only the header, the matrix can be read from the same stream afterwards
   std::istringstream stream( symmetricMtx );
   ASSERT_TRUE( Reader::isSymmetric( stream ) );
   Matrix matrix;
   Reader::readMtx( stream, matrix );
   expectElements( matrix, { { 1, 2, 0 }, { 2, 3, 0 }, { 0, 0, 4 } } );
}

TEST( MatrixReaderTest, isSymmetric_file )
{
   const std::filesystem::path fileName = std::filesystem::temp_directory_path() / "tnl-matrix-reader-test.mtx";
   {
      std::ofstream file( fileName );
      file << symmetricMtx;
   }
   EXPECT_TRUE( Reader::isSymmetric( fileName.string() ) );
   std::filesystem::remove( fileName );

   EXPECT_THROW( (void) Reader::isSymmetric( fileName.string() ), std::runtime_error );
}

#include "../main.h"
