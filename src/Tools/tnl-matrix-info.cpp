#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Matrices/MatrixInfo.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/MatrixType.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/sort.h>

using namespace TNL;
using namespace TNL::Matrices;

using Real = double;
using Index = long;
using CSRHostMatrix = Matrices::SparseMatrix< Real, Devices::Host, Index, Matrices::GeneralMatrix, Algorithms::Segments::CSR >;

bool
printInfo( const std::string& fileName, bool verbose = true )
{
   CSRHostMatrix matrix;
   MatrixReader< CSRHostMatrix >::readMtx( fileName, matrix, verbose );

   // Nonzero elements per row statistics
   const int nonzeros = matrix.getNonzeroElementsCount();
   TNL::Containers::Vector< Index, Devices::Host, Index > nonzerosPerRow( matrix.getRows() );
   TNL::Containers::Vector< double, Devices::Host, Index > aux;
   matrix.getCompressedRowLengths( nonzerosPerRow );
   double average = sum( nonzerosPerRow ) / nonzerosPerRow.getSize();
   aux = nonzerosPerRow - average;
   double std_dev = lpNorm( aux, 2.0 ) / nonzerosPerRow.getSize();
   TNL::Algorithms::ascendingSort( nonzerosPerRow );
   double percentile_25 = nonzerosPerRow[ nonzerosPerRow.getSize() * 0.25 ];
   double percentile_50 = nonzerosPerRow[ nonzerosPerRow.getSize() * 0.5 ];
   double percentile_75 = nonzerosPerRow[ nonzerosPerRow.getSize() * 0.75 ];

   // Print the info
   std::cout << fileName << ":\n"
             << "\tNumber of rows:\t" << matrix.getRows() << "\n"
             << "\tNumber of columns:\t" << matrix.getColumns() << "\n"
             << "\tNumber of non-zero elements:\t" << nonzeros << "\n"
             << "\tAverage number of non-zero elements per row:\t" << average << "\n"
             << "\tStandard deviation of non-zero elements per row:\t" << std_dev << "\n"
             << "\tPercentile 25 of non-zero elements per row:\t" << percentile_25 << "\n"
             << "\tPercentile 50 of non-zero elements per row:\t" << percentile_50 << "\n"
             << "\tPercentile 75 of non-zero elements per row:\t" << percentile_75 << "\n"
             << std::endl;

   return true;
}

int
main( int argc, char* argv[] )
{
   if( argc < 2 ) {
      std::cerr << "Usage: " << argv[ 0 ] << " filename.mtx ..." << std::endl;
      return EXIT_FAILURE;
   }

   bool result = true;

   for( int i = 1; i < argc; i++ ) {
      const std::string fileName = argv[ i ];
      result &= printInfo( fileName );
   }

   return static_cast< int >( ! result );
}
