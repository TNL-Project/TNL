#include <iostream>
#include <TNL/Algorithms/Segments/AdaptiveCSR.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Arithmetics/Complex.h>

#include <gtest/gtest.h>

const char* saveAndLoadFileName = "test_SparseMatrixTest_SortedAdaptiveCSR_segments";

// types for which MatrixTest is instantiated
using MatrixTypes = ::testing::Types<
   TNL::Matrices::
      SparseMatrix< int, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::
      SparseMatrix< long, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::
      SparseMatrix< float, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::SparseMatrix< double,
                                TNL::Devices::Host,
                                int,
                                TNL::Matrices::GeneralMatrix,
                                TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::
      SparseMatrix< int, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::
      SparseMatrix< long, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::SparseMatrix< float,
                                TNL::Devices::Host,
                                long,
                                TNL::Matrices::GeneralMatrix,
                                TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::SparseMatrix< double,
                                TNL::Devices::Host,
                                long,
                                TNL::Matrices::GeneralMatrix,
                                TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::SparseMatrix< std::complex< float >,
                                TNL::Devices::Host,
                                long,
                                TNL::Matrices::GeneralMatrix,
                                TNL::Algorithms::Segments::SortedAdaptiveCSR >
#if defined( __CUDACC__ )
   ,
   TNL::Matrices::
      SparseMatrix< int, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::
      SparseMatrix< long, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::
      SparseMatrix< float, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::SparseMatrix< double,
                                TNL::Devices::Cuda,
                                int,
                                TNL::Matrices::GeneralMatrix,
                                TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::
      SparseMatrix< int, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::
      SparseMatrix< long, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::SparseMatrix< float,
                                TNL::Devices::Cuda,
                                long,
                                TNL::Matrices::GeneralMatrix,
                                TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::SparseMatrix< double,
                                TNL::Devices::Cuda,
                                long,
                                TNL::Matrices::GeneralMatrix,
                                TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::SparseMatrix< TNL::Arithmetics::Complex< float >,
                                TNL::Devices::Cuda,
                                long,
                                TNL::Matrices::GeneralMatrix,
                                TNL::Algorithms::Segments::SortedAdaptiveCSR >
#elif defined( __HIP__ )
   ,
   TNL::Matrices::
      SparseMatrix< int, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::
      SparseMatrix< long, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::
      SparseMatrix< float, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::
      SparseMatrix< double, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::
      SparseMatrix< int, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::
      SparseMatrix< long, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::
      SparseMatrix< float, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::SparseMatrix< double,
                                TNL::Devices::Hip,
                                long,
                                TNL::Matrices::GeneralMatrix,
                                TNL::Algorithms::Segments::SortedAdaptiveCSR >,
   TNL::Matrices::SparseMatrix< TNL::Arithmetics::Complex< float >,
                                TNL::Devices::Hip,
                                long,
                                TNL::Matrices::GeneralMatrix,
                                TNL::Algorithms::Segments::SortedAdaptiveCSR >
#endif
   >;

#include "SparseMatrixTest.h"
#include "../main.h"
