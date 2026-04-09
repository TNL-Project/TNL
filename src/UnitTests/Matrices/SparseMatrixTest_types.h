#pragma once

#include <complex>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Arithmetics/Complex.h>
#include <TNL/Devices/Host.h>

#if defined( __CUDACC__ )
   #include <TNL/Devices/Cuda.h>
#elif defined( __HIP__ )
   #include <TNL/Devices/Hip.h>
#endif

#include <gtest/gtest.h>

template< template< typename Device, typename Index, typename IndexAllocator > class Segments >
using MatrixTypesTemplate = ::testing::Types<
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::SparseMatrix< std::complex< float >, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, Segments >
#if defined( __CUDACC__ )
   ,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::
      SparseMatrix< TNL::Arithmetics::Complex< float >, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, Segments >
#elif defined( __HIP__ )
   ,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, Segments >,
   TNL::Matrices::
      SparseMatrix< TNL::Arithmetics::Complex< float >, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, Segments >
#endif
   >;

template< template< typename Device, typename Index, typename IndexAllocator > class RowMajorSegments,
          template< typename Device, typename Index, typename IndexAllocator > class ColMajorSegments >
using MatrixTypesTemplateMixed = ::testing::Types<
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, ColMajorSegments >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, RowMajorSegments >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, RowMajorSegments >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, RowMajorSegments >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, RowMajorSegments >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, RowMajorSegments >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, RowMajorSegments >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, RowMajorSegments >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, RowMajorSegments >,
   TNL::Matrices::SparseMatrix< std::complex< float >, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, RowMajorSegments >
#if defined( __CUDACC__ )
   ,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, RowMajorSegments >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, ColMajorSegments >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, ColMajorSegments >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, ColMajorSegments >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, ColMajorSegments >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, ColMajorSegments >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, ColMajorSegments >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, ColMajorSegments >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, ColMajorSegments >,
   TNL::Matrices::
      SparseMatrix< TNL::Arithmetics::Complex< float >, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, ColMajorSegments >
#elif defined( __HIP__ )
   ,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, RowMajorSegments >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, ColMajorSegments >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, ColMajorSegments >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, ColMajorSegments >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, ColMajorSegments >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, ColMajorSegments >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, ColMajorSegments >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, ColMajorSegments >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, ColMajorSegments >,
   TNL::Matrices::
      SparseMatrix< TNL::Arithmetics::Complex< float >, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, ColMajorSegments >
#endif
   >;
