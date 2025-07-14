#include <iostream>
#include <functional>
#include <memory>

#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Solvers/Linear/GEM.h>

#include <type_traits>

#include <gtest/gtest.h>

using Dense_host_float = TNL::Matrices::DenseMatrix< float, TNL::Devices::Host, int >;
using Dense_host_int = TNL::Matrices::DenseMatrix< int, TNL::Devices::Host, int >;

using Dense_cuda_float = TNL::Matrices::DenseMatrix< float, TNL::Devices::Cuda, int >;
using Dense_cuda_int = TNL::Matrices::DenseMatrix< int, TNL::Devices::Cuda, int >;

// test fixture for typed tests
template< typename Matrix >
class GEMSolverTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
};

// types for which MatrixTest is instantiated
using MatrixTypes = ::testing::Types<
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, int, TNL::Algorithms::Segments::RowMajorOrder >,
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, int, TNL::Algorithms::Segments::ColumnMajorOrder >
#if defined( __CUDACC__ )
   ,
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::RowMajorOrder >,
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, long, TNL::Algorithms::Segments::ColumnMajorOrder >,
#elif defined( __HIP__ )
   ,
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Hip, int, TNL::Algorithms::Segments::RowMajorOrder >,
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Hip, int, TNL::Algorithms::Segments::ColumnMajorOrder >
#endif
   >;

template< typename Matrix >
void
test_diagonalMatrix()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   /*
    * Sets up the following 5x5 dense matrix:
    *
    *    / 1 . . . . \
    *    | . 2 . . . |
    *    | . . 3 . . |
    *    | . . . 4 . |
    *    \ . . . . 5 /
    */
   const IndexType size = 4;

   auto matrix = std::make_shared< Matrix >( size, size );
   matrix->getValues() = 0;

   IndexType value = 1;
   for( IndexType i = 0; i < size; i++ )
      matrix->setElement( i, i, value++ );

   VectorType x( size, 1.0 ), b( size, 0.0 ), y( size, 0.0 );
   matrix->vectorProduct( x, b );

   TNL::Solvers::Linear::GEM< Matrix > gem;
   gem.setMatrix( matrix );
   gem.setPivoting( true );
   gem.solve( b, y );

   EXPECT_NEAR( maxNorm( x - y ), 0.0, 1.0e-6 );
}

template< typename Matrix >
void
test_upperTriangularMatrix()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   /*
    * Sets up the upper triangular matrix of the following form:
    *
    *    / 1 2 3 4 5 \
    *    | . 1 2 3 4 |
    *    | . . 1 2 3 |
    *    | . . . 1 2 |
    *    \ . . . . 1 /
    */
   const IndexType size = 5;

   auto matrix = std::make_shared< Matrix >( size, size );
   matrix->getValues() = 0;

   for( IndexType i = 0; i < size; i++ )
      for( IndexType j = 0; j < size; j++ )
         if( j >= i )
            matrix->setElement( i, j, j - i + 1 );

   VectorType x( size, 1.0 ), b( size, 0.0 ), y( size, 0.0 );
   matrix->vectorProduct( x, b );

   TNL::Solvers::Linear::GEM< Matrix > gem;
   gem.setMatrix( matrix );
   gem.setPivoting( true );
   gem.solve( b, y );

   EXPECT_NEAR( maxNorm( x - y ), 0.0, 1.0e-6 );
}

template< typename Matrix >
void
test_smallMatrix()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   const IndexType size = 4;
   // clang-format off
   auto matrix = std::make_shared< Matrix >();
   matrix->setElements( { { 2, 1,  0,  3 },
                          { 4, 1, -1,  2 },
                          { 0, 1,  3, -1 },
                          { 1, 0,  2,  1 } } );
   // clang-format on

   VectorType x( size, 1.0 ), b( size, 0.0 ), y( size, 0.0 );
   matrix->vectorProduct( x, b );

   TNL::Solvers::Linear::GEM< Matrix > gem;
   gem.setMatrix( matrix );
   gem.setPivoting( true );
   gem.solve( b, y );

   EXPECT_NEAR( maxNorm( x - y ), 0.0, 1.0e-6 );
}

template< typename Matrix >
void
test_largeMatrix()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   const IndexType size = 16;
   auto matrix = std::make_shared< Matrix >();
   // clang-format off
   matrix->setElements({
      {  4, -1,  2,  0,  3,  1, -2,  0,  1,  0, -1,  1,  2,  1,  0, -2 },
      {  1,  5,  0,  2, -1,  0,  1,  3,  0, -1,  1,  0,  1, -2,  3,  1 },
      { -2,  1,  6, -1,  0,  2,  1,  0,  1,  2,  0,  1, -1,  0,  1,  2 },
      {  0, -1,  3,  7,  1,  0, -1,  2,  0,  3,  1, -2,  0,  1,  0,  1 },
      {  2,  0,  0, -2,  8,  2,  1, -1,  1,  0,  1,  2, -1,  0,  2, -1 },
      { -1,  1,  1,  0,  1,  9,  0,  3,  1,  2, -2,  0,  0,  1,  3,  1 },
      {  0, -2,  0,  1,  2,  1, 10,  1, -1,  0,  1,  1,  2, -1,  0,  0 },
      {  1,  3, -1,  0,  0,  1,  1, 11,  1,  0,  0,  2,  0,  3,  1, -2 },
      {  0,  1,  2, -1,  1, -1,  2,  0, 12,  1,  3,  0, -1,  1,  2,  0 },
      { -1,  0,  0,  3,  2,  1, -1,  0,  1, 13,  1, -1,  0,  2, -2,  1 },
      {  1, -1,  2,  1, -2,  0,  0,  1,  2,  1, 14,  2,  0,  1,  3, -1 },
      {  0,  1, -1,  0,  1,  2,  1,  3,  0, -2,  1, 15, -1,  2,  1,  2 },
      {  2,  0,  0,  1, -1,  1,  2,  0,  1,  0, -2,  2, 16,  0,  0, -1 },
      {  0,  2,  1,  0,  1,  0, -1,  1,  0,  1,  3, -1,  0, 17,  1,  0 },
      { -1,  0,  1,  2,  3,  1,  0,  2,  2, -1,  1,  0, -1,  2, 18,  1 },
      {  1,  1,  0,  1,  0,  2,  1, -2,  0,  2,  0,  1,  3,  0,  1, 19 }
   });
   // clang-format on
   VectorType x( size, 1.0 ), b( size, 0.0 ), y( size, 0.0 );
   matrix->vectorProduct( x, b );

   TNL::Solvers::Linear::GEM< Matrix > gem;
   gem.setMatrix( matrix );
   gem.setPivoting( true );
   gem.solve( b, y );

   EXPECT_NEAR( maxNorm( x - y ), 0.0, 1.0e-6 );
}

template< typename Matrix >
void
test_Cage3Matrix()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   const IndexType size = 5;
   auto matrix = std::make_shared< Matrix >();
   // clang-format off
   matrix->setElements( {
      { 0.666667, 0.366556, 0.300111, 0.366556, 0.300111 },
      { 0.100037, 0.533407, 0.000000, 0.200074, 0.000000 },
      { 0.122185, 0.000000, 0.577704, 0.000000, 0.244371 },
      { 0.050018, 0.100037, 0.000000, 0.283315, 0.183278 },
      { 0.061093, 0.000000, 0.122185, 0.150055, 0.272241 } } );
   // clang-format on
   VectorType x( size, 1.0 ), b( size, 0.0 ), y( size, 0.0 );
   matrix->vectorProduct( x, b );

   TNL::Solvers::Linear::GEM< Matrix > gem;
   gem.setMatrix( matrix );
   gem.setPivoting( true );
   gem.solve( b, y );

   EXPECT_NEAR( maxNorm( x - y ), 0.0, 1.0e-6 );
}

template< typename Matrix >
void
test_Cage4Matrix()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   const IndexType size = 9;
   auto matrix = std::make_shared< Matrix >();
   // clang-format off
   matrix->setElements( {
      { 0.750000, 0.137458, 0.000000, 0.112542, 0.137458, 0.000000, 0.000000, 0.112542, 0.000000 },
      { 0.075028, 0.687569, 0.112542, 0.000000, 0.075028, 0.112542, 0.000000, 0.000000, 0.000000 },
      { 0.000000, 0.091639, 0.666667, 0.075028, 0.000000, 0.091639, 0.075028, 0.000000, 0.000000 },
      { 0.091639, 0.000000, 0.137458, 0.729097, 0.000000, 0.000000, 0.137458, 0.091639, 0.000000 },
      { 0.037514, 0.037514, 0.000000, 0.000000, 0.537514, 0.137458, 0.112542, 0.000000, 0.250000 },
      { 0.000000, 0.045819, 0.045819, 0.000000, 0.075028, 0.445875, 0.000000, 0.075028, 0.150055 },
      { 0.000000, 0.000000, 0.037514, 0.037514, 0.091639, 0.000000, 0.470792, 0.091639, 0.183278 },
      { 0.045819, 0.000000, 0.000000, 0.045819, 0.000000, 0.137458, 0.112542, 0.545819, 0.250000 },
      { 0.000000, 0.000000, 0.000000, 0.000000, 0.083333, 0.075028, 0.091639, 0.083333, 0.166667 } } );
   // clang-format on
   VectorType x( size, 1.0 ), b( size, 0.0 ), y( size, 0.0 );
   matrix->vectorProduct( x, b );

   TNL::Solvers::Linear::GEM< Matrix > gem;
   gem.setMatrix( matrix );
   gem.setPivoting( true );
   gem.solve( b, y );

   EXPECT_NEAR( maxNorm( x - y ), 0.0, 1.0e-6 );
}

template< typename Matrix >
void
test_Ex5Matrix()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   const IndexType size = 27;
   auto matrix = std::make_shared< Matrix >();
   // clang-format off
   matrix->setElements( {
      { 1037038.992593, 259259.059259, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -1185186.251852,
-296296.651852, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 148148.148148, 37037.148148, 0.000000,
0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 }, { 259259.059259, 296297.540741, 259259.059259, -18518.551852,
0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -296296.651852, -148148.548148, -296296.651852, -74073.962963, 0.000000,
0.000000, 0.000000, 0.000000, 0.000000, 37037.148148, -148148.214815, 37037.148148, 92592.570370, 0.000000, 0.000000, 0.000000,
0.000000, 0.000000 }, { 0.000000, 259259.059259, 1037038.992593, 259259.059259, 0.000000, 0.000000, 0.000000, 0.000000,
0.000000, 0.000000, -296296.651852, -1185186.251852, -296296.651852, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
37037.148148, 148148.148148, 37037.148148, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 }, { 0.000000, -18518.551852,
259259.059259, 296297.540741, 259259.059259, -18518.551852, 0.000000, 0.000000, 0.000000, 0.000000, -74073.962963,
-296296.651852, -148148.548148, -296296.651852, -74073.962963, 0.000000, 0.000000, 0.000000, 0.000000, 92592.570370,
37037.148148, -148148.214815, 37037.148148, 92592.570370, 0.000000, 0.000000, 0.000000 }, { 0.000000, 0.000000, 0.000000,
259259.059259, 1037038.992593, 259259.059259, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -296296.651852,
-1185186.251852, -296296.651852, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 37037.148148, 148148.148148,
37037.148148, 0.000000, 0.000000, 0.000000 }, { 0.000000, 0.000000, 0.000000, -18518.551852, 259259.059259, 296297.540741,
259259.059259, -18518.551852, 0.000000, 0.000000, 0.000000, 0.000000, -74073.962963, -296296.651852, -148148.548148,
-296296.651852, -74073.962963, 0.000000, 0.000000, 0.000000, 0.000000, 92592.570370, 37037.148148, -148148.214815, 37037.148148,
92592.570370, 0.000000 }, { 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 259259.059259, 1037038.992593, 259259.059259,
0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -296296.651852, -1185186.251852, -296296.651852, 0.000000, 0.000000,
0.000000, 0.000000, 0.000000, 0.000000, 37037.148148, 148148.148148, 37037.148148, 0.000000 }, { 0.000000, 0.000000, 0.000000,
0.000000, 0.000000, -18518.551852, 259259.059259, 296297.540741, 259259.059260, 0.000000, 0.000000, 0.000000, 0.000000,
0.000000, -74073.962963, -296296.651852, -148148.548148, -296296.651852, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
92592.570370, 37037.148148, -148148.214815, 37037.148148 }, { 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
0.000000, 259259.059260, 1037038.992593, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -296296.651852,
-1185186.251852, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 37037.148148, 148148.148147 }, {
-1185186.251852, -296296.651852, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 2370376.059259,
592591.525926, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -1185186.251852, -296296.651852, 0.000000,
0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 }, { -296296.651852, -148148.548148, -296296.651852, -74073.962963,
0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 592591.525926, 296300.207407, 592591.525926, 148148.148148, 0.000000,
0.000000, 0.000000, 0.000000, 0.000000, -296296.651852, -148148.548148, -296296.651852, -74073.962963, 0.000000, 0.000000,
0.000000, 0.000000, 0.000000 }, { 0.000000, -296296.651852, -1185186.251852, -296296.651852, 0.000000, 0.000000, 0.000000,
0.000000, 0.000000, 0.000000, 592591.525926, 2370376.059259, 592591.525926, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
0.000000, -296296.651852, -1185186.251852, -296296.651852, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 }, { 0.000000,
-74073.962963, -296296.651852, -148148.548148, -296296.651852, -74073.962963, 0.000000, 0.000000, 0.000000, 0.000000,
148148.148148, 592591.525926, 296300.207407, 592591.525926, 148148.148148, 0.000000, 0.000000, 0.000000, 0.000000,
-74073.962963, -296296.651852, -148148.548148, -296296.651852, -74073.962963, 0.000000, 0.000000, 0.000000 }, { 0.000000,
0.000000, 0.000000, -296296.651852, -1185186.251852, -296296.651852, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
592591.525926, 2370376.059259, 592591.525926, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -296296.651852,
-1185186.251852, -296296.651852, 0.000000, 0.000000, 0.000000 }, { 0.000000, 0.000000, 0.000000, -74073.962963, -296296.651852,
-148148.548148, -296296.651852, -74073.962963, 0.000000, 0.000000, 0.000000, 0.000000, 148148.148148, 592591.525926,
296300.207407, 592591.525926, 148148.148148, 0.000000, 0.000000, 0.000000, 0.000000, -74073.962963, -296296.651852,
-148148.548148, -296296.651852, -74073.962963, 0.000000 }, { 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -296296.651852,
-1185186.251852, -296296.651852, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 592591.525926, 2370376.059259,
592591.525926, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -296296.651852, -1185186.251852, -296296.651852,
0.000000 }, { 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -74073.962963, -296296.651852, -148148.548148, -296296.651852,
0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 148148.148148, 592591.525926, 296300.207407, 592591.525926, 0.000000,
0.000000, 0.000000, 0.000000, 0.000000, -74073.962963, -296296.651852, -148148.548148, -296296.651852 }, { 0.000000, 0.000000,
0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -296296.651852, -1185186.251852, 0.000000, 0.000000, 0.000000, 0.000000,
0.000000, 0.000000, 0.000000, 592591.525926, 2370376.059259, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
0.000000, -296296.651852, -1185186.251852 }, { 148148.148148, 37037.148148, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
0.000000, 0.000000, -1185186.251852, -296296.651852, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
1037038.992593, 259259.059259, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 }, { 37037.148148,
-148148.214815, 37037.148148, 92592.570370, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -296296.651852, -148148.548148,
-296296.651852, -74073.962963, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 259259.059259, 296297.540741, 259259.059259,
-18518.551852, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 }, { 0.000000, 37037.148148, 148148.148148, 37037.148148,
0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -296296.651852, -1185186.251852, -296296.651852, 0.000000, 0.000000,
0.000000, 0.000000, 0.000000, 0.000000, 259259.059259, 1037038.992593, 259259.059259, 0.000000, 0.000000, 0.000000, 0.000000,
0.000000 }, { 0.000000, 92592.570370, 37037.148148, -148148.214815, 37037.148148, 92592.570370, 0.000000, 0.000000, 0.000000,
0.000000, -74073.962963, -296296.651852, -148148.548148, -296296.651852, -74073.962963, 0.000000, 0.000000, 0.000000, 0.000000,
-18518.551852, 259259.059259, 296297.540741, 259259.059259, -18518.551852, 0.000000, 0.000000, 0.000000 }, { 0.000000, 0.000000,
0.000000, 37037.148148, 148148.148148, 37037.148148, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -296296.651852,
-1185186.251852, -296296.651852, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 259259.059259, 1037038.992592,
259259.059259, 0.000000, 0.000000, 0.000000 }, { 0.000000, 0.000000, 0.000000, 92592.570370, 37037.148148, -148148.214815,
37037.148148, 92592.570370, 0.000000, 0.000000, 0.000000, 0.000000, -74073.962963, -296296.651852, -148148.548148,
-296296.651852, -74073.962963, 0.000000, 0.000000, 0.000000, 0.000000, -18518.551852, 259259.059259, 296297.540741,
259259.059259, -18518.551852, 0.000000 }, { 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 37037.148148, 148148.148148,
37037.148148, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -296296.651852, -1185186.251852, -296296.651852,
0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 259259.059259, 1037038.992593, 259259.059259, 0.000000 }, {
0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 92592.570370, 37037.148148, -148148.214815, 37037.148148, 0.000000, 0.000000,
0.000000, 0.000000, 0.000000, -74073.962963, -296296.651852, -148148.548148, -296296.651852, 0.000000, 0.000000, 0.000000,
0.000000, 0.000000, -18518.551852, 259259.059259, 296297.540741, 259259.059259 }, { 0.000000, 0.000000, 0.000000, 0.000000,
0.000000, 0.000000, 0.000000, 37037.148148, 148148.148147, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
-296296.651852, -1185186.251852, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 259259.059259,
1037038.992593 } } );
   // clang-format on
   VectorType x( size, 1.0 ), b( size, 0.0 ), y( size, 0.0 );
   matrix->vectorProduct( x, b );

   TNL::Solvers::Linear::GEM< Matrix > gem;
   gem.setMatrix( matrix );
   gem.setPivoting( true );
   gem.solve( b, y );

   EXPECT_NEAR( maxNorm( x - y ), 0.0, 1.0e-6 );
}

TYPED_TEST_SUITE( GEMSolverTest, MatrixTypes );

TYPED_TEST( GEMSolverTest, diagonalMatrix )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_diagonalMatrix< MatrixType >();
}

TYPED_TEST( GEMSolverTest, upperTriangularMatrix )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_upperTriangularMatrix< MatrixType >();
}

TYPED_TEST( GEMSolverTest, smallMatrix )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_smallMatrix< MatrixType >();
}

TYPED_TEST( GEMSolverTest, largeMatrix )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_largeMatrix< MatrixType >();
}

TYPED_TEST( GEMSolverTest, Cage3Matrix )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_Cage3Matrix< MatrixType >();
}

TYPED_TEST( GEMSolverTest, Cage4Matrix )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_Cage4Matrix< MatrixType >();
}

TYPED_TEST( GEMSolverTest, Ex5Matrix )
{
   using MatrixType = typename TestFixture::MatrixType;

   test_Ex5Matrix< MatrixType >();
}

#include "../../main.h"
