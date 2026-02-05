#include <TNL/Containers/Vector.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Solvers/Linear/UmfpackWrapper.h>

#include <gtest/gtest.h>

using namespace TNL;
using namespace TNL::Containers;

#ifdef HAVE_UMFPACK
   #include <umfpack.h>

// test fixture for typed tests
template< typename DofContainer >
class UmfpackWrapperTest : public ::testing::Test
{
protected:
   using DofContainerType = DofContainer;
};

// types for which DofContainerTest is instantiated
using DofVectorTypes = ::testing::Types<  //Vector< float, Devices::Host, int >,
   Vector< double, Devices::Host, int > >;

TYPED_TEST_SUITE( UmfpackWrapperTest, DofVectorTypes );

TYPED_TEST( UmfpackWrapperTest, LinearFunctionTest )
{
   using DofContainerType = typename TestFixture::DofContainerType;
   using Real = typename DofContainerType::RealType;
   using Index = typename DofContainerType::IndexType;
   using CSRMatrix = Matrices::SparseMatrix< Real, Devices::Host, Index, Matrices::GeneralMatrix, Algorithms::Segments::CSR >;

   // Test with the following matrix:
   //   1   2    3   4   5  <-  x
   // --------------------------------
   // | 2   3               | = |  8 |
   // | 3        4       6  | = | 45 |
   // |    -1   -3   2      | = | -3 |
   // |          1          | = |  3 |
   // |     4    2       1  | = | 19 |
   // clang-format off
   const int size = 5;
   CSRMatrix csr_matrix( size, size,
   {
      {0, 0,  2.}, {0, 1,  3.},
      {1, 0,  3.},              {1, 2,  4.},             {1, 4, 6.},
                   {2, 1, -1.}, {2, 2, -3.}, {2, 3, 2.},
                                {3, 2,  1.},
                   {4, 1,  4.}, {4, 2,  2.},             {4, 4, 1.}
   } );
   // clang-format on
   auto matrix_pointer = std::make_shared< CSRMatrix >( csr_matrix );
   Solvers::Linear::UmfpackWrapper< CSRMatrix > umfpack;
   Containers::Vector< double > vec_b{ 8, 45, -3, 3, 19 };
   Containers::Vector< double > vec_x( size, 0 );
   umfpack.setMatrix( matrix_pointer );
   umfpack.solve( vec_b.getConstView(), vec_x.getView() );

   EXPECT_NEAR( vec_x[ 0 ], 1, 1.0e-5 );
   EXPECT_NEAR( vec_x[ 1 ], 2, 1.0e-5 );
   EXPECT_NEAR( vec_x[ 2 ], 3, 1.0e-5 );
   EXPECT_NEAR( vec_x[ 3 ], 4, 1.0e-5 );
   EXPECT_NEAR( vec_x[ 4 ], 5, 1.0e-5 );
}

#endif
#include "../../main.h"
