#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Solvers/Optimization/PDLP.h>
#include <TNL/Solvers/Optimization/LPProblem.h>
#include <TNL/Solvers/Optimization/LPProblemReader.h>

#include <gtest/gtest.h>

using namespace TNL;

// test fixture for typed tests
template< typename Matrix >
class PDLPTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
};

// types for which PDLPTest is instantiated
using MatrixTypes = ::testing::Types<
#if defined( __CUDACC__ )
   Matrices::SparseMatrix< double, Devices::Cuda, int >,
#elif defined( __HIP__ )
   Matrices::SparseMatrix< double, Devices::Hip, int >,
#else
   Matrices::SparseMatrix< double, Devices::Sequential, int >
#endif
   >;

TYPED_TEST_SUITE( PDLPTest, MatrixTypes );

/***
 * The following unit tests solve LP problems of the following form:
 *
 * min c * x
 * s.t. G * x >= h
 *      A * x = b if A is given
 *      l <= x <= u
 *
 * and we denote K^T = [ G^T | A^T ] and q = [ h | b ].
 */

TYPED_TEST( PDLPTest, SmallProblemOnlyInequalitiesTest )
{
   using MatrixType = typename TestFixture::MatrixType;
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;
   using DenseMatrixType = Matrices::DenseMatrix< RealType, Devices::Host, IndexType >;

   /***
    * We solve the following problem:
    * min ( -2 * x1 - 5 * x2 )
    * s.t.     -x1 - 2 * x2 >= -16
    *      -5 * x1 - 3 * x2 >= -45
    *      0 <= x1 <= inf
    *      0 <= x2 <= inf
    * The exact solution is x1 = 0, x2 = 8.
    * The minimum value is -40.
    */

   DenseMatrixType K_dense( { { -1, -2 }, { -5, -3 } } );
   MatrixType K;
   K = K_dense;
   VectorType q( { -16, -45 } );
   VectorType c( { -2, -5 } );
   VectorType l( { 0, 0 } );
   VectorType u( 2, std::numeric_limits< RealType >::infinity() );
   using LPProblemType = Solvers::Optimization::LPProblem< MatrixType >;
   LPProblemType lpProblem( K, q, 2, true, c, l, u );
   VectorType exact_solution( { 0, 8 } );
   VectorType x( 2, 0 );

   Solvers::Optimization::PDLP< LPProblemType > solver;
   solver.solve( lpProblem, x );

   EXPECT_NEAR( TNL::max( TNL::abs( x - exact_solution ) ), (RealType) 0.0, 0.1 );
}

TYPED_TEST( PDLPTest, SmallProblemMixedConstraintsTest )
{
   using MatrixType = typename TestFixture::MatrixType;
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;
   using DenseMatrixType = Matrices::DenseMatrix< RealType, Devices::Host, IndexType >;

   /***
    * We solve the following problem:
    * min ( -2 * x1 - 5 * x2 )
    * s.t.     -x1 - 2 * x2  = -16
    *      -5 * x1 - 3 * x2 >= -45
    *      0 <= x1 <= inf
    *      0 <= x2 <= inf
    * The exact solution is x1 = 0, x2 = 8.
    * The minimum value is -40.
    */

   DenseMatrixType K_dense( { { -5, -3 }, { -1, -2 } } );
   MatrixType K;
   K = K_dense;
   VectorType q( { -45, -16 } );
   VectorType c( { -2, -5 } );
   VectorType l( { 0, 0 } );
   VectorType u( 2, std::numeric_limits< RealType >::infinity() );
   using LPProblemType = Solvers::Optimization::LPProblem< MatrixType >;
   LPProblemType lpProblem( K, q, 1, true, c, l, u );
   VectorType exact_solution( { 0, 8 } );
   VectorType x( 2, 0 );

   Solvers::Optimization::PDLP< LPProblemType > solver;
   solver.solve( lpProblem, x );

   EXPECT_NEAR( TNL::max( TNL::abs( x - exact_solution ) ), (RealType) 0.0, 0.1 );
}

TYPED_TEST( PDLPTest, TransportationProblemTest )
{
   using MatrixType = typename TestFixture::MatrixType;
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;
   using DenseMatrixType = Matrices::DenseMatrix< RealType, Devices::Host, IndexType >;

   /***
    * We solve the following problem:
    * min ( 4 * x1 + 3 * x2  + 2 * x3  + 7 * x4 +
            2 * x5 + 5 * x6  + 4 * x7  + 3 * x8 +
            5 * x9 + 1 * x10 + 3 * x11 + 2 * x12 )
    * s.t. - x1 - x2  - x3  - x4                                              >= -50
    *                            - x5 - x6  - x7  - x8                        >= -60
    *                                                  - x9 - x10 - x11 - x12 >= -50
    *        x1                  + x5                  + x9                    = 30
    *             x2                  + x6                  + x10              = 40
    *                   x3                  + x7                  + x11        = 40
    *                         x4                  + x8                  + x12  = 50
    *
    *      0 <= xi for all i = 1, ..., 12
    *
    * The exact solution is ( 0, 10, 40, 0, 30, 0, 0, 30, 0, 30, 0, 20 ).
    * The minimum value is 330.
    */

   DenseMatrixType K_dense( { { -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0 },
                              { 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0 },
                              { 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1 },
                              { 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 },
                              { 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0 },
                              { 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0 },
                              { 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 } } );
   MatrixType K;
   K = K_dense;
   VectorType q( { -50, -60, -50, 30, 40, 40, 50 } );
   VectorType c( { 4, 3, 2, 7, 2, 5, 4, 3, 5, 1, 3, 2 } );
   VectorType l( 12, 0 );
   VectorType u( 12, std::numeric_limits< RealType >::infinity() );
   using LPProblemType = Solvers::Optimization::LPProblem< MatrixType >;
   LPProblemType lpProblem( K, q, 3, true, c, l, u );
   VectorType exact_solution( { 0, 10, 40, 0, 30, 0, 0, 30, 0, 30, 0, 20 } );
   VectorType x( 12, 0 );

   Solvers::Optimization::PDLP< LPProblemType > solver;
   solver.solve( lpProblem, x );

   EXPECT_NEAR( TNL::max( TNL::abs( x - exact_solution ) ), (RealType) 0.0, 0.1 );
}

TYPED_TEST( PDLPTest, LargerLPProblemTest )
{
   using MatrixType = typename TestFixture::MatrixType;
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;

   MatrixType K( 13,
                 24,
                 // clang-format off
                 {{0,0,-1},  { 0, 1, -1 }, { 0, 6,  1 },
                 {1,0,1},  { 1, 2,  1 }, { 1, 4, -1 }, { 1, 5, 1 }, { 1,6,-1 },
                 {2,2,-1},{2,3,-1},{2,7,1},
                 {3,1,1},{3,3,1},{3,4,1},{3,5,-1},{3,7,-1},
                 {4,8,-1},{4,9,-1},{4,14,1},
                 {5,8,1},{5,10,1},{5,12,-1},{5,13,1},{5,14,-1},
                 {6,10,-1},{6,11,-1},{6,15,1},
                 {7,9,1},{7,11,1},{7,12,1},{7,13,-1},{7,15,-1},
                 {8,16,-1},{8,17,-1},{8,22,1},
                 {9,16,1},{9,18,1},{9,20,-1},{9,21,1},{9,22,-1},
                 {10,18,-1},{10,19,-1},{10,23,1},
                 {11,17,1},{11,19,1},{11,20,1},{11,21,-1},{11,23,-1},
                 {12,1,-1},{12,9,-1},{12,17,-1}}  // clang-format on
   );

   VectorType q( { -1, 1, -1, 1, -1, 1, -2, 2, -1, 1, -3, 3, -1 } );
   VectorType c( { 1.0, 2.0, 3.0, 4.2, 5.0, 6.2, 5.0, 6.2, 1.0, 2.0, 3.0, 4.2,
                   5.0, 6.2, 5.0, 6.2, 1.0, 2.0, 3.0, 4.2, 5.0, 6.2, 5.0, 6.2 } );
   VectorType l( c.getSize(), 0 );
   VectorType u( c.getSize(), std::numeric_limits< RealType >::infinity() );
   using LPProblemType = Solvers::Optimization::LPProblem< MatrixType >;
   LPProblemType lpProblem( K, q, 12, false, c, l, u );
   VectorType exact_solution( {
      0.6674747628519729, 0.33255197275511167, 0.33250228351893557, 0.6674900963208183, 0.0, 0.0, 0.0, 0.0,
      0.7064361385953793, 0.293590609676839,   0.29354092614957256, 1.706451473625096,  0.0, 0.0, 0.0, 0.0,
      0.6262331146270321, 0.3737936344340823,  0.37374395124470894, 2.626248449761962,  0.0, 0.0, 0.0, 0.0,
   } );
   VectorType x( c.getSize(), 0 );

   Solvers::Optimization::PDLP< LPProblemType > solver;
   solver.setInequalitiesFirst( false );
   auto [ converged, cost, error ] = solver.solve( lpProblem, x );
   std::cout << "x = " << x << std::endl;
   EXPECT_TRUE( converged );
   EXPECT_NEAR( cost, 28, 1.0e-4 );
   EXPECT_NEAR( TNL::max( TNL::abs( x - exact_solution ) ), (RealType) 0.0, 0.1 );
}

// Tests based on MPS instances from - https://www.cenapad.unicamp.br/parque/manuais/OSL/oslweb/features/feat24DT.htm

TYPED_TEST( PDLPTest, MPSTest1 )
{
   using MatrixType = typename TestFixture::MatrixType;
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;
   using LPProblemType = Solvers::Optimization::LPProblem< MatrixType >;

   const char* mps = " \
************************************************************************\n \
*                                                                       \n \
*  The data in this file represents the following problem:              \n \
*                                                                       \n \
*  Minimize or maximize Z = x1 + 2x5 - x8                               \n \
*                                                                       \n \
*  Subject to:                                                          \n \
*                                                                       \n \
*  2.5 <=   3x1 +  x2          - 2x4  - x5              -    x8         \n \
*                 2x2 + 1.1x3                                   <=  2.1 \n \
*                          x3              + x6                  =  4.0 \n \
*  1.8 <=                      2.8x4             -1.2x7         <=  5.0 \n \
*  3.0 <= 5.6x1                       + x5              + 1.9x8 <= 15.0 \n \
*                                                                       \n \
*  where:                                                               \n \
*                                                                       \n \
*  2.5 <= x1 <= 4.1                                                     \n \
*    0 <= x2 <= 4.1                                                     \n \
*    0 <= x3 <= 4.1                                                     \n \
*    0 <= x4 <= 4.1                                                     \n \
*  0.5 <= x5 <= 4.0                                                     \n \
*    0 <= x6 <= 4.1                                                     \n \
*    0 <= x7 <= 4.1                                                     \n \
*    0 <= x8 <= 4.3                                                     \n \
*                                                                       \n \
************************************************************************\n \
NAME          EXAMPLE                                                   \n \
ROWS                                                                    \n \
 N  OBJ                                                                 \n \
 G  ROW01                                                               \n \
 L  ROW02                                                               \n \
 E  ROW03                                                               \n \
 G  ROW04                                                               \n \
 L  ROW05                                                               \n \
COLUMNS                                                                 \n \
    COL01     OBJ                1.0                                    \n \
    COL01     ROW01              3.0   ROW05              5.6           \n \
    COL02     ROW01              1.0   ROW02              2.0           \n \
    COL03     ROW02              1.1   ROW03              1.0           \n \
    COL04     ROW01             -2.0   ROW04              2.8           \n \
    COL05     OBJ                2.0                                    \n \
    COL05     ROW01             -1.0   ROW05              1.0           \n \
    COL06     ROW03              1.0                                    \n \
    COL07     ROW04             -1.2                                    \n \
    COL08     OBJ               -1.0                                    \n \
    COL08     ROW01             -1.0   ROW05              1.9           \n \
RHS                                                                     \n \
    RHS1      ROW01              2.5                                    \n \
    RHS1      ROW02              2.1                                    \n \
    RHS1      ROW03              4.0                                    \n \
    RHS1      ROW04              1.8                                    \n \
    RHS1      ROW05             15.0                                    \n \
RANGES                                                                  \n \
    RNG1      ROW04              3.2                                    \n \
    RNG1      ROW05             12.0                                    \n \
BOUNDS                                                                  \n \
 LO BND1      COL01              2.5                                    \n \
 UP BND1      COL02              4.1                                    \n \
 LO BND1      COL05              0.5                                    \n \
 UP BND1      COL05              4.0                                    \n \
 UP BND1      COL08              4.3                                    \n \
ENDATA                                                                  \n";

   std::istringstream iss( mps );
   TNL::Solvers::Optimization::LPProblemReader< LPProblemType > reader;
   auto lpProblem = reader.read( iss );
   typename LPProblemType::VectorType x( lpProblem.getVariableCount() );
   TNL::Solvers::Optimization::PDLP< LPProblemType > solver;
   auto [ converged, cost, error ] = solver.solve( lpProblem, x );
   EXPECT_TRUE( converged );
   EXPECT_NEAR( cost, 3.23684, 2.0e-4 );
}

TYPED_TEST( PDLPTest, MPSTest2 )
{
   using MatrixType = typename TestFixture::MatrixType;
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;
   using LPProblemType = Solvers::Optimization::LPProblem< MatrixType >;

   //clang-format off
   const char* mps = " \
NAME          LPDCMP1                                         \n \
ROWS                                                          \n \
 N  OBJCTV01                                                  \n \
 E  B0101                                                     \n \
 E  B0102                                                     \n \
 E  B0103                                                     \n \
 E  B0104                                                     \n \
 E  B0205                                                     \n \
 E  B0206                                                     \n \
 E  B0207                                                     \n \
 E  B0208                                                     \n \
 E  B0309                                                     \n \
 E  B0310                                                     \n \
 E  B0311                                                     \n \
 E  B0312                                                     \n \
 L  CPL13                                                     \n \
COLUMNS                                                       \n \
    X00       OBJCTV01      1.000000   B0102         1.000000 \n \
    X00       B0101        -1.000000                          \n \
    X01       OBJCTV01      2.000000   B0104         1.000000 \n \
    X01       B0101        -1.000000   CPL13         1.000000 \n \
    X02       OBJCTV01      3.000000   B0102         1.000000 \n \
    X02       B0103        -1.000000                          \n \
    X03       OBJCTV01      4.200000   B0104         1.000000 \n \
    X03       B0103        -1.000000                          \n \
    X04       OBJCTV01      5.000000   B0104         1.000000 \n \
    X04       B0102        -1.000000                          \n \
    X05       OBJCTV01      6.200000   B0102         1.000000 \n \
    X05       B0104        -1.000000                          \n \
    X06       OBJCTV01      5.000000   B0101         1.000000 \n \
    X06       B0102        -1.000000                          \n \
    X07       OBJCTV01      6.200000   B0103         1.000000 \n \
    X07       B0104        -1.000000                          \n \
    X10       OBJCTV01      1.000000   B0206         1.000000 \n \
    X10       B0205        -1.000000                          \n \
    X11       OBJCTV01      2.000000   B0208         1.000000 \n \
    X11       B0205        -1.000000   CPL13         1.000000 \n \
    X12       OBJCTV01      3.000000   B0206         1.000000 \n \
    X12       B0207        -1.000000                          \n \
    X13       OBJCTV01      4.200000   B0208         1.000000 \n \
    X13       B0207        -1.000000                          \n \
    X14       OBJCTV01      5.000000   B0208         1.000000 \n \
    X14       B0206        -1.000000                          \n \
    X15       OBJCTV01      6.200000   B0206         1.000000 \n \
    X15       B0208        -1.000000                          \n \
    X16       OBJCTV01      5.000000   B0205         1.000000 \n \
    X16       B0206        -1.000000                          \n \
    X17       OBJCTV01      6.200000   B0207         1.000000 \n \
    X17       B0208        -1.000000                          \n \
    X20       OBJCTV01      1.000000   B0310         1.000000 \n \
    X20       B0309        -1.000000                          \n \
    X21       OBJCTV01      2.000000   B0312         1.000000 \n \
    X21       B0309        -1.000000   CPL13         1.000000 \n \
    X22       OBJCTV01      3.000000   B0310         1.000000 \n \
    X22       B0311        -1.000000                          \n \
    X23       OBJCTV01      4.200000   B0312         1.000000 \n \
    X23       B0311        -1.000000                          \n \
    X24       OBJCTV01      5.000000   B0312         1.000000 \n \
    X24       B0310        -1.000000                          \n \
    X25       OBJCTV01      6.200000   B0310         1.000000 \n \
    X25       B0312        -1.000000                          \n \
    X26       OBJCTV01      5.000000   B0309         1.000000 \n \
    X26       B0310        -1.000000                          \n \
    X27       OBJCTV01      6.200000   B0311         1.000000 \n \
    X27       B0312        -1.000000                          \n \
RHS                                                           \n \
    RHS001    B0101        -1.000000   B0102         1.000000 \n \
    RHS001    B0103        -1.000000   B0104         1.000000 \n \
    RHS001    B0205        -1.000000   B0206         1.000000 \n \
    RHS001    B0207        -2.000000   B0208         2.000000 \n \
    RHS001    B0309        -1.000000   B0310         1.000000 \n \
    RHS001    B0311        -3.000000   B0312         3.000000 \n \
    RHS001    CPL13         1.000000                          \n \
ENDATA                                                        \n";
   // clang-format on

   std::istringstream iss( mps );
   TNL::Solvers::Optimization::LPProblemReader< LPProblemType > reader;
   auto lpProblem = reader.read( iss );
   typename LPProblemType::VectorType x( lpProblem.getVariableCount() );
   VectorType exact_solution( {
      0.6674747628519729, 0.33255197275511167, 0.33250228351893557, 0.6674900963208183, 0.0, 0.0, 0.0, 0.0,
      0.7064361385953793, 0.293590609676839,   0.29354092614957256, 1.706451473625096,  0.0, 0.0, 0.0, 0.0,
      0.6262331146270321, 0.3737936344340823,  0.37374395124470894, 2.626248449761962,  0.0, 0.0, 0.0, 0.0,
   } );

   TNL::Solvers::Optimization::PDLP< LPProblemType > solver;
   auto [ converged, cost, error ] = solver.solve( lpProblem, x );
   EXPECT_TRUE( converged );
   EXPECT_NEAR( cost, 28, 1.0e-4 );
   EXPECT_NEAR( TNL::max( TNL::abs( x - exact_solution ) ), (RealType) 0.0, 0.1 );
}

#include "../../main.h"
