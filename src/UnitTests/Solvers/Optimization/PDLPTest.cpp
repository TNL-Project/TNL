#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Solvers/Optimization/PDLP.h>
#include <TNL/Solvers/Optimization/LPProblem.h>
#include <TNL/Solvers/Optimization/LPProblemReader.h>

#include <gtest/gtest.h>

using namespace TNL;

/***
 * The unit tests in this file solve the following problem:
 *
 * min c * x
 * s.t. G * x >= h
 *      A * x = b if A is given
 *      l <= x <= u
 */

TEST( PDLPTest, SmallProblemOnlyInequalitiesTest )
{
   using RealType = double;
   using VectorType = Containers::Vector< RealType >;
   using MatrixType = Matrices::DenseMatrix< RealType >;

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

   MatrixType G( { { -1, -2 }, { -5, -3 } } );
   VectorType h( { -16, -45 } );
   VectorType c( { -2, -5 } );
   VectorType l( { 0, 0 } );
   VectorType u( 2, std::numeric_limits< RealType >::infinity() );
   using LPProblemType = Solvers::Optimization::LPProblem< MatrixType >;
   LPProblemType lpProblem( G, h, 2, c, l, u );
   VectorType exact_solution( { 0, 8 } );
   VectorType x( 2, 0 );

   Solvers::Optimization::PDLP< LPProblemType > solver;
   solver.solve( lpProblem, x );

   EXPECT_NEAR( TNL::max( TNL::abs( x - exact_solution ) ), (RealType) 0.0, 0.1 );
}

TEST( PDLPTest, SmallProblemMixedConstraintsTest )
{
   using RealType = double;
   using VectorType = Containers::Vector< RealType >;
   using MatrixType = Matrices::DenseMatrix< RealType >;

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

   MatrixType GA( { { -5, -3 }, { -1, -2 } } );
   VectorType hb( { -45, -16 } );
   VectorType c( { -2, -5 } );
   VectorType l( { 0, 0 } );
   VectorType u( 2, std::numeric_limits< RealType >::infinity() );
   using LPProblemType = Solvers::Optimization::LPProblem< MatrixType >;
   LPProblemType lpProblem( GA, hb, 1, c, l, u );
   VectorType exact_solution( { 0, 8 } );
   VectorType x( 2, 0 );

   Solvers::Optimization::PDLP< LPProblemType > solver;
   solver.solve( lpProblem, x );

   EXPECT_NEAR( TNL::max( TNL::abs( x - exact_solution ) ), (RealType) 0.0, 0.1 );
}

TEST( PDLPTest, TransportationProblemTest )
{
   using RealType = double;
   using VectorType = Containers::Vector< RealType >;
   using MatrixType = Matrices::DenseMatrix< RealType >;

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

   MatrixType GA( { { -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0 },
                    { 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0 },
                    { 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1 },
                    { 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 },
                    { 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0 },
                    { 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0 },
                    { 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 } } );
   VectorType hb( { -50, -60, -50, 30, 40, 40, 50 } );
   VectorType c( { 4, 3, 2, 7, 2, 5, 4, 3, 5, 1, 3, 2 } );
   VectorType l( 12, 0 );
   VectorType u( 12, std::numeric_limits< RealType >::infinity() );
   using LPProblemType = Solvers::Optimization::LPProblem< MatrixType >;
   LPProblemType lpProblem( GA, hb, 3, c, l, u );
   VectorType exact_solution( { 0, 10, 40, 0, 30, 0, 0, 30, 0, 30, 0, 20 } );
   VectorType x( 12, 0 );

   Solvers::Optimization::PDLP< LPProblemType > solver;
   solver.solve( lpProblem, x );

   EXPECT_NEAR( TNL::max( TNL::abs( x - exact_solution ) ), (RealType) 0.0, 0.1 );
}

TEST( PDLPTest, MPSTest1 )
{
   using RealType = double;
   using VectorType = Containers::Vector< RealType >;
   using MatrixType = Matrices::DenseMatrix< RealType >;
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
   EXPECT_NEAR( cost, 3.23684, 1.0e-5 );
}

// TODO: Added test given by MPS - https://www.cenapad.unicamp.br/parque/manuais/OSL/oslweb/features/feat24DT.htm

#include "../../main.h"
