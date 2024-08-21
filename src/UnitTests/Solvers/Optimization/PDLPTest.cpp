#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Solvers/Optimization/PDLP.h>

#include <gtest/gtest.h>

using namespace TNL;

/***
 * The unit tests in this file solve the fillowing problem:
 *
 * min c * x
 * s.t. G * x >= h
 *      A * x = b if A is given
 *      l <= x <= u
 */

TEST( PDLPTest, NoATest )
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
   VectorType exact_solution( { 0, 8 } );
   VectorType x( 2, 0 );

   Solvers::Optimization::PDLP< VectorType > solver;
   solver.setRelaxation( 0.05, 0.05 );
   solver.solve( c, G, h, l, u, x );

   EXPECT_NEAR( TNL::max( TNL::abs( x - exact_solution ) ), (RealType) 0.0, 0.1 );
}

TEST( PDLPTest, AGTest )
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

   MatrixType G( { { -5, -3 } } );
   MatrixType A( { { -1, -2 } } );
   VectorType h( { -45 } );
   VectorType b( { -16 } );
   VectorType c( { -2, -5 } );
   VectorType l( { 0, 0 } );
   VectorType u( 2, std::numeric_limits< RealType >::infinity() );
   VectorType exact_solution( { 0, 8 } );
   VectorType x( 2, 0 );

   Solvers::Optimization::PDLP< VectorType > solver;
   solver.setRelaxation( 0.02, 0.02 );
   solver.solve( c, G, h, A, b, l, u, x );

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

   MatrixType G( { { -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0 },
                   { 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0 },
                   { 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1 } } );
   MatrixType A( { { 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 },
                   { 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0 },
                   { 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0 },
                   { 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 } } );
   VectorType h( { -50, -60, -50 } );
   VectorType b( { 30, 40, 40, 50 } );
   VectorType c( { 4, 3, 2, 7, 2, 5, 4, 3, 5, 1, 3, 2 } );
   VectorType l( 12, 0 );
   VectorType u( 12, std::numeric_limits< RealType >::infinity() );
   VectorType exact_solution( { 0, 10, 40, 0, 30, 0, 0, 30, 0, 30, 0, 20 } );
   VectorType x( 12, 0 );

   Solvers::Optimization::PDLP< VectorType > solver;
   solver.setRelaxation( 0.02, 0.02 );
   solver.solve( c, G, h, A, b, l, u, x );

   EXPECT_NEAR( TNL::max( TNL::abs( x - exact_solution ) ), (RealType) 0.0, 0.1 );
}

#include "../../main.h"
