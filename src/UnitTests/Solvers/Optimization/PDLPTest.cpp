#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Solvers/Optimization/PDLP.h>

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

#ifdef undef
TEST( PDLPTest, NetworkFlowProblemTest )
{
   using RealType = double;
   using VectorType = Containers::Vector< RealType >;
   using MatrixType = Matrices::DenseMatrix< RealType >;

   /****
    * We solve the following problem:
    * We have a network with source S, sink T and nodes A, B, C, ... H.
    * The following is table of the costs and capacities of the edges:
    *
    * Edge  Capacity  Cost
    * S-A   20        2
    * S-B   30        3
    * A-C   15        2
    * A-D   10        4
    * B-C   10        1
    * B-E   15        2
    * C-D    5        3
    * C-F   10        2
    * C-G   15        1
    * D-H   10        2
    * E-G   15        1
    * F-T   15        4
    * G-F   15        2
    * G-H   10        2
    * H-T   20        3
    *
    * The following is the transposed constraint matrix:
    *
    *       S	A	B	C	D	E	F	G	H	T
    *  S-A	-1	1	0	0	0	0	0	0	0	0
    *  S-B	-1	0	1	0	0	0	0	0	0	0
    *  A-C	0	-1	0	1	0	0	0	0	0	0
    *  A-D	0	-1	0	0	1	0	0	0	0	0
    *  B-C	0	0	-1	1	0	0	0	0	0	0
    *  B-E	0	0	-1	0	0	1	0	0	0	0
    *  C-D	0	0	0	-1	1	0	0	0	0	0
    *  C-F	0	0	0	-1	0	0	1	0	0	0
    *  C-G	0	0	0	-1	0	0	0	1	0	0
    *  D-H	0	0	0	0	-1	0	0	0	1	0
    *  E-G	0	0	0	0	0	-1	0	1	0	0
    *  F-T	0	0	0	0	0	0	-1	0	0	1
    *  G-F	0	0	0	0	0	0	1	-1	0	0
    *  G-H	0	0	0	0	0	0	0	-1	1	0
    *  H-T	0	0	0	0	0	0	0	0	-1	1
    *
    * Total Minimum Cost = 370.0
    */

   // clang-format off
   MatrixType GA( {
      // S-A  S-B  A-C  A-D  B-C  B-E  C-D  C-F  C-G  D-H  E-G  F-T  G-F  G-H  H-T
        { -1,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 }, // S
        {  1,   0,  -1,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 }, // A
        {  0,   1,   0,   0,  -1,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0 }, // B
        {  0,   0,   1,   0,   1,   0,  -1,  -1,  -1,   0,   0,   0,   0,   0,   0 }, // C
        {  0,   0,   0,   1,   0,   0,   1,   0,   0,  -1,   0,   0,   0,   0,   0 }, // D
        {  0,   0,   0,   0,   0,   1,   0,   0,   0,   0,  -1,   0,   0,   0,   0 }, // E
        {  0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,  -1,   1,   0,   0 }, // F
        {  0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   1,   0,  -1,  -1,   0 }, // G
        {  0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   1,  -1 }, // H
        {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   1 }, // T
   } );
   VectorType hb( { 35, 0, 0, 0, 0, 0, 0, 0, 0, 35 } );
   VectorType c( { 2, 3, 2, 4, 1, 2, 3, 2, 1, 2, 1, 4, 2, 2, 3 } );
   VectorType l( 15, 0 );
   VectorType u( { 20, 30, 15, 10, 10, 15, 5, 10, 15, 10, 15, 15, 15, 10, 20 } );
   VectorType exact_solution( { 20, 15, 10, 10, 10, 5, 0, 10, 10, 10, 5, 15, 5, 10, 20 });
   // clang-format on

   using LPProblemType = Solvers::Optimization::LPProblem< MatrixType >;
   LPProblemType lpProblem( GA, hb, 0, c, l, u );

   VectorType x( 15, 0 );
   Solvers::Optimization::PDLP< LPProblemType > solver;
   solver.solve( lpProblem, x );

   EXPECT_NEAR( TNL::max( TNL::abs( x - exact_solution ) ), (RealType) 0.0, 0.1 );
}
#endif

#include "../../main.h"
