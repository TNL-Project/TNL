#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/StaticVector.h>

#include <gtest/gtest.h>

using namespace TNL;
using namespace TNL::Containers;

// test fixture for typed tests
template< typename DofContainer >
class ODESolverTest : public ::testing::Test
{
protected:
   using DofContainerType = DofContainer;
};

// types for which DofContainerTest is instantiated
using DofVectorTypes = ::testing::Types<
#if defined( __CUDACC__ )
   Vector< float, Devices::Cuda, int >,
   Vector< double, Devices::Cuda, int >,
   Vector< float, Devices::Cuda, long >,
   Vector< double, Devices::Cuda, long >
#elif defined( __HIP__ )
   Vector< float, Devices::Hip, int >,
   Vector< double, Devices::Hip, int >,
   Vector< float, Devices::Hip, long >,
   Vector< double, Devices::Hip, long >
#else
   // we can't test all types because the argument list would be too long...
   Vector< float, Devices::Sequential, int >,
   Vector< double, Devices::Sequential, int >,
   Vector< float, Devices::Sequential, long >,
   Vector< double, Devices::Sequential, long >
#endif
   >;

TYPED_TEST_SUITE( ODESolverTest, DofVectorTypes );

TYPED_TEST( ODESolverTest, LinearFunctionTest )
{
   using DofContainerType = typename TestFixture::DofContainerType;
   using SolverType = ODETestSolver< DofContainerType >;
   using Real = typename DofContainerType::RealType;

   const Real final_time = 10.0;
   SolverType solver;
   solver.setTime( 0.0 );
   solver.setStopTime( final_time );
   solver.setTau( 0.005 );
   solver.setConvergenceResidue( 0.0 );

   DofContainerType u( 5, 0.0 );
   EXPECT_TRUE( solver.solve(
      u,
      []( const Real& time, const Real& tau, const auto& u, auto& fu )
      {
         fu = time;
      } ) );

   EXPECT_EQ( solver.getTime(), final_time );

   Real exact_solution = 0.5 * final_time * final_time;
   EXPECT_NEAR( TNL::max( TNL::abs( u - exact_solution ) ), (Real) 0.0, 0.1 );
}

TYPED_TEST( ODESolverTest, LinearFunctionTest_iterate )
{
   using DofContainerType = typename TestFixture::DofContainerType;
   using SolverType = ODETestSolver< DofContainerType >;
   using Real = typename DofContainerType::RealType;

   const Real final_time = 10.0;
   SolverType solver;
   solver.setTime( 0.0 );
   solver.setStopTime( final_time );
   solver.setTau( 0.005 );
   solver.setConvergenceResidue( 0.0 );

   DofContainerType u( 5, 0.0 );
   solver.init( u );
   while( solver.getTime() < final_time ) {
      solver.iterate( u,
                      []( const Real& time, const Real& tau, const auto& u, auto& fu )
                      {
                         fu = time;
                      } );
   }
   solver.reset();
   Real exact_solution = 0.5 * final_time * final_time;
   EXPECT_NEAR( TNL::max( TNL::abs( u - exact_solution ) ), (Real) 0.0, 0.1 );
}

#include "../../main.h"
