#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Solvers/ODE/ODESolver.h>

#include <gtest/gtest.h>

using namespace TNL;
using namespace TNL::Containers;

template< typename DofContainer >
class ODEStaticSolverTest : public ::testing::Test
{
protected:
   using DofContainerType = DofContainer;
   using ODEMethodType = ODEMethod;  // defined in the root header of the test
};

// types for which DofContainerTest is instantiated
using DofNumericTypes = ::testing::Types< float, double >;

// types for which DofContainerTest is instantiated
using DofStaticVectorTypes = ::testing::Types< StaticVector< 1, float >,
                                               StaticVector< 2, float >,
                                               StaticVector< 3, float >,
                                               StaticVector< 1, double >,
                                               StaticVector< 2, double >,
                                               StaticVector< 3, double > >;

TYPED_TEST_SUITE( ODEStaticSolverTest, DofStaticVectorTypes );

template< typename DofContainerType, typename SolverType >
void
ODEStaticSolverTest_LinearFunctionTest()
{
   using StaticVectorType = DofContainerType;
   using RealType = typename DofContainerType::RealType;

   const RealType final_time = 10.0;
   SolverType solver;
   solver.setTime( 0.0 );
   solver.setStopTime( final_time );
   solver.setTau( 0.005 );
   solver.setConvergenceResidue( 0.0 );

   DofContainerType u( 0.0 );
   EXPECT_TRUE( solver.solve(
      u,
      [] __cuda_callable__( const RealType& time, const RealType& tau, const StaticVectorType& u, StaticVectorType& fu )
      {
         fu = time;
      } ) );

   EXPECT_EQ( solver.getTime(), final_time );

   RealType exact_solution = 0.5 * final_time * final_time;
   EXPECT_NEAR( TNL::max( TNL::abs( u - exact_solution ) ), (RealType) 0.0, 0.1 );
}

TYPED_TEST( ODEStaticSolverTest, LinearFunctionTest )
{
   using DofContainerType = typename TestFixture::DofContainerType;
   using ODEMethodType = typename TestFixture::ODEMethodType;
   using SolverType = TNL::Solvers::ODE::ODESolver< ODEMethodType, DofContainerType >;

   ODEStaticSolverTest_LinearFunctionTest< DofContainerType, SolverType >();
}

template< typename DofContainerType, typename SolverType, typename Device >
void
ODEStaticSolverTest_ParallelLinearFunctionTest()
{
   using RealType = typename DofContainerType::RealType;

   const int size = 10;
   const RealType final_time = 10.0;
   TNL::Containers::Vector< DofContainerType, Device > u( size, 0.0 );
   auto u_view = u.getView();
   // inner_f cannot be defined inside f because it is not accepted by nvcc compiler
   auto inner_f =
      [ = ] __cuda_callable__( const RealType& time, const RealType& tau, const DofContainerType& u, DofContainerType& fu )
   {
      fu = time;
   };
   auto f = [ = ] __cuda_callable__( int idx ) mutable
   {
      SolverType solver;
      solver.setTime( 0.0 );
      solver.setStopTime( final_time );
      solver.setTau( 0.005 );
      solver.setConvergenceResidue( 0.0 );
      const bool status = solver.solve( u_view[ idx ], inner_f );
#if ! defined( __CUDA_ARCH__ ) && ! defined( __HIP_DEVICE_COMPILE__ )
      // gtest macros do not work inside GPU kernels (dynamic allocation with std::unique_ptr)
      EXPECT_TRUE( status );
      EXPECT_EQ( solver.getTime(), final_time );
#else
      (void) status;
#endif
   };
   TNL::Algorithms::parallelFor< Device >( 0, size, f );

   RealType exact_solution( 0.5 * final_time * final_time );
   auto error = TNL::Algorithms::reduce< Device >(
      0,
      size,
      [ = ] __cuda_callable__( int idx ) -> RealType
      {
         return TNL::max( u_view[ idx ] - exact_solution );
      },
      TNL::Max() );
   EXPECT_NEAR( error, (RealType) 0.0, 0.1 );
}

TYPED_TEST( ODEStaticSolverTest, ParallelLinearFunctionTest )
{
   using DofContainerType = typename TestFixture::DofContainerType;
   using ODEMethodType = typename TestFixture::ODEMethodType;
   using SolverType = TNL::Solvers::ODE::ODESolver< ODEMethodType, DofContainerType >;

#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   ODEStaticSolverTest_ParallelLinearFunctionTest< DofContainerType, SolverType, TNL::Devices::Host >();
#endif

#ifdef __CUDACC__
   ODEStaticSolverTest_ParallelLinearFunctionTest< DofContainerType, SolverType, TNL::Devices::Cuda >();
#endif

#ifdef __HIP__
   ODEStaticSolverTest_ParallelLinearFunctionTest< DofContainerType, SolverType, TNL::Devices::Hip >();
#endif
}

template< typename DofContainerType, typename SolverType >
void
ODEStaticSolverTest_EOCTest()
{
   using StaticVectorType = DofContainerType;
   using RealType = typename DofContainerType::RealType;

   const RealType final_time = 1.0;
   auto f =
      [ = ] __cuda_callable__( const RealType& time, const RealType& tau, const StaticVectorType& u, StaticVectorType& fu )
   {
      fu = TNL::exp( time );
   };

   StaticVectorType u1( 0.0 );
   StaticVectorType u2( 0.0 );
   SolverType solver;
   solver.setStopTime( final_time );
   solver.setConvergenceResidue( 0.0 );
   solver.setAdaptivity( 0.0 );

   solver.setTime( 0.0 );
   solver.setTau( 0.1 );
   EXPECT_TRUE( solver.solve( u1, f ) );
   EXPECT_EQ( solver.getTime(), final_time );

   solver.setTime( 0.0 );
   solver.setTau( 0.05 );
   EXPECT_TRUE( solver.solve( u2, f ) );
   EXPECT_EQ( solver.getTime(), final_time );

   const RealType exact_solution = exp( 1.0 ) - exp( 0.0 );
   const RealType error_1 = TNL::max( TNL::abs( u1 - exact_solution ) );
   const RealType error_2 = TNL::max( TNL::abs( u2 - exact_solution ) );
   const RealType eoc = log( error_1 / error_2 ) / log( 2.0 );
   EXPECT_NEAR( eoc, expected_eoc, 0.1 ) << "exact_solution = " << exact_solution << " u1 = " << u1 << " u2 = " << u2
                                         << " error_1 = " << error_1 << " error_2 = " << error_2 << " eoc = " << eoc;
}

TYPED_TEST( ODEStaticSolverTest, EOCTest )
{
   using DofContainerType = typename TestFixture::DofContainerType;
   using ODEMethodType = typename TestFixture::ODEMethodType;
   using SolverType = TNL::Solvers::ODE::ODESolver< ODEMethodType, DofContainerType >;

   if constexpr( std::is_same_v< DofContainerType, StaticVector< 1, double > > )
      ODEStaticSolverTest_EOCTest< DofContainerType, SolverType >();
}

template< typename DofContainerType, typename SolverType >
void
ODEStaticSolverTest_EOCTest_iterate()
{
   using StaticVectorType = DofContainerType;
   using RealType = typename DofContainerType::RealType;

   const RealType final_time = 1.0;
   auto f =
      [ = ] __cuda_callable__( const RealType& time, const RealType& tau, const StaticVectorType& u, StaticVectorType& fu )
   {
      fu = TNL::exp( time );
   };

   StaticVectorType u1( 0.0 );
   StaticVectorType u2( 0.0 );
   SolverType solver;
   solver.init( u1 );
   solver.setStopTime( final_time );
   solver.setConvergenceResidue( 0.0 );
   solver.setAdaptivity( 0.0 );

   solver.setTime( 0.0 );
   solver.setTau( 0.1 );
   while( solver.getTime() < final_time ) {
      solver.iterate( u1, f );
   }
   solver.setTime( 0.0 );
   solver.setTau( 0.05 );
   while( solver.getTime() < final_time ) {
      solver.iterate( u2, f );
   }
   solver.reset();

   const RealType exact_solution = exp( 1.0 ) - exp( 0.0 );
   const RealType error_1 = TNL::max( TNL::abs( u1 - exact_solution ) );
   const RealType error_2 = TNL::max( TNL::abs( u2 - exact_solution ) );
   const RealType eoc = log( error_1 / error_2 ) / log( 2.0 );
   EXPECT_NEAR( eoc, expected_eoc, 0.1 ) << "exact_solution = " << exact_solution << " u1 = " << u1 << " u2 = " << u2
                                         << " error_1 = " << error_1 << " error_2 = " << error_2 << " eoc = " << eoc;
}

TYPED_TEST( ODEStaticSolverTest, EOCTest_iterate )
{
   using DofContainerType = typename TestFixture::DofContainerType;
   using ODEMethodType = typename TestFixture::ODEMethodType;
   using SolverType = TNL::Solvers::ODE::ODESolver< ODEMethodType, DofContainerType >;

   if constexpr( std::is_same_v< DofContainerType, StaticVector< 1, double > > )
      ODEStaticSolverTest_EOCTest< DofContainerType, SolverType >();
}

#include "../../main.h"
