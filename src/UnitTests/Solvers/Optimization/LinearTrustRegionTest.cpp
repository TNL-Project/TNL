#include <gtest/gtest.h>

#include <TNL/Containers/Vector.h>
#include <TNL/Solvers/Optimization/LinearTrustRegion.h>

using Vector = TNL::Containers::Vector< double >;

TEST( LinearTrustRegionTest, ZeroRadius )
{
   Vector z = { 1.0, 2.0, 3.0 };
   Vector l = { 0.0, 0.0, 0.0 };
   Vector u( 3, std::numeric_limits< double >::infinity() );
   Vector g = { 1.0, 1.0, 1.0 };
   double r = 0.0;
   Vector z_hat( z.getSize() );

   TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, r, z_hat );

   EXPECT_EQ( z, z_hat );
}

TEST( LinearTrustRegionTest, ZeroGradient )
{
   Vector z = { 1.0, 2.0, 3.0 };
   Vector l = { 0.5, 1.5, 2.5 };
   Vector u( 3, std::numeric_limits< double >::infinity() );
   Vector g = { 0.0, 0.0, 0.0 };
   double r = 1.0;
   Vector z_hat( z.getSize() );

   EXPECT_FALSE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, r, z_hat ) );
}

TEST( LinearTrustRegionTest, LowerBoundConstraint )
{
   Vector z = { 1.5, 2.0, 3.0 };
   Vector l = { 1.5, 1.5, 1.5 };
   Vector u( 3, std::numeric_limits< double >::infinity() );
   Vector g = { 1.0, 1.0, 1.0 };
   double r = 1.0;
   Vector z_hat( z.getSize() );

   TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, r, z_hat );

   for( size_t i = 0; i < z.getSize(); ++i ) {
      EXPECT_GE( z_hat[ i ], l[ i ] );
   }
}

TEST( LinearTrustRegionTest, GrowingRadiusToReachLimit )
{
   using Real = typename Vector::RealType;
   Vector z( 8, 0.0 );
   Vector l = { -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0 };
   Vector u( 8, std::numeric_limits< double >::infinity() );
   Vector g( 8, 1.0 );
   Vector z_hat( z.getSize() );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 1.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 1.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 2.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 2.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 3.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 3.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 4.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 4.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 5.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 5.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 6.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 6.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 7.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 7.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 8.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 8.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 9.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 9.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 10.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 10.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 11.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 11.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 12.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 12.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 13.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 13.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 14.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 14.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 15.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), TNL::l2Norm( l - z ), std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 16.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), TNL::l2Norm( l - z ), std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );
}

TEST( LinearTrustRegionTest, GrowingRadiusToReachLimitWithUpperBound )
{
   using Real = typename Vector::RealType;
   Vector z( 8, 0.0 );
   Vector l = { -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0 };
   Vector u( 8, 4.0 );
   Vector g = { 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0 };
   Vector z_hat( z.getSize() );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 1.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 1.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );
   EXPECT_TRUE( TNL::all( lessEqual( z_hat, u ) ) );
   EXPECT_TRUE( TNL::all( equalTo( TNL::sign( z_hat ), TNL::sign( -g ) ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 2.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 2.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );
   EXPECT_TRUE( TNL::all( lessEqual( z_hat, u ) ) );
   EXPECT_TRUE( TNL::all( equalTo( TNL::sign( z_hat ), TNL::sign( -g ) ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 3.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 3.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );
   EXPECT_TRUE( TNL::all( lessEqual( z_hat, u ) ) );
   EXPECT_TRUE( TNL::all( equalTo( TNL::sign( z_hat ), TNL::sign( -g ) ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 4.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 4.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );
   EXPECT_TRUE( TNL::all( lessEqual( z_hat, u ) ) );
   EXPECT_TRUE( TNL::all( equalTo( TNL::sign( z_hat ), TNL::sign( -g ) ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 5.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 5.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );
   EXPECT_TRUE( TNL::all( lessEqual( z_hat, u ) ) );
   EXPECT_TRUE( TNL::all( equalTo( TNL::sign( z_hat ), TNL::sign( -g ) ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 6.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 6.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );
   EXPECT_TRUE( TNL::all( lessEqual( z_hat, u ) ) );
   EXPECT_TRUE( TNL::all( equalTo( TNL::sign( z_hat ), TNL::sign( -g ) ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 7.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 7.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );
   EXPECT_TRUE( TNL::all( lessEqual( z_hat, u ) ) );
   EXPECT_TRUE( TNL::all( equalTo( TNL::sign( z_hat ), TNL::sign( -g ) ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 8.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 8.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );
   EXPECT_TRUE( TNL::all( lessEqual( z_hat, u ) ) );
   EXPECT_TRUE( TNL::all( equalTo( TNL::sign( z_hat ), TNL::sign( -g ) ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 9.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 9.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );
   EXPECT_TRUE( TNL::all( lessEqual( z_hat, u ) ) );
   EXPECT_TRUE( TNL::all( equalTo( TNL::sign( z_hat ), TNL::sign( -g ) ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 10.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 10.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );
   EXPECT_TRUE( TNL::all( lessEqual( z_hat, u ) ) );
   EXPECT_TRUE( TNL::all( equalTo( TNL::sign( z_hat ), TNL::sign( -g ) ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 11.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 11.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );
   EXPECT_TRUE( TNL::all( lessEqual( z_hat, u ) ) );
   EXPECT_TRUE( TNL::all( equalTo( TNL::sign( z_hat ), TNL::sign( -g ) ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 12.0, z_hat ) );
   EXPECT_NEAR( TNL::l2Norm( z_hat - z ), 12.0, std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );
   EXPECT_TRUE( TNL::all( lessEqual( z_hat, u ) ) );
   EXPECT_TRUE( TNL::all( equalTo( TNL::sign( z_hat ), TNL::sign( -g ) ) ) );

   EXPECT_TRUE( TNL::Solvers::Optimization::linearTrustRegion( z, l, u, g, 13.0, z_hat ) );
   EXPECT_LE( TNL::l2Norm( z_hat - z ), 13.0 + std::numeric_limits< Real >::round_error() );
   EXPECT_TRUE( TNL::all( lessEqual( l, z_hat ) ) );
   EXPECT_TRUE( TNL::all( lessEqual( z_hat, u ) ) );
   EXPECT_TRUE( TNL::all( equalTo( TNL::sign( z_hat ), TNL::sign( -g ) ) ) );
}

#include "../../main.h"
