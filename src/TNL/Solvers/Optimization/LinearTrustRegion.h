// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Backend/Macros.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Algorithms/contains.h>

#pragma once

namespace TNL::Solvers::Optimization {

template< typename Vector, typename VectorView >
void
linearTrustRegionNormalize( Vector& g, Vector& l, const VectorView& u );

/**
 * \brief Implementation of the linear trust region method.
 *
 * This implements the linear trust region method for a problem
 *
 * argmin[ dot( g, z_hat)] for  z_hat \in R^N ||z_hat - z|| <= r and l <= z_hat <= u
 *
 * See the following paper for more details (Alg. 2, page 179):
 *
 * Applegate, D., Hinder, O., Lu, H. et al. Faster first-order primal-dual methods for linear programming using restarts and
 * sharpness. Math. Program. 201, 133–184 (2023). https://doi.org/10.1007/s10107-022-01901-9.
 *
 * \tparam Vector The type of the vectors.
 * \tparam Real The type of the real numbers.
 * \param z The center of the trust region.
 * \param l_ The lower bounds.
 * \param u The upper bounds.
 * \param g_ The linear objective.
 * \param r The trust region radius.
 * \param z_hat The solution.
 * \return `true` if the has finite feasible solution, `false` otherwise.
 */
template< typename VectorView, typename Real_ >
bool
linearTrustRegion( const VectorView& z,
                   const VectorView& l_,
                   const VectorView& u,
                   const VectorView& g_,
                   const Real_& r,
                   VectorView& z_hat )
{
   using Real = typename VectorView::RealType;
   using Device = typename VectorView::DeviceType;
   using Index = typename VectorView::IndexType;
   using Vector = Containers::Vector< Real, Device, Index >;

   TNL_ASSERT_EQ( z.getSize(), l_.getSize(), "" );
   TNL_ASSERT_EQ( z.getSize(), g_.getSize(), "" );
   TNL_ASSERT_EQ( z.getSize(), z_hat.getSize(), "" );
   TNL_ASSERT_GE( r, 0, "" );
   if( g_ == 0 )
      return false;
   if( r == 0 ) {
      z_hat = z;
      return true;
   }
   if( ! all( greaterEqual( z, l_ - std::numeric_limits< Real >::round_error() ) )
       || ! all( lessEqual( z, u + std::numeric_limits< Real >::round_error() ) ) )
   {
      std::cout << "z is not in the feasible region" << std::endl;
      return false;
   }
   const Index N = z.getSize();
   Vector lambda_i( N );
   Vector l, g;
   l = l_;
   g = g_;
   linearTrustRegionNormalize( g, l, u );

   std::vector< Real > I;
   const auto z_view = z.getConstView();
   const auto l_view = l.getConstView();
   const auto g_view = g.getConstView();
   const auto lambda_i_view = lambda_i.getView();
   const auto z_hat_view = z_hat.getView();
   Real lambda_lo = 0;
   Real lambda_hi = std::numeric_limits< Real >::max();
   lambda_i.forAllElements(
      [ = ] __cuda_callable__( Index i, Real & value )
      {
         if( l_view[ i ] == -std::numeric_limits< Real >::infinity() || g_view[ i ] == 0 )
            value = std::numeric_limits< Real >::infinity();
         else
            value = ( z_view[ i ] - l_view[ i ] ) / g_view[ i ];
      } );
   Real f_lo = Algorithms::reduce< Device >( (Index) 0,
                                             N,
                                             [ = ] __cuda_callable__( Index i ) -> Real
                                             {
                                                return lambda_i_view[ i ] < lambda_lo ? pow( l_view[ i ] - z_view[ i ], 2 ) : 0;
                                             },
                                             TNL::Plus{} );
   Real f_hi = Algorithms::reduce< Device >( (Index) 0,
                                             N,
                                             [ = ] __cuda_callable__( Index i ) -> Real
                                             {
                                                return lambda_i_view[ i ] >= lambda_hi ? g_view[ i ] * g_view[ i ] : 0;
                                             },
                                             TNL::Plus{} );
   while( true ) {
      I.clear();
      for( Index i = 0; i < N; i++ )
         if( lambda_i_view[ i ] > lambda_lo && lambda_i_view[ i ] < lambda_hi ) {
            I.push_back( lambda_i_view[ i ] );
         }
      if( I.empty() ) {
         // solving r^2 = || z^hat(lambda_mid) - z ||^2 = f_lo + \lambda^2 f_hi
         Real lambda_mid;
         if( f_hi != 0 )
            lambda_mid = sqrt( ( r * r - f_lo ) / f_hi );
         else
            lambda_mid = Algorithms::reduce< Device >( (Index) 0,
                                                       N,
                                                       [ = ] __cuda_callable__( Index i ) -> Real
                                                       {
                                                          return lambda_i_view[ i ] < std::numeric_limits< Real >::infinity()
                                                                  ? lambda_i_view[ i ]
                                                                  : 0;
                                                       },
                                                       TNL::Max{} );
         z_hat = minimum( u, maximum( l_, z - lambda_mid * g_ ) );
         if( l2Norm( z_hat - z ) > r + std::numeric_limits< Real >::round_error() )
            return false;
         return true;
      }
      auto m = I.begin() + I.size() / 2;
      std::nth_element( I.begin(), m, I.end() );  // TODO: Implement nth_element in TNL

      const Real lambda_mid = I[ I.size() / 2 ];
      z_hat = maximum( z - lambda_mid * g, l );
      const Real f_mid =
         Algorithms::reduce< Device >( (Index) 0,
                                       N,
                                       [ = ] __cuda_callable__( Index i ) -> Real
                                       {
                                          return ( lambda_i_view[ i ] > lambda_lo && lambda_i_view[ i ] < lambda_hi )
                                                  ? pow( z_hat_view[ i ] - z_view[ i ], 2.0 )
                                                  : 0;
                                       },
                                       TNL::Plus{} )
         + f_lo + f_hi * lambda_mid * lambda_mid;  // f_mid = || z^hat(lambda_mid) - z ||^2
      if( f_mid <= r * r ) {
         f_lo += Algorithms::reduce< Device >( (Index) 0,
                                               N,
                                               [ = ] __cuda_callable__( Index i ) -> Real
                                               {
                                                  return ( lambda_i_view[ i ] > lambda_lo && lambda_i_view[ i ] <= lambda_mid )
                                                          ? pow( l_view[ i ] - z_view[ i ], 2 )
                                                          : 0;
                                               },
                                               TNL::Plus{} );
         lambda_lo = lambda_mid;
         if( f_lo > r * r ) {
            return false;
         }
      }
      else {
         f_hi += Algorithms::reduce< Device >( (Index) 0,
                                               N,
                                               [ = ] __cuda_callable__( Index i )
                                               {
                                                  return ( lambda_i_view[ i ] >= lambda_mid && lambda_i_view[ i ] < lambda_hi )
                                                          ? g_view[ i ] * g_view[ i ]
                                                          : 0;
                                               },
                                               TNL::Plus{} );
         lambda_hi = lambda_mid;
      }
   }
}

/**
 * \brief Normalize vectors g and l for the linear trust region method.
 *
 * In \ref TNL::Solvers::Optimization::linearTrustRegion, we assume that g >= 0.
 * This function normalizes the vectors g and l such that g >= 0. For each g_i <0,
 * we set g_i := abs( g_i ) and l_i = -u_i.
 *
 * \tparam Vector The type of the vectors.
 * \param g The linear objective.
 * \param l The lower bounds.
 * \param u The upper bounds.
 */
template< typename Vector, typename VectorView >
void
linearTrustRegionNormalize( Vector& g, Vector& l, const VectorView& u )
{
   using Real = typename Vector::RealType;
   using Index = typename Vector::IndexType;
   auto l_view = l.getView();
   auto u_view = u.getConstView();
   g.forAllElements(
      [ = ] __cuda_callable__( Index i, Real & value ) mutable
      {
         if( value < 0 ) {
            l_view[ i ] = -u_view[ i ];
            value = -value;
         }
      } );
}

}  // namespace TNL::Solvers::Optimization
