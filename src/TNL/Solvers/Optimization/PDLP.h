// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Solvers/IterativeSolver.h>

namespace TNL::Solvers::Optimization {

/**
 * \brief Implementation of Primal-Dual Hybrid Gradient Method for Linear Programming (PDLP).
 *
 * See the following paper for more details:
 *
 * Applegate, David, et al. "Practical large-scale linear programming using primal-dual hybrid gradient." Advances in Neural
 * Information Processing Systems 34 (2021): 20243-20257.
 *
 * https://proceedings.neurips.cc/paper/2021/file/a8fbbd3b11424ce032ba813493d95ad7-Paper.pdf
 *
 */
template< typename Vector,
          typename SolverMonitor = IterativeSolverMonitor< typename Vector::RealType, typename Vector::IndexType > >
class PDLP : public IterativeSolver< typename Vector::RealType, typename Vector::IndexType, SolverMonitor >
{
public:
   using RealType = typename Vector::RealType;
   using DeviceType = typename Vector::DeviceType;
   using IndexType = typename Vector::IndexType;
   using VectorType = Vector;
   using VectorView = typename Vector::ViewType;

   PDLP() = default;

   static void
   configSetup( Config::ConfigDescription& config, const std::string& prefix = "" );

   bool
   setup( const Config::ParameterContainer& parameters, const std::string& prefix = "" );

   void
   setRelaxation( const RealType& tau, const RealType& sigma )
   {
      this->tau = tau;
      this->sigma = sigma;
   }

   template< typename AMatrixType, typename GMatrixType >
   bool
   solve( const Vector& c,
          const GMatrixType& G,
          const Vector& h,
          const AMatrixType& A,
          const Vector& b,
          const Vector& l,
          const Vector& u,
          Vector& x )
   {
      const IndexType m1 = G.getRows();
      const IndexType m2 = A.getRows();
      const IndexType n = G.getColumns();
      TNL_ASSERT_EQ( c.getSize(), n, "" );
      TNL_ASSERT_EQ( h.getSize(), m1, "" );
      TNL_ASSERT_EQ( b.getSize(), m2, "" );
      TNL_ASSERT_EQ( l.getSize(), n, "" );
      TNL_ASSERT_EQ( u.getSize(), n, "" );
      TNL_ASSERT_EQ( x.getSize(), n, "" );
      AMatrixType AT;
      AT.getTransposition( A );
      GMatrixType GT;
      GT.getTransposition( G );

      VectorType y( m1 + m2, 0 ), last_x( n, 0 ), KT_y1( n ), KT_y2( n ), Kx( m1 + m2 );
      VectorView y1 = y.getView( 0, m1 );
      VectorView y2 = y.getView( m1, m1 + m2 );
      VectorView Kx_1 = Kx.getView( 0, m1 );
      VectorView Kx_2 = Kx.getView( m1, m1 + m2 );
      int iter = 0;
      while( iter++ < 100000 ) {  //this->nextIteration() ) {
         last_x = x;
         GT.vectorProduct( y1, KT_y1 );
         AT.vectorProduct( y2, KT_y2 );
         KT_y1 += KT_y2;
         x = minimum( u, maximum( l, x - tau * ( c - KT_y1 ) ) );
         last_x = 2 * x - last_x;
         G.vectorProduct( last_x, Kx_1 );
         A.vectorProduct( last_x, Kx_2 );
         y1 = maximum( 0, y1 + sigma * ( h - Kx_1 ) );
         y2 = y2 + sigma * ( b - Kx_2 );

         std::cout << this->getIterations() << " >>> x = " << x << std::endl;
      }
      return true;
   }

   template< typename GMatrixType >
   bool
   solve( const Vector& c, const GMatrixType& G, const Vector& h, const Vector& l, const Vector& u, Vector& x )
   {
      const IndexType m = G.getRows();
      const IndexType n = G.getColumns();
      TNL_ASSERT_EQ( c.getSize(), n, "" );
      TNL_ASSERT_EQ( h.getSize(), m, "" );
      TNL_ASSERT_EQ( l.getSize(), n, "" );
      TNL_ASSERT_EQ( u.getSize(), n, "" );
      TNL_ASSERT_EQ( x.getSize(), n, "" );
      GMatrixType GT;
      GT.getTransposition( G );

      std::cout << "m = " << m << ", n = " << n << std::endl;

      VectorType y( m, 0 ), last_x( n, 0 ), GT_y( n ), Gx( m );
      int iter = 0;
      while( true && iter++ < 10000 ) {  //this->nextIteration() ) {
         last_x = x;
         GT.vectorProduct( y, GT_y );
         x -= tau * ( c - GT_y );
         x = minimum( u, maximum( l, x ) );
         last_x = 2 * x - last_x;
         G.vectorProduct( last_x, Gx );
         y += sigma * ( h - Gx );
         y = maximum( 0, y );
         std::cout << this->getIterations() << " >>> x = " << x << std::endl;
      }
      return true;
   }

protected:
   RealType tau = 1.0, sigma = 1.0, epsilon = 1.0e-8;
};

}  // namespace TNL::Solvers::Optimization

#include <TNL/Solvers/Optimization/PDLP.hpp>
