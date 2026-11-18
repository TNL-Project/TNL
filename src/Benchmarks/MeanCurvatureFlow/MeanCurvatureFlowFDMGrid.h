// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Math.h>
#include <TNL/Meshes/Grid.h>

/**
 * \brief Finite difference discretization of the mean-curvature flow on a grid of any dimension.
 *
 * The mean-curvature flow of a graph of the function \f$u\f$ is
 *
 * \f[
 *    \frac{\partial u}{\partial t} = Q \nabla \cdot \left( \frac{\nabla u}{Q} \right), \qquad Q = \sqrt{1 + |\nabla u|^2}.
 * \f]
 *
 * The values of \f$u\f$ are stored in the grid cells. In each interior cell \f$c\f$, the gradient norm is approximated
 * by the forward differences, \f$q_c = \sqrt{1 + \sum_i (\partial^+_i u)^2}\f$, and by the average of the forward and
 * backward differences, \f$\bar q_c = \sqrt{1 + \frac{1}{2} \sum_i ((\partial^+_i u)^2 + (\partial^-_i u)^2)}\f$.
 * The operator on the right-hand side is then approximated by
 *
 * \f[
 *    \bar q_c \sum_i \frac{1}{h_i^2} \left( \frac{u_{c + e_i} - u_c}{q_c} - \frac{u_c - u_{c - e_i}}{q_{c - e_i}} \right).
 * \f]
 *
 * The values on the boundary are fixed (Dirichlet boundary conditions).
 *
 * \tparam Grid is the type of the grid.
 */
template< typename Grid >
struct MeanCurvatureFlowFDMGrid
{
   static constexpr int Dimension = Grid::getMeshDimension();
   using RealType = typename Grid::RealType;
   using DeviceType = typename Grid::DeviceType;
   using IndexType = typename Grid::IndexType;
   using CoordinatesType = typename Grid::CoordinatesType;
   using PointType = typename Grid::PointType;
   using Cell = typename Grid::Cell;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   /**
    * \brief Returns the offset of the neighbor cell along the given axis.
    */
   __cuda_callable__
   static CoordinatesType
   getOffset( int axis, IndexType direction )
   {
      CoordinatesType offset( 0 );
      offset[ axis ] = direction;
      return offset;
   }

   /**
    * \brief Computes the approximations \f$q\f$ and \f$\bar q\f$ of the gradient norm in the interior cells.
    */
   static void
   computeGradientNorms( const Grid& grid, const VectorType& u, VectorType& q, VectorType& qBar )
   {
      const PointType hInv = 1.0 / grid.getSpaceSteps();
      const auto uView = u.getConstView();
      auto qView = q.getView();
      auto qBarView = qBar.getView();
      grid.template forInteriorEntities< Dimension >(
         [ = ] __cuda_callable__( const Cell& cell ) mutable
         {
            const IndexType c = cell.getIndex();
            const RealType u_c = uView[ c ];
            RealType forwardSum = 0;
            RealType backwardSum = 0;
            for( int i = 0; i < Dimension; i++ ) {
               const RealType forward = ( uView[ cell.getEntityIndex( getOffset( i, 1 ) ) ] - u_c ) * hInv[ i ];
               const RealType backward = ( u_c - uView[ cell.getEntityIndex( getOffset( i, -1 ) ) ] ) * hInv[ i ];
               forwardSum += forward * forward;
               backwardSum += backward * backward;
            }
            qView[ c ] = TNL::sqrt( 1.0 + forwardSum );
            qBarView[ c ] = TNL::sqrt( 1.0 + 0.5 * ( forwardSum + backwardSum ) );
         } );
   }

   /**
    * \brief Performs one step of the explicit Euler method: `uNew = u + tau * operator( u )`.
    */
   static void
   explicitStep(
      const Grid& grid,
      const VectorType& u,
      const VectorType& q,
      const VectorType& qBar,
      VectorType& uNew,
      RealType tau )
   {
      const PointType hSqrInv = 1.0 / ( grid.getSpaceSteps() * grid.getSpaceSteps() );
      const auto uView = u.getConstView();
      const auto qView = q.getConstView();
      const auto qBarView = qBar.getConstView();
      auto uNewView = uNew.getView();
      grid.template forInteriorEntities< Dimension >(
         [ = ] __cuda_callable__( const Cell& cell ) mutable
         {
            const IndexType c = cell.getIndex();
            const RealType u_c = uView[ c ];
            RealType divergence = 0;
            for( int i = 0; i < Dimension; i++ ) {
               const IndexType east = cell.getEntityIndex( getOffset( i, 1 ) );
               const IndexType west = cell.getEntityIndex( getOffset( i, -1 ) );
               divergence += ( ( uView[ east ] - u_c ) / qView[ c ] - ( u_c - uView[ west ] ) / qView[ west ] ) * hSqrInv[ i ];
            }
            uNewView[ c ] = u_c + tau * qBarView[ c ] * divergence;
         } );
   }

   /**
    * \brief Returns the time step, which is either given or proportional to the square of the smallest space step.
    */
   static RealType
   getTimeStep( const Grid& grid, RealType timeStep )
   {
      if( timeStep > 0 )
         return timeStep;
      const PointType& h = grid.getSpaceSteps();
      RealType hMin = h[ 0 ];
      for( int i = 1; i < Dimension; i++ )
         hMin = TNL::min( hMin, h[ i ] );
      return 0.1 * hMin * hMin;
   }

   /**
    * \brief Integrates the problem in time by the explicit Euler method.
    *
    * The vectors `q` and `qBar` must be initialized to one and `uNew` must have the same boundary values as `u`.
    */
   static void
   solveExplicitly(
      const Grid& grid,
      VectorType& u,
      VectorType& uNew,
      VectorType& q,
      VectorType& qBar,
      RealType timeStep,
      RealType finalTime,
      IndexType maxIterations )
   {
      const RealType tau = getTimeStep( grid, timeStep );
      RealType time = 0;
      IndexType iterations = 0;
      while( time < finalTime && ( maxIterations == 0 || iterations < maxIterations ) ) {
         computeGradientNorms( grid, u, q, qBar );
         explicitStep( grid, u, q, qBar, uNew, tau );
         u.swap( uNew );
         time += tau;
         iterations++;
      }
   }
};
