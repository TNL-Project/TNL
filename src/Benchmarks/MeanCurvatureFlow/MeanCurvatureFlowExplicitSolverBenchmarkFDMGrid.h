// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/Grid.h>
#include "MeanCurvatureFlowFDMGrid.h"
#include "MeanCurvatureFlowSolverBenchmark.h"

/**
 * \brief Common part of the explicit mean-curvature flow solvers for all grid dimensions.
 *
 * See \ref MeanCurvatureFlowFDMGrid for the description of the scheme.
 */
template< typename Grid >
struct MeanCurvatureFlowExplicitSolver
{
   using Scheme = MeanCurvatureFlowFDMGrid< Grid >;
   using RealType = typename Grid::RealType;
   using IndexType = typename Grid::IndexType;
   using VectorType = typename Scheme::VectorType;

   void
   init( const VectorType& u )
   {
      // the auxiliary vector must have the same boundary values as the solution
      uNew = u;
      q.setSize( u.getSize() );
      qBar.setSize( u.getSize() );
      q = 1.0;
      qBar = 1.0;
   }

   void
   solve( const Grid& grid, VectorType& u, RealType timeStep, RealType finalTime, IndexType maxIterations )
   {
      Scheme::solveExplicitly( grid, u, uNew, q, qBar, timeStep, finalTime, maxIterations );
   }

protected:
   VectorType uNew, q, qBar;
};

/**
 * \brief Mean-curvature flow solved by the finite difference method on the grid with the explicit Euler method.
 */
template< int Dimension, typename Real = double, typename Device = TNL::Devices::Host, typename Index = int >
struct MeanCurvatureFlowExplicitSolverBenchmarkFDMGrid;

template< typename Real, typename Device, typename Index >
struct MeanCurvatureFlowExplicitSolverBenchmarkFDMGrid< 1, Real, Device, Index >
: public MeanCurvatureFlowSolverBenchmark< 1, Real, Device, Index >
{
   static constexpr int Dimension = 1;
   using BaseBenchmarkType = MeanCurvatureFlowSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using Grid = TNL::Meshes::Grid< Dimension, Real, Device, Index >;

   TNL::String
   scheme() override
   {
      return "fdm";
   }

   void
   init( const Index xSize ) override
   {
      BaseBenchmarkType::init( xSize, ux, aux );
      grid.setSizes( { xSize } );
      grid.setDomain( { 0.0 }, { this->xDomainSize } );
      solver.init( ux );
   }

   bool
   writeGnuplot( const std::string& filename, const Index xSize ) const override
   {
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize );
   }

   void
   exec( const Index xSize ) override
   {
      solver.solve( grid, ux, this->timeStep, this->finalTime, this->maxIterations );
   }

protected:
   VectorType ux, aux;
   Grid grid;
   MeanCurvatureFlowExplicitSolver< Grid > solver;
};

template< typename Real, typename Device, typename Index >
struct MeanCurvatureFlowExplicitSolverBenchmarkFDMGrid< 2, Real, Device, Index >
: public MeanCurvatureFlowSolverBenchmark< 2, Real, Device, Index >
{
   static constexpr int Dimension = 2;
   using BaseBenchmarkType = MeanCurvatureFlowSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using Grid = TNL::Meshes::Grid< Dimension, Real, Device, Index >;

   TNL::String
   scheme() override
   {
      return "fdm";
   }

   void
   init( const Index xSize, const Index ySize ) override
   {
      BaseBenchmarkType::init( xSize, ySize, ux, aux );
      grid.setSizes( { xSize, ySize } );
      grid.setDomain( { 0.0, 0.0 }, { this->xDomainSize, this->yDomainSize } );
      solver.init( ux );
   }

   bool
   writeGnuplot( const std::string& filename, const Index xSize, const Index ySize ) const override
   {
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize, ySize );
   }

   void
   exec( const Index xSize, const Index ySize ) override
   {
      solver.solve( grid, ux, this->timeStep, this->finalTime, this->maxIterations );
   }

protected:
   VectorType ux, aux;
   Grid grid;
   MeanCurvatureFlowExplicitSolver< Grid > solver;
};

template< typename Real, typename Device, typename Index >
struct MeanCurvatureFlowExplicitSolverBenchmarkFDMGrid< 3, Real, Device, Index >
: public MeanCurvatureFlowSolverBenchmark< 3, Real, Device, Index >
{
   static constexpr int Dimension = 3;
   using BaseBenchmarkType = MeanCurvatureFlowSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using Grid = TNL::Meshes::Grid< Dimension, Real, Device, Index >;

   TNL::String
   scheme() override
   {
      return "fdm";
   }

   void
   init( const Index xSize, const Index ySize, const Index zSize ) override
   {
      BaseBenchmarkType::init( xSize, ySize, zSize, ux, aux );
      grid.setSizes( { xSize, ySize, zSize } );
      grid.setDomain( { 0.0, 0.0, 0.0 }, { this->xDomainSize, this->yDomainSize, this->zDomainSize } );
      solver.init( ux );
   }

   bool
   writeGnuplot( const std::string& filename, const Index xSize, const Index ySize, const Index zSize, const Index zSlice )
      const override
   {
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize, ySize, zSize, zSlice );
   }

   void
   exec( const Index xSize, const Index ySize, const Index zSize ) override
   {
      solver.solve( grid, ux, this->timeStep, this->finalTime, this->maxIterations );
   }

protected:
   VectorType ux, aux;
   Grid grid;
   MeanCurvatureFlowExplicitSolver< Grid > solver;
};
