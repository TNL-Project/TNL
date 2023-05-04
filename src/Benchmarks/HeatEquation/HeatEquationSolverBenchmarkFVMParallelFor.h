// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber

#pragma once

#include "HeatEquationSolverBenchmark.h"

template< int Dimension,
          typename Real = double,
          typename Device = TNL::Devices::Host,
          typename Index = int >
struct HeatEquationSolverBenchmarkFVMParallelFor;

template< typename Real,
          typename Device,
          typename Index >
struct HeatEquationSolverBenchmarkFVMParallelFor< 1, Real, Device, Index > : public HeatEquationSolverBenchmark< 1, Real, Device, Index >
{
   static constexpr int Dimension = 1;
   using BaseBenchmarkType = HeatEquationSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;

   TNL::String scheme() { return "fvm"; }

   void init( const Index xSize )
   {
      BaseBenchmarkType::init( xSize, ux, aux );
      faces.setSize( xSize+1 );
   }

   bool writeGnuplot( const std::string &filename, const Index xSize ) const
   {
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize );
   }

   void exec( const Index xSize )
   {
      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hx_inv = 1.0 / hx;

      Real start = 0;
      Index iterations = 0;
      auto timestep = this->timeStep ? this->timeStep : 0.1 * hx*hx;
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) )
      {
         auto ux_view = this->ux.getView();
         auto aux_view = this->aux.getView();
         auto faces_view = this->faces.getView();

         auto gradients = [=] __cuda_callable__ ( Index i ) mutable {
            faces_view[ i ] = ( ux_view[ i ] - ux_view[ i-1 ] ) * hx_inv;
         };
         TNL::Algorithms::parallelFor< Device >( 1, xSize, gradients );

         auto update = [=] __cuda_callable__( Index i ) mutable
         {
            aux_view[ i ] = ux_view[ i ] + timestep * ( faces_view[ i+1 ] - faces_view[ i ] ) * hx_inv;
         };
         TNL::Algorithms::parallelFor< Device >( 1, xSize - 1, update );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }

protected:

   VectorType ux, aux, faces;
};

template< typename Real,
          typename Device,
          typename Index >
struct HeatEquationSolverBenchmarkFVMParallelFor< 2, Real, Device, Index > : public HeatEquationSolverBenchmark< 2, Real, Device, Index >
{
   static constexpr int Dimension = 2;
   using BaseBenchmarkType = HeatEquationSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using CoordinatesType = TNL::Containers::StaticArray< Dimension, Index >;

   TNL::String scheme() { return "fvm"; }

   void init( const Index xSize, const Index ySize )
   {
      BaseBenchmarkType::init( xSize, ySize, ux, aux );
      x_faces.setSize( xSize * ( ySize + 1 ) );
      y_faces.setSize( ( xSize + 1 ) * ySize );
      x_faces = 0.0;
      y_faces = 0.0;
   }

   bool writeGnuplot( const std::string &filename, const Index xSize, const Index ySize ) const
   {
      return BaseBenchmarkType::writeGnuplot( filename, this->ux, xSize, ySize );
   }

   bool writeGnuplot( const std::string &filename, const Index xSize, const Index ySize, const VectorType& u ) const
   {
      return BaseBenchmarkType::writeGnuplot( filename, u, xSize, ySize );
   }

   void exec( const Index xSize, const Index ySize )
   {
      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;
      const Real hx_inv = 1.0 / hx;
      const Real hy_inv = 1.0 / hy;

      Real start = 0;
      Index iterations = 0;
      auto timestep = this->timeStep ? this->timeStep : 0.1 * std::min(hx * hx, hy * hy);
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) )
      {
         auto ux_view = this->ux.getView();
         auto aux_view = this->aux.getView();
         auto x_faces_view = this->x_faces.getView();
         auto y_faces_view = this->y_faces.getView();

         /////
         // First we iterate over vertical faces of interior cells. Their coordinates and
         // indexes in case of 2D 4x4 mesh are depicted on the following figure:
         //
         //   +-------+-------+-------+-------+       +-------+-------+-------+-------+
         //   |       |       |       |       |       |       |       |       |       |
         //   |       |       |       |       |       |       |       |       |       |
         //   |       |       |       |       |       |       |       |       |       |
         //   +-------+-------+-------+-------+       +-------+-------+-------+-------+
         //   |       |       |       |       |       |       |       |       |       |
         //   |     (1,2)   (2,2)   (3,2)     |       |     (11)    (12)    (13)      |
         //   |       |       |       |       |       |       |       |       |       |
         //   +-------+-------+-------+-------+       +-------+-------+-------+-------+
         //   |       |       |       |       |       |       |       |       |       |
         //   |     (1,1)   (2,1)   (3,1)     |       |     ( 6)    ( 7)    ( 8)      |
         //   |       |       |       |       |       |       |       |       |       |
         //   +-------+-------+-------+-------+       +-------+-------+-------+-------+
         //   |       |       |       |       |       |       |       |       |       |
         //   |       |       |       |       |       |       |       |       |       |
         //   |       |       |       |       |       |       |       |       |       |
         //   +-------+-------+-------+-------+       +-------+-------+-------+-------+
         //
         // Their indexes are given as:
         //
         //    face_idx =  row * ( xSize + 1 ) + column
         //
         // where column = 1 ... xSize-1 and row = 1 ... ySize-2 and
         auto x_gradients = [=] __cuda_callable__ ( const CoordinatesType& idx ) mutable {
            const Index& column = idx[ 0 ];
            const Index& row = idx[ 1 ];
            const Index face_idx = row * ( xSize + 1 ) + column;
            const Index cell_idx = row * xSize + column;
            y_faces_view[ face_idx ] = ( ux_view[ cell_idx ] - ux_view[ cell_idx - 1 ] ) * hx_inv;
         };
         TNL::Algorithms::parallelFor< Device >( CoordinatesType{ 1, 1 }, CoordinatesType{ xSize, ySize - 1 }, x_gradients );

         /////
         // Next we iterate over horizontal faces of interior cells. Their coordinates and
         // indexes in case of 2D 4x4 mesh are depicted on the following figure:
         //
         //   +-------+-------+-------+-------+       +-------+-------+-------+-------+
         //   |       |       |       |       |       |       |       |       |       |
         //   |       |       |       |       |       |       |       |       |       |
         //   |       |       |       |       |       |       |       |       |       |
         //   +-------+-(1,3)-+-(2,3)-+-------+       +-------+-( 13)-+-( 14)-+-------+
         //   |       |       |       |       |       |       |       |       |       |
         //   |       |       |       |       |       |       |       |       |       |
         //   |       |       |       |       |       |       |       |       |       |
         //   +-------+-(1,2)-+-(2,2)-+-------+       +-------+-(  9)-+-( 10)-+-------+
         //   |       |       |       |       |       |       |       |       |       |
         //   |       |       |       |       |       |       |       |       |       |
         //   |       |       |       |       |       |       |       |       |       |
         //   +-------+-(1,1)-+-(2,1)-+-------+       +-------+-(  5)-+-(  6)-+-------+
         //   |       |       |       |       |       |       |       |       |       |
         //   |       |       |       |       |       |       |       |       |       |
         //   |       |       |       |       |       |       |       |       |       |
         //   +-------+-------+-------+-------+       +-------+-------+-------+-------+
         //
         // Their indexes are given as:
         //
         //    face_idx =  row * xSize + column
         // where column = 1 ... xSize-2 and row = 1 ... ySize-1
         auto y_gradients = [=] __cuda_callable__ ( const CoordinatesType& idx ) mutable {
            const Index& column = idx[ 0 ];
            const Index& row = idx[ 1 ];
            //const Index face_idx = row * xSize + column;
            //const Index cell_idx = row * xSize + column;
            const Index i = row * xSize + column;
            x_faces_view[ i ] = ( ux_view[ i ] - ux_view[ i - xSize ] ) * hy_inv;
         };
         TNL::Algorithms::parallelFor< Device >( CoordinatesType{ 1, 1 }, CoordinatesType{ xSize - 1, ySize }, y_gradients );

         ////
         // From the first derivatives stored on the faces, we will now compute the laplacian:
         //
         //   +-----------+-----------+-----------+-----------+
         //   |           |           |           |           |
         //   |           |           |           |           |
         //   |    BC     |    BC     |   BC      |    BC     |
         //   |           |           |           |           |
         //   |           |           |           |           |
         //   +-----------+---(1,3)---+---(2,3)---+-----------+
         //   |           |           |           |           |
         //   |           |           |           |           |
         //   |    BC   (1,2) [1,2] (2,2) [2,2] (3,2)  BC     |
         //   |           |           |           |           |
         //   |           |           |           |           |
         //   +-----------+---(1,2)---+---(2,2)---+-----------+
         //   |           |           |           |           |
         //   |           |           |           |           |
         //   |    BC   (1,1) [1,1] (2,1) [2,1] (3,1)  BC     |
         //   |           |           |           |           |
         //   |           |           |           |           |
         //   +-----------+---(1,1)---+---(2,1)---+-----------+
         //   |           |           |           |           |
         //   |           |           |           |           |
         //   |    BC     |    BC     |    BC     |    BC     |
         //   |           |           |           |           |
         //   |           |           |           |           |
         //   +-----------+-----------+-----------+-----------+
         //
         // BC denotes cells where we set boundary conditions and therefore we iterate only over the
         // interior cells for which column = 1 ... xSize-2 and row = 1 ... ySize-2.
         auto update = [=] __cuda_callable__( const CoordinatesType& idx ) mutable
         {
            const Index& column = idx[ 0 ];
            const Index& row = idx[ 1 ];
            const Index y_face_idx = row * ( xSize + 1 ) + column;
            auto index = row * xSize + column;
            aux_view[index] = ux_view[ index ] + timestep * (
               ( y_faces_view[ y_face_idx + 1 ] - y_faces_view[ y_face_idx ] ) * hx_inv +
               ( x_faces_view[ index + xSize ]  - x_faces_view[ index ]      ) * hy_inv );
         };
         TNL::Algorithms::parallelFor< Device >( CoordinatesType{ 1, 1 }, CoordinatesType{ xSize - 1, ySize - 1 }, update );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }

protected:

   VectorType ux, aux, x_faces, y_faces;
};

template< typename Real,
          typename Device,
          typename Index >
struct HeatEquationSolverBenchmarkFVMParallelFor< 3, Real, Device, Index > : public HeatEquationSolverBenchmark< 3, Real, Device, Index >
{
   static constexpr int Dimension = 3;
   using BaseBenchmarkType = HeatEquationSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using CoordinatesType = TNL::Containers::StaticArray< Dimension, Index >;

   TNL::String scheme() { return "fvm"; }

   void init( const Index xSize, const Index ySize, const Index zSize )
   {
      BaseBenchmarkType::init( xSize, ySize, zSize, ux, aux );
      yz_faces.setSize( ( xSize + 1 ) * ySize * zSize );
      xz_faces.setSize( xSize * ( ySize + 1 ) * zSize );
      xy_faces.setSize( xSize * ySize * ( zSize + 1 ) );
      yz_faces = 0.0;
      xz_faces = 0.0;
      xy_faces = 0.0;
   }

   bool writeGnuplot( const std::string &filename, const Index xSize, const Index ySize, const Index zSize, const Index zSlice ) const
   {
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize, ySize, zSize, zSlice );
   }

   void exec( const Index xSize, const Index ySize, const Index zSize )
   {
      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;
      const Real hz = this->zDomainSize / (Real) zSize;
      const Real hx_inv = 1.0 / hx;
      const Real hy_inv = 1.0 / hy;
      const Real hz_inv = 1.0 / hz;
      const auto xySize = xSize * ySize;

      Real start = 0;
      Index iterations = 0;
      auto timestep = this->timeStep ? this->timeStep : 0.1 * std::min( hx*hx, std::min( hy*hy, hz*hz ) );
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) )
      {
         auto ux_view = this->ux.getView();
         auto aux_view = this->aux.getView();
         auto yz_faces_view = this->yz_faces.getView();
         auto xz_faces_view = this->xz_faces.getView();
         auto xy_faces_view = this->xy_faces.getView();

         auto x_gradients = [=] __cuda_callable__ ( const CoordinatesType& idx ) mutable {
            const Index& i = idx[0];
            const Index& j = idx[1];
            const Index& k = idx[2];
            const Index face_idx = ( k * ySize + j ) * ( xSize + 1 ) + i;
            const Index cell_idx = ( k * ySize + j ) * xSize + i;
            yz_faces_view[ face_idx ] = ( ux_view[ cell_idx ] - ux_view[ cell_idx - 1 ] ) * hx_inv;
         };
         TNL::Algorithms::parallelFor< Device >( CoordinatesType{ 1, 1, 1 }, CoordinatesType{ xSize, ySize - 1, zSize - 1 }, x_gradients );

         auto y_gradients = [=] __cuda_callable__ ( const CoordinatesType& idx ) mutable {
            const Index& i = idx[0];
            const Index& j = idx[1];
            const Index& k = idx[2];
            const Index face_idx = ( k * ( ySize+1 ) + j ) * xSize + i;
            const Index cell_idx = ( k * ySize + j ) * xSize + i;
            xz_faces_view[ face_idx ] = ( ux_view[ cell_idx ] - ux_view[ cell_idx - xSize ] ) * hy_inv;
         };
         TNL::Algorithms::parallelFor< Device >( CoordinatesType{ 1, 1, 1 }, CoordinatesType{ xSize - 1, ySize, zSize - 1 }, y_gradients );

         auto z_gradients = [=] __cuda_callable__ ( const CoordinatesType& idx ) mutable {
            const Index& i = idx[0];
            const Index& j = idx[1];
            const Index& k = idx[2];
            const Index face_idx = ( k * ySize + j ) * xSize + i;
            const Index cell_idx = ( k * ySize + j ) * xSize + i;
            xy_faces_view[ face_idx ] = ( ux_view[ cell_idx ] - ux_view[ cell_idx - xySize ] ) * hz_inv;
         };
         TNL::Algorithms::parallelFor< Device >( CoordinatesType{ 1, 1, 1 }, CoordinatesType{ xSize - 1, ySize - 1, zSize }, z_gradients );

         auto update = [=] __cuda_callable__( const CoordinatesType& idx ) mutable
         {
            const Index& i = idx[0];
            const Index& j = idx[1];
            const Index& k = idx[2];
            const Index yz_face_idx = ( k * ySize + j ) * ( xSize + 1 ) + i;
            const Index xz_face_idx = ( k * ( ySize+1 ) + j ) * xSize + i;
            const Index xy_face_idx = ( k * ySize + j ) * xSize + i;
            const Index cell_idx = ( k * ySize + j ) * xSize + i;

            aux_view[ cell_idx ] = ux_view[ cell_idx ] + timestep * (
               ( yz_faces_view[ yz_face_idx + 1 ]       - yz_faces_view[ yz_face_idx ] ) * hx_inv +
               ( xz_faces_view[ xz_face_idx + xSize ] - xz_faces_view[ xz_face_idx ] ) * hy_inv +
               ( xy_faces_view[ xy_face_idx + xySize ] - xy_faces_view[ xy_face_idx ] ) * hz_inv );
         };
         TNL::Algorithms::parallelFor< Device >( CoordinatesType{ 1, 1, 1 }, CoordinatesType{ xSize - 1, ySize - 1, zSize - 1 }, update );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }

protected:

   VectorType ux, aux, yz_faces, xz_faces, xy_faces;
};
