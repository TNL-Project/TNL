// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once

#include <TNL/Meshes/Grid.h>
#include "MeanCurvatureFlowSolverBenchmark.h"

template< int Dimension,
          typename Real = double,
          typename Device = TNL::Devices::Host,
          typename Index = int >
struct MeanCurvatureFlowExplicitSolverBenchmarkFDMGrid;

template< typename Real,
          typename Device,
          typename Index >
struct MeanCurvatureFlowExplicitSolverBenchmarkFDMGrid< 1, Real, Device, Index >: public MeanCurvatureFlowSolverBenchmark< 1, Real, Device, Index >
{
   static constexpr int Dimension = 1;
   using BaseBenchmarkType = MeanCurvatureFlowSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using Grid = TNL::Meshes::Grid<1, Real, Device, int>;
   using Coordinates = typename Grid::CoordinatesType;

   TNL::String scheme() { return "fdm"; }

   void init( const Index xSize )
   {
      BaseBenchmarkType::init( xSize, ux, aux );
      this->grid.setSizes( xSize );
      this->grid.setDomain( { 0.0 }, { this->xDomainSize } );
      q.setSize( xSize );
      q_bar.setSize( xSize );
      q = 1.0;
      q_bar = 1.0;
   }

   bool writeGnuplot( const std::string &filename, const Index xSize ) const
   {
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize );
   }

   void exec( const Index xSize )
   {
      const Real hx = this->grid.getSpaceSteps()[ 0 ];
      const Real hx_inv = ( Real ) 1.0 / hx;
      const Real hx_sqr_inv = ( Real ) 1.0 / (hx * hx);

      Real start = 0;
      Index iterations = 0;
      auto timestep = this->timeStep ? this->timeStep : 0.1 * hx * hx;
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) )
      {
         auto uxView = this->ux.getView();
         auto auxView = this->aux.getView();
         auto q_view = this->q.getView();
         auto q_bar_view = this->q_bar.getView();
         this->grid.template forInteriorEntities<1>( [=] __cuda_callable__ ( const typename Grid::Cell& entity ) mutable {
            // c stands for center
            // w stands for west
            // e stands for east
            const Index c = entity.getIndex();
            const Real& u_c = uxView[ c ];
            const Real& u_w = uxView[ entity.getEntityIndex( Coordinates( -1 ) ) ];
            const Real& u_e = uxView[ entity.getEntityIndex( Coordinates(  1 ) ) ];
            Real u_x_f = ( u_e - u_c ) * hx_inv;
            Real u_x_b = ( u_c - u_w ) * hx_inv;
            q_bar_view[ c ] = sqrt( 1.0 + 0.5 * ( u_x_f * u_x_f + u_x_b * u_x_b ) );
            q_view[ c ] = sqrt( 1.0 + u_x_f * u_x_f );
         } );

         this->grid.template forInteriorEntities<1>( [=] __cuda_callable__( const typename Grid::Cell& entity ) mutable {
            const Index c = entity.getIndex();
            const Index w = entity.getEntityIndex( Coordinates( -1 ) );
            const Index e = entity.getEntityIndex( Coordinates(  1 ) );
            const Real& u_c = uxView[ c ];
            const Real& u_w = uxView[ w ];
            const Real& u_e = uxView[ e ];
            auxView[ c ] = u_c + q_bar_view[ c ] * ( ( u_e - u_c ) / q_view[ c ] - ( u_c - u_w ) / q_view[ w ] ) * hx_sqr_inv * timestep;
         } );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;

         if( iterations % 1  == 0 ) {
            std::cout << "iterations = " << iterations << " time = " << start << std::endl;
            /*std::cout <<  "u_x_f = " << u_x_f << std::endl;
            std::cout <<  "u_x_b = " << u_x_b << std::endl;
            std::cout << "q = " << q << std::endl;
            std::cout << "q_bar = " << q_bar << std::endl;*/
            //std::cout << "ux = " << ux << std::endl;
            //getchar();

            BaseBenchmarkType::writeGnuplot( "mcf-u-" + std::to_string( iterations ) + ".gplt", ux, xSize );
            //BaseBenchmarkType::writeGnuplot( "mcf-q-" + std::to_string( iterations ) + ".gplt", q, xSize );
            //BaseBenchmarkType::writeGnuplot( "mcf-q-bar-" + std::to_string( iterations ) + ".gplt", q_bar, xSize );

         }
      }
   }

protected:

   VectorType ux, aux, q, q_bar;
   Grid grid;
};

template< typename Real,
          typename Device,
          typename Index >
struct MeanCurvatureFlowExplicitSolverBenchmarkFDMGrid< 2, Real, Device, Index >: public MeanCurvatureFlowSolverBenchmark< 2, Real, Device, Index >
{
   static constexpr int Dimension = 2;
   using BaseBenchmarkType = MeanCurvatureFlowSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using Grid = TNL::Meshes::Grid<2, Real, Device, int>;
   using Coordinates = typename Grid::CoordinatesType;

   TNL::String scheme() { return "fdm"; }

   void init( const Index xSize, const Index ySize )
   {
      BaseBenchmarkType::init( xSize, ySize, ux, aux );
      this->grid.setSizes( {xSize, ySize} );
      this->grid.setDomain( { 0.0, 0.0}, { this->xDomainSize, this->yDomainSize } );
      q.setSize( this->grid.template getEntitiesCount<0>() );
      q_bar.setSize( this->grid.template getEntitiesCount<0>() );
      q = 1.0;
      q_bar = 1.0;
   }

   bool writeGnuplot( const std::string &filename, const Index xSize, const Index ySize ) const
   {
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize, ySize );
   }

   void exec( const Index xSize, const Index ySize )
   {
      const Real hx = grid.getSpaceSteps()[ 0 ];
      const Real hy = grid.getSpaceSteps()[ 1 ];
      const Real hx_inv = ( Real ) 1.0/( hx * hx );
      const Real hy_inv = ( Real ) 1.0/( hy * hy );
      const Real hx_sqr_inv = ( Real ) 1.0 / (hx * hx);
      const Real hy_sqr_inv = ( Real ) 1.0 / (hy * hy);

      BaseBenchmarkType::writeGnuplot( "initial-u.gplt", ux, xSize, ySize );

      Real start = 0;
      Index iterations = 0;
      auto timestep = this->timeStep ? this->timeStep : 0.1 * std::min(hx * hx, hy * hy);
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) )
     {
         auto uxView = this->ux.getView();
         auto auxView = this->aux.getView();
         auto q_view = this->q.getView();
         auto q_bar_view = this->q_bar.getView();
         this->grid.template forInteriorEntities<2>( [=] __cuda_callable__ ( const typename Grid::Cell& entity ) mutable {
            // c stands for center
            // w stands for west
            // e stands for east
            // s stands for south
            // n stands for north
            const Index c = entity.getIndex();
            const Real& u_c = uxView[ c ];
            const Real& u_w = uxView[ entity.getEntityIndex( Coordinates( -1,  0 ) ) ];
            const Real& u_e = uxView[ entity.getEntityIndex( Coordinates(  1,  0 ) ) ];
            const Real& u_s = uxView[ entity.getEntityIndex( Coordinates(  0, -1 ) ) ];
            const Real& u_n = uxView[ entity.getEntityIndex( Coordinates(  0,  1 ) ) ];
            Real u_x_f = ( u_e - u_c ) * hx_inv;
            Real u_x_b = ( u_c - u_w ) * hx_inv;
            Real u_y_f = ( u_n - u_c ) * hy_inv;
            Real u_y_b = ( u_c - u_s ) * hy_inv;
            q_bar_view[ c ] = sqrt( 1.0 + 0.5 * ( u_x_f * u_x_f + u_x_b * u_x_b + u_y_f * u_y_f + u_y_b * u_y_b) );
            q_view[ c ] = sqrt( 1.0 + u_x_f * u_x_f + u_y_f * u_y_f );
         } );

         this->grid.template forInteriorEntities<2>( [=] __cuda_callable__( const typename Grid::Cell& entity ) mutable {
            const Index c = entity.getIndex();
            const Index w = entity.getEntityIndex( Coordinates( -1,  0 ) );
            const Index e = entity.getEntityIndex( Coordinates(  1,  0 ) );
            const Index s = entity.getEntityIndex( Coordinates(  0, -1 ) );
            const Index n = entity.getEntityIndex( Coordinates(  0,  1 ) );
            const Real& u_c = uxView[ c ];
            const Real& u_w = uxView[ w ];
            const Real& u_e = uxView[ e ];
            const Real& u_s = uxView[ s ];
            const Real& u_n = uxView[ n ];
            auxView[ c ] = u_c + q_bar_view[ c ] * (
               ( ( u_e - u_c ) / q_view[ c ] - ( u_c - u_w ) / q_view[ w ] ) * hx_sqr_inv +
               ( ( u_n - u_c ) / q_view[ c ] - ( u_c - u_s ) / q_view[ s ] ) * hy_sqr_inv ) * timestep;
         } );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
         if( iterations % 1  == 0 ) {
            std::cout << "iterations = " << iterations << " time = " << start << std::endl;
            /*std::cout <<  "u_x_f = " << u_x_f << std::endl;
            std::cout <<  "u_x_b = " << u_x_b << std::endl;
            std::cout << "q = " << q << std::endl;
            std::cout << "q_bar = " << q_bar << std::endl;*/
            //std::cout << "ux = " << ux << std::endl;
            //getchar();

            BaseBenchmarkType::writeGnuplot( "mcf-u-" + std::to_string( iterations ) + ".gplt", ux, xSize, ySize );
            //BaseBenchmarkType::writeGnuplot( "mcf-q-" + std::to_string( iterations ) + ".gplt", q, xSize, ySize );
            //BaseBenchmarkType::writeGnuplot( "mcf-q-bar-" + std::to_string( iterations ) + ".gplt", q_bar, xSize, ySize );

         }
      }
   }

protected:

   VectorType ux, aux, q, q_bar;
   Grid grid;
};

template< typename Real,
          typename Device,
          typename Index >
struct MeanCurvatureFlowExplicitSolverBenchmarkFDMGrid< 3, Real, Device, Index >: public MeanCurvatureFlowSolverBenchmark< 3, Real, Device, Index >
{
   static constexpr int Dimension = 3;
   using BaseBenchmarkType = MeanCurvatureFlowSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using Grid = TNL::Meshes::Grid<3, Real, Device, int>;
   using Coordinates = typename Grid::CoordinatesType;

   TNL::String scheme() { return "fdm"; }

   void init( const Index xSize, const Index ySize, const Index zSize )
   {
      BaseBenchmarkType::init( xSize, ySize, zSize, ux, aux );
      grid.setSizes( {xSize, ySize, zSize} );
      grid.setDomain( { 0.0, 0.0, 0.0}, { this->xDomainSize, this->yDomainSize, this->zDomainSize } );
   }

   bool writeGnuplot( const std::string &filename, const Index xSize, const Index ySize, const Index zSize, const Index zSlice ) const
   {
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize, ySize, zSize, zSlice );
   }

   void exec( const Index xSize, const Index ySize, const Index zSize )
   {
      const Real hx = grid.getSpaceSteps()[ 0 ];
      const Real hy = grid.getSpaceSteps()[ 1 ];
      const Real hz = grid.getSpaceSteps()[ 2 ];
      const Real hx_inv = ( Real ) 1.0/( hx * hx );
      const Real hy_inv = ( Real ) 1.0/( hy * hy );
      const Real hz_inv = ( Real ) 1.0/( hz * hz );

      Real start = 0;
      Index iterations = 0;
      auto timestep = this->timeStep ? this->timeStep : 0.1 * std::min( hx*hx, std::min( hy*hy, hz*hz ) );
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) )
      {
         auto uxView = this->ux.getView();
         auto auxView = this->aux.getView();
         //auto xSize = grid.getSizes().x();
         //auto ySize = grid.getSizes().y();
         //auto xySize = xSize * ySize;

         auto next = [=] __cuda_callable__( const typename Grid::Cell& entity ) mutable {
            const Index centerIdx = entity.getIndex();
            const Real& element = uxView[centerIdx];
            auto center = ( Real ) 2.0 * element;



            auxView[ centerIdx ] = element + ( ( uxView[ entity.getEntityIndex( Coordinates( -1,  0,  0 ) ) ] -
                                                center +
                                                uxView[ entity.getEntityIndex( Coordinates(  1,  0,  0 ) ) ] ) * hx_inv +
                                              ( uxView[ entity.getEntityIndex( Coordinates(  0, -1,  0 ) ) ] -
                                                center +
                                                uxView[ entity.getEntityIndex( Coordinates(  0,  1,  0 ) ) ] ) * hy_inv  +
                                              ( uxView[ entity.getEntityIndex( Coordinates(  0,  0, -1 ) ) ] -
                                                center +
                                                uxView[ entity.getEntityIndex( Coordinates(  0,  0,  1 ) ) ] ) * hz_inv ) * timestep;
            /*auxView[ centerIdx ] = element + ( ( uxView[ centerIdx - 1 ] -         2.0 * center + uxView[ centerIdx + 1 ] ) * hx_inv +
                                                ( uxView[ centerIdx - xSize ] -     2.0 * center + uxView[ centerIdx + xSize ] ) * hy_inv +
                                                ( uxView[ centerIdx - xySize ] -     2.0 * center + uxView[ centerIdx + xySize ] ) * hy_inv
                                              ) * timestep;*/
         };
         grid.template forInteriorEntities<3>( next );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }

protected:

   VectorType ux, aux;
   Grid grid;
};
