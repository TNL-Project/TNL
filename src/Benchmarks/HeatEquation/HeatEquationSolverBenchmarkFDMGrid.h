// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/Grid.h>
#include "HeatEquationSolverBenchmark.h"

template< int Dimension, typename Real = double, typename Device = TNL::Devices::Host, typename Index = int >
struct HeatEquationSolverBenchmarkFDMGrid;

template< typename Real, typename Device, typename Index >
struct HeatEquationSolverBenchmarkFDMGrid< 1, Real, Device, Index >
: public HeatEquationSolverBenchmark< 1, Real, Device, Index >
{
   static constexpr int Dimension = 1;
   using BaseBenchmarkType = HeatEquationSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using Grid = TNL::Meshes::Grid< 1, Real, Device, int >;
   using Coordinates = typename Grid::CoordinatesType;

   TNL::String
   scheme()
   {
      return "fdm";
   }

   void
   init( const Index xSize )
   {
      BaseBenchmarkType::init( xSize, ux, aux );
      this->grid.setSizes( xSize );
      this->grid.setDomain( { 0.0 }, { this->xDomainSize } );
   }

   void
   getSolution( VectorType& u ) const
   {
      u = this->ux;
   }

   void
   exec( const Index xSize )
   {
      const Real hx = this->grid.getSpaceSteps()[ 0 ];
      const Real hx_inv = (Real) 1.0 / ( hx * hx );

      Real start = 0;
      Index iterations = 0;
      // the host does not wait for each kernel; Benchmark::time synchronizes the device after the time loop
      const auto launchConfiguration = TNL::Benchmarks::getAsyncLaunchConfiguration< Device >();
      auto timestep = this->timeStep ? this->timeStep : 0.1 * hx * hx;
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) ) {
         auto uxView = this->ux.getView();
         auto auxView = this->aux.getView();
         auto next = [ = ] __cuda_callable__( const typename Grid::Cell& entity ) mutable
         {
            const Index centerIdx = entity.getIndex();
            const Real& element = uxView[ centerIdx ];
            auto center = (Real) 2.0 * element;

            auxView[ centerIdx ] = element
                                 + ( ( uxView[ entity.getEntityIndex( Coordinates( -1 ) ) ] - center
                                       + uxView[ entity.getEntityIndex( Coordinates( 1 ) ) ] )
                                     * hx_inv )
                                      * timestep;
            /*auxView[ centerIdx ] = element + ( ( uxView[ centerIdx - 1 ] -
                                                2.0 * center +
                                                uxView[ centerIdx + 1 ] ) * hx_inv ) * timestep;*/
         };

         TNL::Meshes::forInteriorEntities< 1 >( this->grid, next, launchConfiguration );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }

protected:
   VectorType ux, aux;
   Grid grid;
};

template< typename Real, typename Device, typename Index >
struct HeatEquationSolverBenchmarkFDMGrid< 2, Real, Device, Index >
: public HeatEquationSolverBenchmark< 2, Real, Device, Index >
{
   static constexpr int Dimension = 2;
   using BaseBenchmarkType = HeatEquationSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using Grid = TNL::Meshes::Grid< 2, Real, Device, int >;
   using Coordinates = typename Grid::CoordinatesType;

   TNL::String
   scheme()
   {
      return "fdm";
   }

   void
   init( const Index xSize, const Index ySize )
   {
      BaseBenchmarkType::init( xSize, ySize, ux, aux );
      this->grid.setSizes( { xSize, ySize } );
      this->grid.setDomain( { 0.0, 0.0 }, { this->xDomainSize, this->yDomainSize } );
   }

   void
   getSolution( VectorType& u ) const
   {
      u = this->ux;
   }

   void
   exec( const Index xSize, const Index ySize )
   {
      const Real hx = grid.getSpaceSteps()[ 0 ];
      const Real hy = grid.getSpaceSteps()[ 1 ];
      const Real hx_inv = (Real) 1.0 / ( hx * hx );
      const Real hy_inv = (Real) 1.0 / ( hy * hy );

      Real start = 0;
      Index iterations = 0;
      // the host does not wait for each kernel; Benchmark::time synchronizes the device after the time loop
      const auto launchConfiguration = TNL::Benchmarks::getAsyncLaunchConfiguration< Device >();
      auto timestep = this->timeStep ? this->timeStep : 0.1 * std::min( hx * hx, hy * hy );
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) ) {
         auto uxView = this->ux.getView();
         auto auxView = this->aux.getView();
         //auto xSize = grid.getSizes().x();
         auto next = [ = ] __cuda_callable__( const typename Grid::Cell& entity ) mutable
         {
            const Index centerIdx = entity.getIndex();
            const Real& element = uxView[ centerIdx ];
            auto center = (Real) 2.0 * element;

            auxView[ centerIdx ] = element
                                 + ( ( uxView[ entity.template getAdjacentEntityIndex< 0, -1 >() ] - center
                                       + uxView[ entity.template getAdjacentEntityIndex< 0, 1 >() ] )
                                        * hx_inv
                                     + ( uxView[ entity.template getAdjacentEntityIndex< 1, -1 >() ] - center
                                         + uxView[ entity.template getAdjacentEntityIndex< 1, 1 >() ] )
                                          * hy_inv )
                                      * timestep;

            /*auxView[ centerIdx ] = element + ( ( uxView[ entity.getNeighbourEntityIndex( Coordinates( -1,  0 ) ) ] -
                                                center +
                                                uxView[ entity.getNeighbourEntityIndex( Coordinates(  1,  0 ) ) ] ) * hx_inv +
                                              ( uxView[ entity.getNeighbourEntityIndex( Coordinates(  0, -1 ) ) ] -
                                                center +
                                                uxView[ entity.getNeighbourEntityIndex( Coordinates(  0,  1 ) ) ] ) * hy_inv ) *
               timestep;*/
            /*auxView[ centerIdx ] = element + ( ( uxView[ centerIdx - 1 ] -
                                                2.0 * center +
                                                uxView[ centerIdx + 1 ] ) * hx_inv +
                                                ( uxView[ centerIdx - xSize ] -
                                                2.0 * center +
                                                uxView[ centerIdx + xSize ] ) * hy_inv ) * timestep;*/
         };
         TNL::Meshes::forInteriorEntities< 2 >( this->grid, next, launchConfiguration );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }

protected:
   VectorType ux, aux;
   Grid grid;
};

template< typename Real, typename Device, typename Index >
struct HeatEquationSolverBenchmarkFDMGrid< 3, Real, Device, Index >
: public HeatEquationSolverBenchmark< 3, Real, Device, Index >
{
   static constexpr int Dimension = 3;
   using BaseBenchmarkType = HeatEquationSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using Grid = TNL::Meshes::Grid< 3, Real, Device, int >;
   using Coordinates = typename Grid::CoordinatesType;

   TNL::String
   scheme()
   {
      return "fdm";
   }

   void
   init( const Index xSize, const Index ySize, const Index zSize )
   {
      BaseBenchmarkType::init( xSize, ySize, zSize, ux, aux );
      grid.setSizes( { xSize, ySize, zSize } );
      grid.setDomain( { 0.0, 0.0, 0.0 }, { this->xDomainSize, this->yDomainSize, this->zDomainSize } );
   }

   void
   getSolution( VectorType& u ) const
   {
      u = this->ux;
   }

   void
   exec( const Index xSize, const Index ySize, const Index zSize )
   {
      const Real hx = grid.getSpaceSteps()[ 0 ];
      const Real hy = grid.getSpaceSteps()[ 1 ];
      const Real hz = grid.getSpaceSteps()[ 2 ];
      const Real hx_inv = (Real) 1.0 / ( hx * hx );
      const Real hy_inv = (Real) 1.0 / ( hy * hy );
      const Real hz_inv = (Real) 1.0 / ( hz * hz );

      Real start = 0;
      Index iterations = 0;
      // the host does not wait for each kernel; Benchmark::time synchronizes the device after the time loop
      const auto launchConfiguration = TNL::Benchmarks::getAsyncLaunchConfiguration< Device >();
      auto timestep = this->timeStep ? this->timeStep : 0.1 * std::min( hx * hx, std::min( hy * hy, hz * hz ) );
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) ) {
         auto uxView = this->ux.getView();
         auto auxView = this->aux.getView();
         //auto xSize = grid.getSizes().x();
         //auto ySize = grid.getSizes().y();
         //auto xySize = xSize * ySize;

         auto next = [ = ] __cuda_callable__( const typename Grid::Cell& entity ) mutable
         {
            const Index centerIdx = entity.getIndex();
            const Real& element = uxView[ centerIdx ];
            auto center = (Real) 2.0 * element;

            auxView[ centerIdx ] = element
                                 + ( ( uxView[ entity.getEntityIndex( Coordinates( -1, 0, 0 ) ) ] - center
                                       + uxView[ entity.getEntityIndex( Coordinates( 1, 0, 0 ) ) ] )
                                        * hx_inv
                                     + ( uxView[ entity.getEntityIndex( Coordinates( 0, -1, 0 ) ) ] - center
                                         + uxView[ entity.getEntityIndex( Coordinates( 0, 1, 0 ) ) ] )
                                          * hy_inv
                                     + ( uxView[ entity.getEntityIndex( Coordinates( 0, 0, -1 ) ) ] - center
                                         + uxView[ entity.getEntityIndex( Coordinates( 0, 0, 1 ) ) ] )
                                          * hz_inv )
                                      * timestep;
            /*auxView[ centerIdx ] = element + ( ( uxView[ centerIdx - 1 ] -         2.0 * center + uxView[ centerIdx + 1 ] ) *
               hx_inv + ( uxView[ centerIdx - xSize ] -     2.0 * center + uxView[ centerIdx + xSize ] ) * hy_inv + ( uxView[
               centerIdx - xySize ] -     2.0 * center + uxView[ centerIdx + xySize ] ) * hy_inv ) * timestep;*/
         };
         TNL::Meshes::forInteriorEntities< 3 >( grid, next, launchConfiguration );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }

protected:
   VectorType ux, aux;
   Grid grid;
};
