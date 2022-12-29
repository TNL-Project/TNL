// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/Grid.h>
#include "HeatEquationSolverBenchmark.h"

template< int Dimension, typename Real = double, typename Device = TNL::Devices::Host, typename Index = int >
struct HeatEquationSolverBenchmarkGrid;

template< typename Real, typename Device, typename Index >
struct HeatEquationSolverBenchmarkGrid< 1, Real, Device, Index > : public HeatEquationSolverBenchmark< 1, Real, Device, Index >
{
   static constexpr int Dimension = 1;
   using BaseBenchmarkType = HeatEquationSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using Grid = TNL::Meshes::Grid< 1, Real, Device, int >;
   using Coordinates = typename Grid::CoordinatesType;

   void
   init( const Index xSize )
   {
      BaseBenchmarkType::init( xSize, ux, aux );
      this->grid.setDimensions( xSize );
      this->grid.setDomain( { 0.0 }, { this->xDomainSize } );
   }

   bool
   writeGnuplot( const std::string& filename, const Index xSize ) const
   {
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize );
   }

   void
   exec( const Index xSize )
   {
      const Real hx = this->grid.template getSpaceStepsProducts< 1 >();
      const Real hx_inv = this->grid.template getSpaceStepsProducts< -2 >();

      TNL_ASSERT_EQ( hx, this->xDomainSize / (Real) xSize, "computed wrong hx on the grid" );

      Real start = 0;
      Index iterations = 0;
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
                                 + ( ( uxView[ entity.getNeighbourEntityIndex( Coordinates( -1 ) ) ] - center
                                       + uxView[ entity.getNeighbourEntityIndex( Coordinates( 1 ) ) ] )
                                     * hx_inv )
                                      * timestep;
            /*auxView[ centerIdx ] = element + ( ( uxView[ centerIdx - 1 ] -
                                                2.0 * center +
                                                uxView[ centerIdx + 1 ] ) * hx_inv ) * timestep;*/
         };

         this->grid.template forInteriorEntities< 1 >( next );
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
struct HeatEquationSolverBenchmarkGrid< 2, Real, Device, Index > : public HeatEquationSolverBenchmark< 2, Real, Device, Index >
{
   static constexpr int Dimension = 2;
   using BaseBenchmarkType = HeatEquationSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using Grid = TNL::Meshes::Grid< 2, Real, Device, int >;
   using Coordinates = typename Grid::CoordinatesType;

   void
   init( const Index xSize, const Index ySize )
   {
      BaseBenchmarkType::init( xSize, ySize, ux, aux );
      this->grid.setDimensions( xSize, ySize );
      this->grid.setDomain( { 0.0, 0.0 }, { this->xDomainSize, this->yDomainSize } );
   }

   bool
   writeGnuplot( const std::string& filename, const Index xSize, const Index ySize ) const
   {
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize, ySize );
   }

   void
   exec( const Index xSize, const Index ySize )
   {
      const Real hx = grid.template getSpaceStepsProducts< 1, 0 >();
      const Real hy = grid.template getSpaceStepsProducts< 0, 1 >();
      const Real hx_inv = grid.template getSpaceStepsProducts< -2, 0 >();
      const Real hy_inv = grid.template getSpaceStepsProducts< 0, -2 >();

      TNL_ASSERT_EQ( hx, this->xDomainSize / (Real) xSize, "computed wrong hx on the grid" );
      TNL_ASSERT_EQ( hy, this->yDomainSize / (Real) ySize, "computed wrong hy on the grid" );

      Real start = 0;
      Index iterations = 0;
      auto timestep = this->timeStep ? this->timeStep : 0.1 * std::min( hx * hx, hy * hy );
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) ) {
         auto uxView = this->ux.getView();
         auto auxView = this->aux.getView();
         //auto xSize = grid.getDimensions().x();
         auto next = [ = ] __cuda_callable__( const typename Grid::Cell& entity ) mutable
         {
            const Index centerIdx = entity.getIndex();
            const Real& element = uxView[ centerIdx ];
            auto center = (Real) 2.0 * element;

            auxView[ centerIdx ] = element
                                 + ( ( uxView[ entity.getNeighbourEntityIndex( Coordinates( -1, 0 ) ) ] - center
                                       + uxView[ entity.getNeighbourEntityIndex( Coordinates( 1, 0 ) ) ] )
                                        * hx_inv
                                     + ( uxView[ entity.getNeighbourEntityIndex( Coordinates( 0, -1 ) ) ] - center
                                         + uxView[ entity.getNeighbourEntityIndex( Coordinates( 0, 1 ) ) ] )
                                          * hy_inv )
                                      * timestep;
            /*auxView[ centerIdx ] = element + ( ( uxView[ centerIdx - 1 ] -
                                                2.0 * center +
                                                uxView[ centerIdx + 1 ] ) * hx_inv +
                                                ( uxView[ centerIdx - xSize ] -
                                                2.0 * center +
                                                uxView[ centerIdx + xSize ] ) * hy_inv ) * timestep;*/
         };
         this->grid.template forInteriorEntities< 2 >( next );
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
struct HeatEquationSolverBenchmarkGrid< 3, Real, Device, Index > : public HeatEquationSolverBenchmark< 3, Real, Device, Index >
{
   static constexpr int Dimension = 3;
   using BaseBenchmarkType = HeatEquationSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using Grid = TNL::Meshes::Grid< 3, Real, Device, int >;
   using Coordinates = typename Grid::CoordinatesType;

   void
   init( const Index xSize, const Index ySize, const Index zSize )
   {
      BaseBenchmarkType::init( xSize, ySize, zSize, ux, aux );
      grid.setDimensions( xSize, ySize, zSize );
      grid.setDomain( { 0.0, 0.0, 0.0 }, { this->xDomainSize, this->yDomainSize, this->zDomainSize } );
   }

   void
   exec( const Index xSize, const Index ySize, const Index zSize )
   {
      const Real hx = grid.template getSpaceStepsProducts< 1, 0, 0 >();
      const Real hy = grid.template getSpaceStepsProducts< 0, 1, 0 >();
      const Real hz = grid.template getSpaceStepsProducts< 0, 0, 1 >();
      const Real hx_inv = grid.template getSpaceStepsProducts< -2, 0, 0 >();
      const Real hy_inv = grid.template getSpaceStepsProducts< 0, -2, 0 >();
      const Real hz_inv = grid.template getSpaceStepsProducts< 0, 0, -2 >();

      TNL_ASSERT_EQ( hx, this->xDomainSize / (Real) xSize, "computed wrong hx on the grid" );
      TNL_ASSERT_EQ( hy, this->yDomainSize / (Real) ySize, "computed wrong hy on the grid" );
      TNL_ASSERT_EQ( hz, this->zDomainSize / (Real) zSize, "computed wrong hz on the grid" );

      Real start = 0;
      Index iterations = 0;
      auto timestep = this->timeStep ? this->timeStep : 0.1 * std::min( hx * hx, std::min( hy * hy, hz * hz ) );
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) ) {
         auto uxView = this->ux.getView();
         auto auxView = this->aux.getView();
         //auto xSize = grid.getDimensions().x();
         //auto ySize = grid.getDimensions().y();
         //auto xySize = xSize * ySize;

         auto next = [ = ] __cuda_callable__( const typename Grid::Cell& entity ) mutable
         {
            const Index centerIdx = entity.getIndex();
            const Real& element = uxView[ centerIdx ];
            auto center = (Real) 2.0 * element;

            auxView[ centerIdx ] = element
                                 + ( ( uxView[ entity.getNeighbourEntityIndex( Coordinates( -1, 0, 0 ) ) ] - center
                                       + uxView[ entity.getNeighbourEntityIndex( Coordinates( 1, 0, 0 ) ) ] )
                                        * hx_inv
                                     + ( uxView[ entity.getNeighbourEntityIndex( Coordinates( 0, -1, 0 ) ) ] - center
                                         + uxView[ entity.getNeighbourEntityIndex( Coordinates( 0, 1, 0 ) ) ] )
                                          * hy_inv
                                     + ( uxView[ entity.getNeighbourEntityIndex( Coordinates( 0, 0, -1 ) ) ] - center
                                         + uxView[ entity.getNeighbourEntityIndex( Coordinates( 0, 0, 1 ) ) ] )
                                          * hz_inv )
                                      * timestep;
            /*auxView[ centerIdx ] = element + ( ( uxView[ centerIdx - 1 ] -         2.0 * center + uxView[ centerIdx + 1 ] ) *
               hx_inv + ( uxView[ centerIdx - xSize ] -     2.0 * center + uxView[ centerIdx + xSize ] ) * hy_inv + ( uxView[
               centerIdx - xySize ] -     2.0 * center + uxView[ centerIdx + xySize ] ) * hy_inv ) * timestep;*/
         };
         grid.template forInteriorEntities< 3 >( next );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }

protected:
   VectorType ux, aux;
   Grid grid;
};
