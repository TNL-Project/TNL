// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/Grid.h>
#include "HeatEquationSolverBenchmark.h"

template< typename Real = double, typename Device = TNL::Devices::Host, typename Index = int >
struct HeatEquationSolverBenchmarkGrid : public HeatEquationSolverBenchmark< Real, Device, Index >
{
   void
   exec( const Index xSize )
   {
      using Grid = TNL::Meshes::Grid< 1, Real, Device, int >;
      using Coordinates = typename Grid::CoordinatesType;

      Grid grid;

      grid.setDimensions( xSize );
      grid.setDomain( { 0.0 }, { this->xDomainSize } );

      const Real hx = grid.template getSpaceStepsProducts< 1 >();
      const Real hx_inv = grid.template getSpaceStepsProducts< -2 >();

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
            const Real& center = (Real) 2.0 * uxView[ centerIdx ];

            auxView[ centerIdx ] = center
                                 + ( ( uxView[ entity.getNeighbourEntityIndex( Coordinates( -1 ) ) ] - center
                                       + uxView[ entity.getNeighbourEntityIndex( Coordinates( 1 ) ) ] )
                                     * hx_inv )
                                      * timestep;
            /*auxView[ centerIdx ] = center + ( ( uxView[ centerIdx - 1 ] -
                                                2.0 * center +
                                                uxView[ centerIdx + 1 ] ) * hx_inv ) * timestep;*/
         };

         grid.template forInteriorEntities< 1 >( next );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }

   void
   exec( const Index xSize, const Index ySize )
   {
      using Grid = TNL::Meshes::Grid< 2, Real, Device, int >;
      using Coordinates = typename Grid::CoordinatesType;

      Grid grid;

      grid.setDimensions( xSize, ySize );
      grid.setDomain( { 0.0, 0.0 }, { this->xDomainSize, this->yDomainSize } );

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
            const Real& center = (Real) 2.0 * uxView[ centerIdx ];

            auxView[ centerIdx ] = center
                                 + ( ( uxView[ entity.getNeighbourEntityIndex( Coordinates( -1, 0 ) ) ] - center
                                       + uxView[ entity.getNeighbourEntityIndex( Coordinates( 1, 0 ) ) ] )
                                        * hx_inv
                                     + ( uxView[ entity.getNeighbourEntityIndex( Coordinates( 0, -1 ) ) ] - center
                                         + uxView[ entity.getNeighbourEntityIndex( Coordinates( 0, 1 ) ) ] )
                                          * hy_inv )
                                      * timestep;
            /*auxView[ centerIdx ] = center + ( ( uxView[ centerIdx - 1 ] -
                                                2.0 * center +
                                                uxView[ centerIdx + 1 ] ) * hx_inv +
                                                ( uxView[ centerIdx - xSize ] -
                                                2.0 * center +
                                                uxView[ centerIdx + xSize ] ) * hy_inv ) * timestep;*/
         };
         grid.template forInteriorEntities< 2 >( next );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }

   void
   exec( const Index xSize, const Index ySize, const Index zSize )
   {
      using Grid = TNL::Meshes::Grid< 3, Real, Device, int >;
      using Coordinates = typename Grid::CoordinatesType;

      Grid grid;

      grid.setDimensions( xSize, ySize, zSize );
      grid.setDomain( { 0.0, 0.0, 0.0 }, { this->xDomainSize, this->yDomainSize, this->zDomainSize } );

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
            const Real& center = (Real) 2.0 * uxView[ centerIdx ];

            auxView[ centerIdx ] = center
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
            /*auxView[ centerIdx ] = center + ( ( uxView[ centerIdx - 1 ] -         2.0 * center + uxView[ centerIdx + 1 ] ) *
               hx_inv + ( uxView[ centerIdx - xSize ] -     2.0 * center + uxView[ centerIdx + xSize ] ) * hy_inv + ( uxView[
               centerIdx - xySize ] -     2.0 * center + uxView[ centerIdx + xySize ] ) * hy_inv ) * timestep;*/
         };
         grid.template forInteriorEntities< 3 >( next );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }
};
