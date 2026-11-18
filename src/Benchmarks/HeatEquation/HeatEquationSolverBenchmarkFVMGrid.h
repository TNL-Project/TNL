// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/Grid.h>
#include "HeatEquationSolverBenchmark.h"

template< int Dimension, typename Real = double, typename Device = TNL::Devices::Host, typename Index = int >
struct HeatEquationSolverBenchmarkFVMGrid;

template< typename Real, typename Device, typename Index >
struct HeatEquationSolverBenchmarkFVMGrid< 1, Real, Device, Index >
: public HeatEquationSolverBenchmark< 1, Real, Device, Index >
{
   static constexpr int Dimension = 1;
   using BaseBenchmarkType = HeatEquationSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using Grid = TNL::Meshes::Grid< 1, Real, Device, int >;
   using Coordinates = typename Grid::CoordinatesType;
   using Point = typename Grid::PointType;

   TNL::String
   scheme()
   {
      return "fvm";
   }

   void
   init( const Index xSize )
   {
      BaseBenchmarkType::init( xSize, ux, aux );
      this->grid.setSizes( xSize );
      this->grid.setDomain( { 0.0 }, { this->xDomainSize } );
      faces.setSize( this->grid.template getEntitiesCount< 0 >() );
   }

   bool
   writeGnuplot( const std::string& filename, const Index xSize ) const
   {
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize );
   }

   void
   exec( const Index xSize )
   {
      const Real hx = this->grid.getSpaceSteps()[ 0 ];
      const Point h_inv = 1.0 / hx;

      Real start = 0;
      Index iterations = 0;
      auto timestep = this->timeStep ? this->timeStep : 0.1 * hx * hx;
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) ) {
         auto ux_view = this->ux.getView();
         auto aux_view = this->aux.getView();
         auto faces_view = this->faces.getView();
         auto gradient = [ = ] __cuda_callable__( const typename Grid::Face& face ) mutable
         {
            faces_view[ face.getIndex() ] = ( ux_view[ face.template getEntityIndex< Dimension >( { 0, 0 }, 0 ) ]
                                              - ux_view[ face.template getEntityIndex< Dimension >( -face.getNormals(), 0 ) ] )
                                          * h_inv[ face.getOrientationIndex() ];
         };
         this->grid.template forInteriorEntities< 0 >( gradient );

         auto update = [ = ] __cuda_callable__( const typename Grid::Cell& cell ) mutable
         {
            //using Face = typename Grid::Face;
            const Index cellIdx = cell.getIndex();
            const Real& element = ux_view[ cellIdx ];
            aux_view[ cellIdx ] = element
                                + timestep
                                     * ( ( faces_view[ cell.template getEntityIndex< 0 >( { 1 }, 0 ) ]
                                           - faces_view[ cell.template getEntityIndex< 0 >( { 0 }, 0 ) ] )
                                         * h_inv.x() );
         };
         this->grid.template forInteriorEntities< 1 >( update );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }

protected:
   VectorType ux, aux, faces;
   Grid grid;
};

template< typename Real, typename Device, typename Index >
struct HeatEquationSolverBenchmarkFVMGrid< 2, Real, Device, Index >
: public HeatEquationSolverBenchmark< 2, Real, Device, Index >
{
   static constexpr int Dimension = 2;
   using BaseBenchmarkType = HeatEquationSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using Grid = TNL::Meshes::Grid< 2, Real, Device, int >;
   using Coordinates = typename Grid::CoordinatesType;
   using Point = typename Grid::PointType;
   using EntitiesOrientations = typename Grid::EntitiesOrientations;

   TNL::String
   scheme()
   {
      return "fvm";
   }

   void
   init( const Index xSize, const Index ySize )
   {
      BaseBenchmarkType::init( xSize, ySize, ux, aux );
      this->grid.setSizes( { xSize, ySize } );
      this->grid.setDomain( { 0.0, 0.0 }, { this->xDomainSize, this->yDomainSize } );
      faces.setSize( this->grid.template getEntitiesCount< 1 >() );
   }

   bool
   writeGnuplot( const std::string& filename, const Index xSize, const Index ySize ) const
   {
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize, ySize );
   }

   void
   exec( const Index xSize, const Index ySize )
   {
      const Real hx = grid.getSpaceSteps()[ 0 ];
      const Real hy = grid.getSpaceSteps()[ 1 ];
      const Point h_inv = (Real) 1.0 / grid.getSpaceSteps();

      Real start = 0;
      Index iterations = 0;
      auto timestep = this->timeStep ? this->timeStep : 0.1 * std::min( hx * hx, hy * hy );
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) ) {
         auto ux_view = this->ux.getView();
         auto aux_view = this->aux.getView();
         auto faces_view = this->faces.getView();

         //constexpr Index x_faces = EntitiesOrientations::template getOrientationIndex< 1, 0, 1 >();
         //constexpr Index y_faces = EntitiesOrientations::template getOrientationIndex< 1, 1, 0 >();
         //auto x_faces_view = this->grid.partitionEntities( this->faces, 1, x_faces );
         //auto y_faces_view = this->grid.partitionEntities( this->faces, 1, y_faces );

         auto gradient = [ = ] __cuda_callable__( const typename Grid::Face& face ) mutable
         {
            Index closer, remoter;
            face.getAdjacentCells( closer, remoter );
            //faces_view[ face.getIndex() ] = ( ux_view[ face.template getNeighbourEntityIndex< Dimension >( { 0, 0 }, 0 ) ] -
            //                                 ux_view[ face.template getNeighbourEntityIndex< Dimension >( -face.getNormals(),
            //                                 0 ) ] )
            //                                * h_inv[ face.getOrientationIndex() ] ;
            faces_view[ face.getIndex() ] = ( ux_view[ remoter ] - ux_view[ closer ] ) * h_inv[ face.getOrientationIndex() ];
         };
         this->grid.template forInteriorEntities< 1 >( gradient );

         auto update = [ = ] __cuda_callable__( const typename Grid::Cell& cell ) mutable
         {
            //using Face = typename Grid::Face;
            const Index cellIdx = cell.getIndex();
            const Real& element = ux_view[ cellIdx ];
            Coordinates closer, remoter;
            cell.getAdjacentFacesIndexes( closer, remoter );
            aux_view[ cellIdx ] = element
                                + ( ( faces_view[ remoter[ 0 ] ] - faces_view[ closer[ 0 ] ] ) * h_inv.x()
                                    + ( faces_view[ remoter[ 1 ] ] - faces_view[ closer[ 1 ] ] ) * h_inv.y() )
                                     * timestep;

            //aux_view[ cellIdx ] = element + ( ( faces_view[ cell.template getNeighbourEntityIndex< 1 >( { 1, 0 }, y_faces ) ]
            //-
            //                                    faces_view[ cell.template getNeighbourEntityIndex< 1 >( { 0, 0 }, y_faces ) ]
            //                                    ) * h_inv.x() +
            //                                  ( faces_view[ cell.template getNeighbourEntityIndex< 1 >( { 0, 1 }, x_faces ) ]
            //                                  -
            //                                    faces_view[ cell.template getNeighbourEntityIndex< 1 >( { 0, 0 }, x_faces ) ]
            //                                    ) * h_inv.y() )  * timestep;
         };
         this->grid.template forInteriorEntities< 2 >( update );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }

protected:
   VectorType ux, aux, faces;

   Grid grid;
};

template< typename Real, typename Device, typename Index >
struct HeatEquationSolverBenchmarkFVMGrid< 3, Real, Device, Index >
: public HeatEquationSolverBenchmark< 3, Real, Device, Index >
{
   static constexpr int Dimension = 3;
   using BaseBenchmarkType = HeatEquationSolverBenchmark< Dimension, Real, Device, Index >;
   using VectorType = typename BaseBenchmarkType::VectorType;
   using Grid = TNL::Meshes::Grid< 3, Real, Device, int >;
   using Coordinates = typename Grid::CoordinatesType;
   using Point = typename Grid::PointType;
   using EntitiesOrientations = typename Grid::EntitiesOrientations;

   TNL::String
   scheme()
   {
      return "fvm";
   }

   void
   init( const Index xSize, const Index ySize, const Index zSize )
   {
      BaseBenchmarkType::init( xSize, ySize, zSize, ux, aux );
      grid.setSizes( { xSize, ySize, zSize } );
      grid.setDomain( { 0.0, 0.0, 0.0 }, { this->xDomainSize, this->yDomainSize, this->zDomainSize } );
      faces.setSize( this->grid.template getEntitiesCount< 2 >() );
   }

   bool
   writeGnuplot( const std::string& filename, const Index xSize, const Index ySize, const Index zSize, const Index zSlice )
      const
   {
      return BaseBenchmarkType::writeGnuplot( filename, ux, xSize, ySize, zSize, zSlice );
   }

   void
   exec( const Index xSize, const Index ySize, const Index zSize )
   {
      const Real hx = grid.getSpaceSteps()[ 0 ];
      const Real hy = grid.getSpaceSteps()[ 1 ];
      const Real hz = grid.getSpaceSteps()[ 2 ];
      const Point h_inv = 1.0 / grid.getSpaceSteps();

      Real start = 0;
      Index iterations = 0;
      auto timestep = this->timeStep ? this->timeStep : 0.1 * std::min( hx * hx, std::min( hy * hy, hz * hz ) );
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) ) {
         auto ux_view = this->ux.getView();
         auto aux_view = this->aux.getView();
         auto faces_view = this->faces.getView();
         auto gradient = [ = ] __cuda_callable__( const typename Grid::Face& face ) mutable
         {
            faces_view[ face.getIndex() ] = ( ux_view[ face.template getEntityIndex< Dimension >( { 0, 0, 0 }, 0 ) ]
                                              - ux_view[ face.template getEntityIndex< Dimension >( -face.getNormals(), 0 ) ] )
                                          * h_inv[ face.getOrientationIndex() ];
         };
         this->grid.template forInteriorEntities< 2 >( gradient );

         constexpr Index yz_faces = EntitiesOrientations::template getOrientationIndex< 2, 1, 0, 0 >();
         constexpr Index xz_faces = EntitiesOrientations::template getOrientationIndex< 2, 0, 1, 0 >();
         constexpr Index xy_faces = EntitiesOrientations::template getOrientationIndex< 2, 0, 0, 1 >();
         auto update = [ = ] __cuda_callable__( const typename Grid::Cell& cell ) mutable
         {
            //using Face = typename Grid::Face;
            const Index cell_idx = cell.getIndex();
            aux_view[ cell_idx ] = ux_view[ cell_idx ]
                                 + timestep
                                      * ( ( faces_view[ cell.template getEntityIndex< 2 >( { 1, 0, 0 }, yz_faces ) ]
                                            - faces_view[ cell.template getEntityIndex< 2 >( { 0, 0, 0 }, yz_faces ) ] )
                                             * h_inv.x()
                                          + ( faces_view[ cell.template getEntityIndex< 2 >( { 0, 1, 0 }, xz_faces ) ]
                                              - faces_view[ cell.template getEntityIndex< 2 >( { 0, 0, 0 }, xz_faces ) ] )
                                               * h_inv.y()
                                          + ( faces_view[ cell.template getEntityIndex< 2 >( { 0, 0, 1 }, xy_faces ) ]
                                              - faces_view[ cell.template getEntityIndex< 2 >( { 0, 0, 0 }, xy_faces ) ] )
                                               * h_inv.z() );
         };
         grid.template forInteriorEntities< 3 >( update );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }

protected:
   VectorType ux, aux, faces;
   Grid grid;
};
