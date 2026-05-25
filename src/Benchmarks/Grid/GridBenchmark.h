
// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include "Operations.h"

#include <TNL/Meshes/Grid.h>
#include <TNL/Config/parseCommandLine.h>

#include <TNL/Devices/Host.h>
#include <TNL/Devices/GPU.h>

#include <TNL/Benchmarks/Benchmark.h>

static std::vector< TNL::String > dimensionParameterIds = { "x-dimension", "y-dimension", "z-dimension" };

template< int EntityDimension, typename Grid, typename Operation >
void
timeTraverse( TNL::Benchmarks::Benchmark& benchmark, const Grid& grid )
{
   auto exec = [] __cuda_callable__( typename Grid::template EntityType< EntityDimension > & entity ) mutable
   {
      Operation::exec( entity );
   };

   auto operation = TNL::getType< Operation >();

   const TNL::Benchmarks::Benchmark::MetadataColumns columns = {
      { "dimensions", TNL::convertToString( grid.getDimensions() ) },
      { "entity dimension", TNL::convertToString( EntityDimension ) },
      { "entities count", TNL::convertToString( grid.getEntitiesCount( EntityDimension ) ) },
      { "operation", operation }
   };

   TNL::Benchmarks::Benchmark::MetadataColumns forAllColumns( columns );
   forAllColumns.emplace_back( "traverse", "forAll" );
   benchmark.setMetadataColumns( forAllColumns );
   auto measureAll = [ = ]()
   {
      grid.template forAllEntities< EntityDimension >( exec );
   };
   benchmark.time< typename Grid::DeviceType >( "TNL", measureAll );

   TNL::Benchmarks::Benchmark::MetadataColumns forInteriorColumns( columns );
   forInteriorColumns.emplace_back( "traverse", "forInterior" );
   benchmark.setMetadataColumns( forInteriorColumns );
   auto measureInterior = [ = ]()
   {
      grid.template forInteriorEntities< EntityDimension >( exec );
   };
   benchmark.time< typename Grid::DeviceType >( "TNL", measureInterior );

   TNL::Benchmarks::Benchmark::MetadataColumns forBoundaryColumns( columns );
   forBoundaryColumns.emplace_back( "traverse", "forBoundary" );
   benchmark.setMetadataColumns( forBoundaryColumns );
   auto measureBoundary = [ = ]()
   {
      grid.template forBoundaryEntities< EntityDimension >( exec );
   };
   benchmark.time< typename Grid::DeviceType >( "TNL", measureBoundary );
}

template< int GridDimension, typename Real, typename Device >
void
runBenchmark( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
   using Grid = TNL::Meshes::Grid< GridDimension, Real, Device, int >;
   using CoordinatesType = typename Grid::CoordinatesType;

   CoordinatesType dimensions;

   for( int i = 0; i < GridDimension; i++ )
      dimensions[ i ] = parameters.getParameter< int >( dimensionParameterIds[ i ] );

   Grid grid;

   grid.setDimensions( dimensions );

   auto forEachEntityDimension = [ & ]( const auto entityDimension )
   {
      timeTraverse< entityDimension, Grid, VoidOperation >( benchmark, grid );

      timeTraverse< entityDimension, Grid, GetEntityIsBoundaryOperation >( benchmark, grid );
      timeTraverse< entityDimension, Grid, GetEntityCoordinateOperation >( benchmark, grid );
      timeTraverse< entityDimension, Grid, GetEntityIndexOperation >( benchmark, grid );
      timeTraverse< entityDimension, Grid, GetEntityNormalsOperation >( benchmark, grid );
      timeTraverse< entityDimension, Grid, RefreshEntityOperation >( benchmark, grid );

      timeTraverse< entityDimension, Grid, GetMeshDimensionOperation >( benchmark, grid );
      timeTraverse< entityDimension, Grid, GetOriginOperation >( benchmark, grid );
      timeTraverse< entityDimension, Grid, GetEntitiesCountsOperation >( benchmark, grid );
   };
   TNL::Algorithms::staticFor< int, 0, GridDimension + 1 >( forEachEntityDimension );
}
