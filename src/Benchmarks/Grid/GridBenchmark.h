
// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "Operations.h"

#include <vector>

#include <TNL/Meshes/Grid.h>
#include <TNL/Config/parseCommandLine.h>

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

#include <TNL/Benchmarks/Benchmarks.h>

static std::vector< TNL::String > dimensionParameterIds = { "x-dimension", "y-dimension", "z-dimension" };

template< typename Real = double, typename Device = TNL::Devices::Host, typename Index = int >
class GridBenchmark
{
public:
   using Benchmark = TNL::Benchmarks::Benchmark;

   static void
   setupConfig( TNL::Config::ConfigDescription& config )
   {
      config.addDelimiter( "Grid settings:" );
      for( int i = 0; i < 3; i++ )
         config.addEntry< int >( dimensionParameterIds[ i ], "Grid resolution.", 100 );
   }

   template< int GridDimension >
   [[nodiscard]] int
   runBenchmark( const TNL::Config::ParameterContainer& parameters ) const
   {
      Benchmark benchmark;
      benchmark.setup( parameters );

      time< GridDimension >( benchmark, parameters );
      return 0;
   }

   template< int GridDimension >
   void
   time( Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters ) const
   {
      using Grid = typename TNL::Meshes::Grid< GridDimension, Real, Device, int >;
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

   template< int EntityDimension, typename Grid, typename Operation >
   void
   timeTraverse( Benchmark& benchmark, const Grid& grid ) const
   {
      auto exec = [] __cuda_callable__( typename Grid::template EntityType< EntityDimension > & entity ) mutable
      {
         Operation::exec( entity );
      };

      TNL::String device;
      if( std::is_same_v< Device, TNL::Devices::Sequential > )
         device = "sequential";
      if( std::is_same_v< Device, TNL::Devices::Host > )
         device = "host";
      if( std::is_same_v< Device, TNL::Devices::Cuda > )
         device = "cuda";

      auto operation = TNL::getType< Operation >();

      const Benchmark::MetadataColumns columns = { { "dimensions", TNL::convertToString( grid.getDimensions() ) },
                                                   { "entity_dimension", TNL::convertToString( EntityDimension ) },
                                                   { "entitiesCounts",
                                                     TNL::convertToString( grid.getEntitiesCount( EntityDimension ) ) },
                                                   { "operation_id", operation } };

      Benchmark::MetadataColumns forAllColumns( columns );
      forAllColumns.emplace_back( "traverse_id", "forAll" );
      benchmark.setMetadataColumns( forAllColumns );
      auto measureAll = [ = ]()
      {
         grid.template forAllEntities< EntityDimension >( exec );
      };
      benchmark.time< typename Grid::DeviceType >( device, measureAll );

      Benchmark::MetadataColumns forInteriorColumns( columns );
      forInteriorColumns.emplace_back( "traverse_id", "forInterior" );
      benchmark.setMetadataColumns( forInteriorColumns );
      auto measureInterior = [ = ]()
      {
         grid.template forInteriorEntities< EntityDimension >( exec );
      };
      benchmark.time< typename Grid::DeviceType >( device, measureInterior );

      Benchmark::MetadataColumns forBoundaryColumns( columns );
      forBoundaryColumns.emplace_back( "traverse_id", "forBoundary" );
      benchmark.setMetadataColumns( forInteriorColumns );
      auto measureBoundary = [ = ]()
      {
         grid.template forBoundaryEntities< EntityDimension >( exec );
      };
      benchmark.time< typename Grid::DeviceType >( device, measureBoundary );
   }
};
