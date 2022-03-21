
#pragma once

#include "Operations.h"

#include <vector>

#include <TNL/Meshes/Grid.h>
#include <TNL/Config/parseCommandLine.h>

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

#include <TNL/Benchmarks/Benchmarks.h>

static std::vector<TNL::String> dimensionParameterIds = { "x-dimension", "y-dimension", "z-dimension" };

class GridBenchmark {
   public:
      using Benchmark = typename TNL::Benchmarks::Benchmark<>;

      template<int GridDimension>
      int runBenchmark(const TNL::Config::ParameterContainer& parameters) const {
         if (!TNL::Devices::Host::setup( parameters ) || !TNL::Devices::Cuda::setup( parameters ) )
            return EXIT_FAILURE;

         const TNL::String logFileName = parameters.getParameter<TNL::String>( "log-file" );
         const TNL::String outputMode = parameters.getParameter<TNL::String>( "output-mode" );
         const TNL::String precision = parameters.getParameter<TNL::String>( "precision" );
         const TNL::String device = parameters.getParameter<TNL::String>("device");

         const int verbose = parameters.getParameter< int >("verbose");
         const int loops = parameters.getParameter< int >("loops");

         auto mode = std::ios::out;

         if( outputMode == "append" )
            mode |= std::ios::app;

         std::ofstream logFile( logFileName.getString(), mode );

         Benchmark benchmark(logFile, loops, verbose);

         // write global metadata into a separate file
         std::map< std::string, std::string > metadata = TNL::Benchmarks::getHardwareMetadata();
         TNL::Benchmarks::writeMapAsJson( metadata, logFileName, ".metadata.json" );

         if (device == "host" || device == "all") {
            if (precision == "all" || precision == "float")
               time<GridDimension, float, TNL::Devices::Host>(benchmark, parameters);
            if (precision == "all" || precision == "double")
               time<GridDimension, double, TNL::Devices::Host>(benchmark, parameters);
         }

         #ifdef HAVE_CUDA
            if (device == "cuda" || device == "all") {
               if (precision == "all" || precision == "float")
                  time<GridDimension, float, TNL::Devices::Cuda>(benchmark, parameters);
               if (precision == "all" || precision == "double")
                  time<GridDimension, double, TNL::Devices::Cuda>(benchmark, parameters);
            }
         #endif

         return 0;
      }

      template<int GridDimension, typename Real, typename Device>
      void time(Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters) const {
         using Grid = typename TNL::Meshes::Grid<GridDimension, Real, Device, int>;
         using Coordinate = typename Grid::Coordinate;

         Coordinate dimensions;

         for (int i = 0; i < GridDimension; i++)
            dimensions[i] = parameters.getParameter<int>(dimensionParameterIds[i]);

         Grid grid;

         grid.setDimensions(dimensions);

         auto forEachEntityDimension = [&](const auto entityDimension) {
            using Entity = TNL::Meshes::GridEntity<Grid, entityDimension>;

            timeTraverse<entityDimension, Grid, VoidOperation>(benchmark, grid);

            timeTraverse<entityDimension, Grid, GetEntityIsBoundaryOperation>(benchmark, grid);
            timeTraverse<entityDimension, Grid, GetEntityCoordinateOperation>(benchmark, grid);
            timeTraverse<entityDimension, Grid, GetEntityIndexOperation>(benchmark, grid);
            timeTraverse<entityDimension, Grid, GetEntityBasisOperation>(benchmark, grid);
            timeTraverse<entityDimension, Grid, RefreshEntityOperation>(benchmark, grid);

            timeTraverse<entityDimension, Grid, GetMeshDimensionOperation>(benchmark, grid);
            timeTraverse<entityDimension, Grid, GetOriginOperation>(benchmark, grid);
            timeTraverse<entityDimension, Grid, GetEntitiesCountsOperation>(benchmark, grid);
         };

         TNL::Meshes::Templates::DescendingFor<GridDimension>::exec(forEachEntityDimension);
      }

      static TNL::Config::ConfigDescription makeInputConfig(int gridDimension) {
         TNL_ASSERT_LE(gridDimension, 3, "Only support for grids with dimension less or equal 3");
         TNL::Config::ConfigDescription config;

         config.addDelimiter("Benchmark settings:");
         config.addEntry<TNL::String>("id", "Identifier of the run", "unknown");
         config.addEntry<TNL::String>("log-file", "Log file name.", "output.log");
         config.addEntry<TNL::String>("output-mode", "Mode for opening the log file.", "overwrite");
         config.addEntryEnum("append");
         config.addEntryEnum("overwrite");

         config.addEntry<TNL::String>("device", "Device the computation will run on.", "cuda");
         config.addEntryEnum<TNL::String>("all");
         config.addEntryEnum<TNL::String>("host");

         #ifdef HAVE_CUDA
            config.addEntryEnum<TNL::String>("cuda");
         #endif

         config.addEntry<TNL::String>("precision", "Precision of the arithmetics.", "double");
         config.addEntryEnum("float");
         config.addEntryEnum("double");
         config.addEntryEnum("all");

         config.addEntry<int>("loops", "Number of iterations for every computation.", 10);
         config.addEntry<int>("verbose", "Verbose mode.", 1);

         for (int i = 0; i < gridDimension; i++)
            config.addEntry<int>(dimensionParameterIds[i], "The " + dimensionParameterIds[i] + " of grid:", 100);

         config.addDelimiter("Device settings:");
         TNL::Devices::Host::configSetup( config );

         #ifdef HAVE_CUDA
            TNL::Devices::Cuda::configSetup( config );
         #endif

         return config;
      }

      template<int EntityDimension, typename Grid, typename Operation>
      void timeTraverse(Benchmark& benchmark, const Grid& grid) const {
         auto exec = [] __cuda_callable__ (typename Grid::EntityType<EntityDimension>& entity) mutable {
            Operation::exec(entity);
         };

         auto device = TNL::getType<typename Grid::DeviceType>();
         auto operation = TNL::getType<Operation>();

         const Benchmark::MetadataColumns columns = {
            { "operation_id", operation },
            { "dimensions", TNL::convertToString(grid.getDimensions()) },
            { "entity_dimension", TNL::convertToString(EntityDimension) },
            { "entitiesCounts", TNL::convertToString(grid.getEntitiesCount(EntityDimension)) }
         };

         Benchmark::MetadataColumns forAllColumns = {
            { "traverse_id", "forAll" }
         };

         forAllColumns.insert(forAllColumns.end(), columns.begin(), columns.end());
         benchmark.setMetadataColumns(forAllColumns);

         auto measureAll = [=]() {
            grid.template forAll<EntityDimension>(exec);
         };

         benchmark.time<typename Grid::DeviceType>(device, measureAll);


         Benchmark::MetadataColumns forInteriorColumns = {
            { "traverse_id", "forInterior" }
         };

         forInteriorColumns.insert(forInteriorColumns.end(), columns.begin(), columns.end());
         benchmark.setMetadataColumns(forInteriorColumns);

         auto measureInterior = [=]() {
            grid.template forInterior<EntityDimension>(exec);
         };

         benchmark.time<typename Grid::DeviceType>(device, measureInterior);


         Benchmark::MetadataColumns forBoundaryColumns = {
            { "traverse_id", "forBoundary" }
         };

         forBoundaryColumns.insert(forBoundaryColumns.end(), columns.begin(), columns.end());
         benchmark.setMetadataColumns(forInteriorColumns);

         auto measureBoundary = [=]() {
            grid.template forBoundary<EntityDimension>(exec);
         };

         benchmark.time<typename Grid::DeviceType>(device, measureBoundary);
      }
};
