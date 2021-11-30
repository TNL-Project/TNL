
#include <TNL/Config/parseCommandLine.h>

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

#include <TNL/Benchmarks/Benchmarks.h>

#include "HeatmapSolver.h"

class HeatmapSolverBenchmark {
   public:
      template<typename Real, typename Device>
      void exec(const HeatmapSolver::Parameters& params) const;

      template< typename Real >
      void runBenchmark(const Benchmark& benchmark,
                        const Benchmark::MetadataMap,
                        const int minXDimension,
                        const int maxXDimension,
                        const int xStepSizeFactor,
                        const int minYDimension,
                        const int maxYDimension,
                        const int yStepSizeFactor,
                        const TNL::Config::ParameterContainer& parameters) const;

      static TNL::Config::ConfigDescription makeInputConfig();
};

TNL::Config::ConfigDescription HeatmapSolverBenchmark::makeInputConfig() {
   TNL::Config::ConfigDescription config;

   config.addDelimiter("Benchmark settings:");
   config.addEntry<TNL::String>("log-file", "Log file name.", "tnl-benchmark-heatmap.log");
   config.addEntry<TNL::String>("output-mode", "Mode for opening the log file.", "overwrite");
   config.addEntryEnum("append");
   config.addEntryEnum("overwrite");

   config.addEntry<TNL::String>("precision", "Precision of the arithmetics.", "double");
   config.addEntryEnum("float");
   config.addEntryEnum("double");
   config.addEntryEnum("all");

   config.addEntry<int>("min-x-dimension", "Minimum dimension over x axis used in the benchmark.", 100);
   config.addEntry<int>("max-x-dimension", "Maximum dimension over x axis used in the benchmark.", 10000);
   config.addEntry<int>("x-size-step-factor", "Factor determining the dimension grows over x axis. First size is min-x-dimension and each following size is stepFactor*previousSize, up to max-x-dimension.", 2);

   config.addEntry<int>("min-y-dimension", "Minimum dimension over x axis used in the benchmark.", 100);
   config.addEntry<int>("max-y-dimension", "Maximum dimension over x axis used in the benchmark.", 10000);
   config.addEntry<int>("y-size-step-factor", "Factor determining the dimension grows over y axis. First size is min-y-dimension and each following size is stepFactor*previousSize, up to max-y-dimension.", 2);

   config.addEntry<int>("loops", "Number of iterations for every computation.", 10);

   config.addEntry<int>("verbose", "Verbose mode.", 1);

   config.addDelimiter("Problem settings:");
   config.addEntry<double>("domain-x-size", "Domain size along x-axis.", 2.0);
   config.addEntry<double>("domain-y-size", "Domain size along y-axis.", 2.0);

   config.addEntry<double>("sigma", "Sigma in exponential initial condition.", 1.0);

   config.addEntry<double>("time-step", "Time step. By default it is proportional to one over space step square.", 0.0);
   config.addEntry<double>("final-time", "Final time of the simulation.", 0.012);

   config.addDelimiter("Device settings:");
   TNL::Devices::Host::configSetup( config );
   TNL::Devices::Cuda::configSetup( config );
}

template<typename Real,
         template Device>
void HeatmapSolverBenchmark::runBenchmark(Benchmark& benchmark,
                                          Benchmark::MetadataMap metadata,
                                          const int minXDimension,
                                          const int maxXDimension,
                                          const int xStepSizeFactor,
                                          const int minYDimension,
                                          const int maxYDimension,
                                          const int yStepSizeFactor,
                                          const TNL::Config::ParameterContainer& parameters) const {
   const TNL::String precision = getType< Real >();
   metadata["precision"] = precision;

   benchmark.newBenchmark(String("Heatmap grid with (") + precision + ", host allocator" + Device + ")");

   auto xDomainSize = parameters.getParameter<Real>("domain-x-size");
   auto yDomainSize = parameters.getParameter<Real>("domain-y-size");

   auto sigma = parameters.getParameter<Real>("sigma");

   auto timeStep = parameters.getParameter<Real>("time-step");

   auto finalTime = parameters.getParameter<Real>("final-time");


   for(std::size_t xSize = minXDimension; xSize <= maxXDimension; size *= xStepSizeFactor) {
      for(std::size_t ySize = minXDimension; ySize <= maxXDimension; size *= yStepSizeFactor) {
         benchmark.setMetadataColumns( Benchmark::MetadataColumns({
            { "xSize", convertToString(xSize) },
            { "ySize", convertToString(ySize) }
         }));

         auto lambda = [=]() {
            HeatmapSolver::Parameters params(xSize, ySize, xDomainSize, yDomainSize, sigma, timeStep, finalTime, false, false);

            exec(params);
         };

         benchmark.time<Device>()
      }
   }
}

int main(int argc, char* argv[]) {
   HeatmapSolverBenchmark solver;

   auto config = HeatmapSolverBenchmark::makeInputConfig();

   TNL::Config::ParameterContainer parameters;

   if (!parseCommandLine(argc, argv, config, parameters))
      return EXIT_FAILURE;

   if (!TNL::Devices::Host::setup( parameters ) || !TNL::Devices::Cuda::setup( parameters ) )
      return EXIT_FAILURE;

   const TNL::String logFileName = parameters.getParameter<TNL::String>( "log-file" );
   const TNL::String outputMode = parameters.getParameter<TNL::String>( "output-mode" );
   const TNL::String precision = parameters.getParameter<TNL::String>( "precision" );

   const std::size_t minXDimension = parameters.getParameter<int>("min-x-dimension");
   const std::size_t maxXDimension = parameters.getParameter<int>("max-x-dimension");
   const std::size_t xSizeStepFactor = parameters.getParameter<int>("x-size-step-factor");

   if(xSizeStepFactor <= 1) {
      std::cerr << "The value of --x-size-step-factor must be greater than 1." << std::endl;
      return EXIT_FAILURE;
   }

   const std::size_t minYDimension = parameters.getParameter<int>("min-y-dimension");
   const std::size_t maxYDimension = parameters.getParameter<int>("max-y-dimension");
   const std::size_t ySizeStepFactor = parameters.getParameter<int>("y-size-step-factor");

   const int loops = parameters.getParameter< int >("loops");
   const int verbose = parameters.getParameter< int >("verbose");

   if(ySizeStepFactor <= 1) {
      std::cerr << "The value of --y-size-step-factor must be greater than 1." << std::endl;
      return EXIT_FAILURE;
   }

   auto mode = std::ios::out;

   if( outputMode == "append" )
      mode |= std::ios::app;

   std::ofstream logFile( logFileName.getString(), mode );

   TNL::Benchmarks::Benchmark benchmark(loops, verbose);

   if( precision == "all" || precision == "float" )
      solver.runBenchmark<float>(benchmark, metadata, minXDimension, maxXDimension, xSizeStepFactor, minXDimension, maxYDimension, ySizeStepFactor, parameters);
   if( precision == "all" || precision == "double" )
      solver.runBenchmark<double>(benchmark, metadata, minXDimension, maxXDimension, xSizeStepFactor, minXDimension, maxYDimension, ySizeStepFactor, parameters);

   if(!benchmark.save(logFile)) {
      std::cerr << "Failed to write the benchmark results to file '" << logFileName << "'." << std::endl;
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
