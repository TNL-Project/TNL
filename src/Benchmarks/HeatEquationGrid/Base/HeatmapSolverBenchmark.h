
#include <TNL/Config/parseCommandLine.h>

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

#include <TNL/Benchmarks/Benchmarks.h>

#include "HeatmapSolver.h"

#pragma once

class HeatmapSolverBenchmark {
   public:
      template<typename Real, typename Device>
      void exec(const typename HeatmapSolver<Real>::Parameters& params) const;

      template<typename Real, typename Device>
      void runBenchmark(TNL::Benchmarks::Benchmark<> & benchmark,
                        const std::size_t minXDimension,
                        const std::size_t maxXDimension,
                        const std::size_t xStepSizeFactor,
                        const std::size_t minYDimension,
                        const std::size_t maxYDimension,
                        const std::size_t yStepSizeFactor,
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

   config.addEntry<int>("min-x-dimension", "Minimum dimension over x axis used in the benchmark.", 100);
   config.addEntry<int>("max-x-dimension", "Maximum dimension over x axis used in the benchmark.", 200);
   config.addEntry<int>("x-size-step-factor", "Factor determining the dimension grows over x axis. First size is min-x-dimension and each following size is stepFactor*previousSize, up to max-x-dimension.", 2);

   config.addEntry<int>("min-y-dimension", "Minimum dimension over x axis used in the benchmark.", 100);
   config.addEntry<int>("max-y-dimension", "Maximum dimension over x axis used in the benchmark.", 200);
   config.addEntry<int>("y-size-step-factor", "Factor determining the dimension grows over y axis. First size is min-y-dimension and each following size is stepFactor*previousSize, up to max-y-dimension.", 2);

   config.addEntry<int>("loops", "Number of iterations for every computation.", 10);

   config.addEntry<int>("verbose", "Verbose mode.", 1);

   config.addDelimiter("Problem settings:");
   config.addEntry<double>("domain-x-size", "Domain size along x-axis.", 2.0);
   config.addEntry<double>("domain-y-size", "Domain size along y-axis.", 2.0);

   config.addEntry<double>("sigma", "Sigma in exponential initial condition.", 1.0);

   config.addEntry<double>("time-step", "Time step. By default it is proportional to one over space step square.", 0.00001);
   config.addEntry<double>("final-time", "Final time of the simulation.", 0.01);

   config.addDelimiter("Device settings:");
   TNL::Devices::Host::configSetup( config );
   TNL::Devices::Cuda::configSetup( config );

   return config;
}

template<typename Real, typename Device>
void HeatmapSolverBenchmark::exec(const typename HeatmapSolver<Real>::Parameters& params) const {
   HeatmapSolver<Real> solver;

   auto result = solver.template solve<Device>(params);

   if (!result)
      printf("Fail to solve for grid size (%d,%d)", params.xSize, params.ySize);
}

template<typename Real, typename Device>
void HeatmapSolverBenchmark::runBenchmark(TNL::Benchmarks::Benchmark<>& benchmark,
                                          const std::size_t minXDimension,
                                          const std::size_t maxXDimension,
                                          const std::size_t xStepSizeFactor,
                                          const std::size_t minYDimension,
                                          const std::size_t maxYDimension,
                                          const std::size_t yStepSizeFactor,
                                          const TNL::Config::ParameterContainer& parameters) const {
   Real xDomainSize = parameters.getParameter<Real>("domain-x-size");
   Real yDomainSize = parameters.getParameter<Real>("domain-y-size");
   Real sigma = parameters.getParameter<Real>("sigma");
   Real timeStep = parameters.getParameter<Real>("time-step");
   Real finalTime = parameters.getParameter<Real>("final-time");

   auto precision = TNL::getType<Real>(), device = TNL::getType<Device>();

   std::cout << "Heatmap grid with (" + precision + ", host allocator " + device + ")" << std::endl;

   for(std::size_t xSize = minXDimension; xSize <= maxXDimension; xSize *= xStepSizeFactor) {
      for(std::size_t ySize = minXDimension; ySize <= maxXDimension; ySize *= yStepSizeFactor) {
         benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns({
            { "precision", precision },
            { "xSize", TNL::convertToString(xSize) },
            { "ySize", TNL::convertToString(ySize) }
         }));

         benchmark.setDatasetSize(xSize * ySize);

         auto lambda = [=]() {
            typename HeatmapSolver<Real>::Parameters params(xSize, ySize, xDomainSize, yDomainSize, sigma, timeStep, finalTime, false, false);

            exec<Real, Device>(params);
         };

         benchmark.time<Device>(device, lambda);
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

   TNL::Benchmarks::Benchmark<> benchmark(logFile, loops, verbose);

   // write global metadata into a separate file
   std::map< std::string, std::string > metadata = TNL::Benchmarks::getHardwareMetadata();
   TNL::Benchmarks::writeMapAsJson( metadata, logFileName, ".metadata.json" );

   auto device = parameters.getParameter<TNL::String>("device");

   if (device == "host" || device == "all") {
      if(precision == "all" || precision == "float")
         solver.runBenchmark<float, TNL::Devices::Host>(benchmark, minXDimension, maxXDimension, xSizeStepFactor, minYDimension, maxYDimension, ySizeStepFactor, parameters);
      if(precision == "all" || precision == "double")
         solver.runBenchmark<double, TNL::Devices::Host>(benchmark, minXDimension, maxXDimension, xSizeStepFactor, minYDimension, maxYDimension, ySizeStepFactor, parameters);
   }

#ifdef HAVE_CUDA
   if (device == "cuda" || device == "all") {
      if( precision == "all" || precision == "float" )
         solver.runBenchmark<float, TNL::Devices::Cuda>(benchmark, minXDimension, maxXDimension, xSizeStepFactor, minYDimension, maxYDimension, ySizeStepFactor, parameters);
      if( precision == "all" || precision == "double" )
         solver.runBenchmark<double, TNL::Devices::Cuda>(benchmark, minXDimension, maxXDimension, xSizeStepFactor, minYDimension, maxYDimension, ySizeStepFactor, parameters);
   }
#endif

   return EXIT_SUCCESS;
}
