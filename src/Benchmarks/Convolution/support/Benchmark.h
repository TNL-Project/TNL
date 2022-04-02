
#pragma once

#include <TNL/Config/parseCommandLine.h>

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Containers/Array.h>

template< int Dimension, typename Device >
class Benchmark
{
public:
   using TNLBenchmark = typename TNL::Benchmarks::Benchmark<>;

   void
   run( const TNL::Config::ParameterContainer& parameters ) const
   {
      if( ! TNL::Devices::Host::setup( parameters ) || ! TNL::Devices::Cuda::setup( parameters ) )
         return;

      const TNL::String logFileName = parameters.getParameter< TNL::String >( "log-file" );
      const TNL::String outputMode = parameters.getParameter< TNL::String >( "output-mode" );
      const TNL::String device = parameters.getParameter< TNL::String >( "device" );

      const int verbose = parameters.getParameter< int >( "verbose" );
      const int loops = parameters.getParameter< int >( "loops" );

      auto mode = std::ios::out;

      if( outputMode == "append" )
         mode |= std::ios::app;

      std::ofstream logFile( logFileName.getString(), mode );

      TNLBenchmark benchmark( logFile, loops, verbose );

      std::map< std::string, std::string > metadata = TNL::Benchmarks::getHardwareMetadata();
      TNL::Benchmarks::writeMapAsJson( metadata, logFileName, ".metadata.json" );

      start(benchmark, parameters);
   }

   virtual void start( TNLBenchmark& benchmark, const TNL::Config::ParameterContainer& parameters) const {
      TNL_ASSERT_TRUE(false, "Should be overriden");
   }

   virtual TNL::Config::ConfigDescription makeInputConfig() const {
      TNL::Config::ConfigDescription config;

      config.addDelimiter( "Benchmark settings:" );
      config.addEntry< TNL::String >( "id", "Identifier of the run", "unknown" );
      config.addEntry< TNL::String >( "log-file", "Log file name.", "output.log" );
      config.addEntry< TNL::String >( "output-mode", "Mode for opening the log file.", "overwrite" );
      config.addEntryEnum( "append" );
      config.addEntryEnum( "overwrite" );

      config.addEntry< TNL::String >( "device", "Device the computation will run on.", "cuda" );
      config.addEntryEnum< TNL::String >( "all" );
      config.addEntryEnum< TNL::String >( "host" );

#ifdef HAVE_CUDA
      config.addEntryEnum< TNL::String >( "cuda" );
#endif

      config.addEntry< int >( "loops", "Number of iterations for every computation.", 10 );
      config.addEntry< int >( "verbose", "Verbose mode.", 1 );

      config.addDelimiter( "Device settings:" );
      TNL::Devices::Host::configSetup( config );

#ifdef HAVE_CUDA
      TNL::Devices::Cuda::configSetup( config );
#endif
      return config;
   }
};
