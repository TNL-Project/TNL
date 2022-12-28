// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Containers/StaticArray.h>

template< typename Real = double,
          typename Device = TNL::Devices::Host,
          typename Index = int >
struct HeatEquationSolverBenchmark
{
   static void configSetup( TNL::Config::ConfigDescription& config )
   {
      config.addDelimiter("Benchmark settings:");
      config.addEntry<TNL::String>("id", "Identifier of the run", "unknown");
      config.addEntry<TNL::String>("log-file", "Log file name.", "tnl-benchmark-heat-equation.log");
      config.addEntry<TNL::String>("output-mode", "Mode for opening the log file.", "overwrite");
      config.addEntryEnum("append");
      config.addEntryEnum("overwrite");
      config.addEntry<int>("dimension", "Dimension of the benchmark problem.", 2);
      config.addEntry<int>("min-x-dimension", "Minimum dimension over x axis used in the benchmark.", 100);
      config.addEntry<int>("max-x-dimension", "Maximum dimension over x axis used in the benchmark.", 200);
      config.addEntry<int>("x-size-step-factor", "Factor determining the dimension grows over x axis. First size is min-x-dimension and each following size is stepFactor*previousSize, up to max-x-dimension.", 2);

      config.addEntry<int>("min-y-dimension", "Minimum dimension over y axis used in the benchmark.", 100);
      config.addEntry<int>("max-y-dimension", "Maximum dimension over y axis used in the benchmark.", 200);
      config.addEntry<int>("y-size-step-factor", "Factor determining the dimension grows over y axis. First size is min-y-dimension and each following size is stepFactor*previousSize, up to max-y-dimension.", 2);

      config.addEntry<int>("min-z-dimension", "Minimum dimension over z axis used in the benchmark.", 100);
      config.addEntry<int>("max-z-dimension", "Maximum dimension over z axis used in the benchmark.", 200);
      config.addEntry<int>("z-size-step-factor", "Factor determining the dimension grows over z axis. First size is min-z-dimension and each following size is stepFactor*previousSize, up to max-z-dimension.", 2);

      config.addEntry<int>("loops", "Number of iterations for every computation.", 10);

      config.addEntry<int>("verbose", "Verbose mode.", 1);
      config.addEntry<bool>("write-data", "Write initial condition and final state to a file.", false );

      config.addDelimiter("Problem settings:");
      config.addEntry<double>("domain-x-size", "Domain size along x-axis.", 2.0);
      config.addEntry<double>("domain-y-size", "Domain size along y-axis.", 2.0);

      config.addDelimiter( "Initial condition settings ( (x^2/alpha + y^2/beta + z^2/beta) + delta)):" );
      config.addEntry< double >( "alpha", "Alpha value in initial condition", -0.05 );
      config.addEntry< double >( "beta", "Beta value in initial condition", -0.05 );
      config.addEntry< double >( "gamma", "Gamma value in initial condition", -0.05 );
      config.addEntry< double >( "delta", "Delta value in initial condition", 5 );

      config.addEntry<double>("sigma", "Sigma in exponential initial condition.", 1.0);

      config.addEntry<double>("time-step", "Time step. By default it is proportional to one over space step square.", 0.0 );
      config.addEntry<double>("final-time", "Final time of the simulation.", 0.01);
      config.addEntry<int>("max-iterations", "Maximum time iterations.", 0 );
   }

   void init( const Index xSize )
   {
      this->ux.setSize( xSize );
      this->aux.setSize( xSize );

      this->ux = 0;
      this->aux = 0;

      const Real hx = this->xDomainSize / (Real) xSize;

      auto uxView = this->ux.getView();
      auto xDomainSize_ = this->xDomainSize;
      auto alpha_ = this->alpha;
      auto delta_ = this->delta;
      auto init = [=] __cuda_callable__( Index i ) mutable
      {
         auto x = i * hx - xDomainSize_ / 2.;
         uxView[i] = TNL::max( ( ( x*x / alpha_ ) + delta_ ) * 0.2, 0.0 );
      };
      TNL::Algorithms::ParallelFor<Device>::exec( 1, xSize - 1, init );
   }
   void init( const Index xSize, const Index ySize )
   {
      this->ux.setSize( xSize * ySize );
      this->aux.setSize( xSize * ySize );

      this->ux = 0;
      this->aux = 0;

      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;

      auto uxView = this->ux.getView();
      auto xDomainSize_ = this->xDomainSize;
      auto yDomainSize_ = this->yDomainSize;
      auto alpha_ = this->alpha;
      auto beta_ = this->beta;
      auto gamma_ = this->gamma;
      auto init = [=] __cuda_callable__( const TNL::Containers::StaticArray< 2, int >& i ) mutable
      {
         auto index = i.y() * xSize + i.x();

         auto x = i.x() * hx - xDomainSize_ / 2.;
         auto y = i.y() * hy - yDomainSize_ / 2.;

         uxView[index] = TNL::max( ( ( ( x*x / alpha_ )  + ( y*y / beta_ ) ) + delta_ ) * 0.2, 0.0 );
      };
      const TNL::Containers::StaticArray< 2, int > begin = { 1, 1 };
      const TNL::Containers::StaticArray< 2, int > end = { xSize - 1, ySize - 1 };
      TNL::Algorithms::parallelFor<Device>( begin, end, init );
   }
   void init( const Index xSize, const Index ySize, const Index zSize )
   {
      this->ux.setSize( xSize * ySize * zSize );
      this->aux.setSize( xSize * ySize * zSize );

      this->ux = 0;
      this->aux = 0;

      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;
      const Real hz = this->zDomainSize / (Real) zSize;

      auto uxView = this->ux.getView();
      auto xDomainSize_ = this->xDomainSize;
      auto yDomainSize_ = this->yDomainSize;
      auto zDomainSize_ = this->zDomainSize;
      auto alpha_ = this->alpha;
      auto beta_ = this->beta;
      auto gamma_ = this->gamma;
      auto delta_ = this->delta;
      auto init = [=] __cuda_callable__( Index i, Index j, Index k) mutable
      {
         auto index = ( k * ySize + j ) * xSize + i;

         auto x = i * hx - xDomainSize_ / 2.;
         auto y = j * hy - yDomainSize_ / 2.;
         auto z = k * hz - zDomainSize_ / 2.;

         uxView[index] = TNL::max( ( ( ( x*x / alpha_ ) + ( y*y / beta_ ) + ( z*z / gamma_ ) ) + delta_ ) * 0.2, 0.0 );
      };
      TNL::Algorithms::ParallelFor3D<Device>::exec( 1, 1, 1, xSize - 1, ySize - 1, zSize - 1, init );
   }

   bool writeGnuplot( const std::string &filename,
                      const Index xSize ) const
   {
      std::ofstream out(filename, std::ios::out);
      if( !out.is_open() )
         return false;
      const Real hx = this->xDomainSize / (Real) xSize;
      for( Index i = 0; i < xSize; i++)
         out << i * hx - this->xDomainSize / 2. << " "
            << this->ux.getElement(  i ) << std::endl;
      return out.good();
   }

   bool writeGnuplot( const std::string &filename,
                      const Index xSize, const Index ySize ) const
   {
      std::ofstream out(filename, std::ios::out);
      if( !out.is_open() )
         return false;
      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;
      for( Index j = 0; j < ySize; j++)
         for( Index i = 0; i < xSize; i++)
            out << i * hx - this->xDomainSize / 2. << " "
               << j * hy - this->yDomainSize / 2. << " "
               << this->ux.getElement( j * xSize + i ) << std::endl;
      return out.good();
   }

   virtual void exec( const Index xSize ) = 0;

   virtual void exec( const Index xSize, const Index ySize ) = 0;

   virtual void exec( const Index xSize, const Index ySize, const Index zSize ) = 0;

   bool runBenchmark( const TNL::Config::ParameterContainer& parameters )
   {
      auto implementation = parameters.getParameter< TNL::String >( "implementation" );
      const TNL::String logFileName = parameters.getParameter< TNL::String >( "log-file" );
      const TNL::String outputMode = parameters.getParameter< TNL::String >( "output-mode" );
      bool writeData = parameters.getParameter< bool >( "write-data" );

      const int dimension = parameters.getParameter< int >("dimension");

      const Index minXDimension = parameters.getParameter< int >("min-x-dimension");
      const Index maxXDimension = parameters.getParameter< int >("max-x-dimension");
      const Index xSizeStepFactor = parameters.getParameter< int >("x-size-step-factor");

      if( xSizeStepFactor <= 1 ) {
         std::cerr << "The value of --x-size-step-factor must be greater than 1." << std::endl;
         return false;
      }

      const Index minYDimension = parameters.getParameter< int >("min-y-dimension");
      const Index maxYDimension = parameters.getParameter< int >("max-y-dimension");
      const Index ySizeStepFactor = parameters.getParameter< int >("y-size-step-factor");

      if( dimension > 1 && ySizeStepFactor <= 1 ) {
         std::cerr << "The value of --y-size-step-factor must be greater than 1." << std::endl;
         return false;
      }

      const Index minZDimension = parameters.getParameter< int >("min-z-dimension");
      const Index maxZDimension = parameters.getParameter< int >("max-z-dimension");
      const Index zSizeStepFactor = parameters.getParameter< int >("z-size-step-factor");

      if( dimension > 2 && zSizeStepFactor <= 1 ) {
         std::cerr << "The value of --z-size-step-factor must be greater than 1." << std::endl;
         return false;
      }

      const int loops = parameters.getParameter< int >("loops");
      const int verbose = parameters.getParameter< int >("verbose");

      auto mode = std::ios::out;
      if( outputMode == "append" )
         mode |= std::ios::app;
      std::ofstream logFile( logFileName.getString(), mode );
      TNL::Benchmarks::Benchmark<> benchmark(logFile, loops, verbose);

      // write global metadata into a separate file
      std::map< std::string, std::string > metadata = TNL::Benchmarks::getHardwareMetadata();
      TNL::Benchmarks::writeMapAsJson( metadata, logFileName, ".metadata.json" );

      this->xDomainSize = parameters.getParameter<Real>( "domain-x-size" );
      this->yDomainSize = parameters.getParameter<Real>( "domain-y-size" );
      this->alpha = parameters.getParameter<Real>( "alpha" );
      this->beta = parameters.getParameter<Real>( "beta" );
      this->gamma = parameters.getParameter<Real>( "gamma" );
      this->timeStep = parameters.getParameter<Real>( "time-step" );
      this->finalTime = parameters.getParameter<Real>( "final-time" );
      this->maxIterations = parameters.getParameter< int >( "max-iterations" );

      auto precision = TNL::getType<Real>();
      TNL::String device;
      if( std::is_same< Device, TNL::Devices::Sequential >::value )
         device = "sequential";
      if( std::is_same< Device, TNL::Devices::Host >::value )
         device = "host";
      if( std::is_same< Device, TNL::Devices::Cuda >::value )
         device = "cuda";

      std::cout << "Heat equation benchmark  with (" << precision << ", " << device << ")" << std::endl;

      if( dimension == 1 ) {
         for( Index xSize = minXDimension; xSize <= maxXDimension; xSize *= xSizeStepFactor ) {
            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns( {
               { "precision", precision },
               { "xSize", TNL::convertToString( xSize ) },
               { "implementation", implementation }
            }));

            benchmark.setDatasetSize( xSize );
            this->init( xSize );
            if( writeData ) {
               TNL::String fileName = TNL::String( "initial-" ) + implementation +
                  "-" + TNL::convertToString( xSize) + ".gplt";
               writeGnuplot( fileName.data(), xSize );
            }
            auto lambda = [&]() { this->exec( xSize ); };
            benchmark.time<Device>(device, lambda );
            if( writeData ) {
               TNL::String fileName = TNL::String( "final-" ) + implementation +
                  "-" + TNL::convertToString( xSize) + ".gplt";
               writeGnuplot( fileName.data(), xSize );
            }
         }
      }
      if( dimension == 2 ) {
         for( Index xSize = minXDimension; xSize <= maxXDimension; xSize *= xSizeStepFactor ) {
            for( Index ySize = minYDimension; ySize <= maxYDimension; ySize *= ySizeStepFactor ) {
               benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns( {
                  { "precision", precision },
                  { "xSize", TNL::convertToString( xSize ) },
                  { "ySize", TNL::convertToString( ySize ) },
                  { "implementation", implementation }
               }));

               benchmark.setDatasetSize( xSize * ySize );
               this->init( xSize, ySize );
               if( writeData ) {
                  TNL::String fileName = TNL::String( "initial-" ) + implementation +
                     "-" + TNL::convertToString( xSize) + "-" + TNL::convertToString( ySize ) + ".gplt";
                  writeGnuplot( fileName.data(), xSize, ySize );
               }
               auto lambda = [&]() { this->exec( xSize, ySize ); };
               benchmark.time<Device>(device, lambda );
               if( writeData ) {
                  TNL::String fileName = TNL::String( "final-" ) + implementation +
                     "-" + TNL::convertToString( xSize) + "-" + TNL::convertToString( ySize ) + ".gplt";
                  writeGnuplot( fileName.data(), xSize, ySize );
               }
            }
         }
      }
      if( dimension == 3 ) {
         for( Index xSize = minXDimension; xSize <= maxXDimension; xSize *= xSizeStepFactor ) {
            for( Index ySize = minYDimension; ySize <= maxYDimension; ySize *= ySizeStepFactor ) {
               for( Index zSize = minZDimension; zSize <= maxZDimension; zSize *= zSizeStepFactor ) {
                  benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns( {
                     { "precision", precision },
                     { "xSize", TNL::convertToString( xSize ) },
                     { "ySize", TNL::convertToString( ySize ) },
                     { "zSize", TNL::convertToString( zSize ) },
                     { "implementation", implementation }
                  }));

                  benchmark.setDatasetSize( xSize * ySize * zSize );
                  this->init( xSize, ySize, zSize );
                  auto lambda = [&]() { this->exec( xSize, ySize, zSize ); };
                  benchmark.time<Device>(device, lambda );
               }
            }
         }
      }
      return true;
   }

protected:

   Real xDomainSize = 0.0, yDomainSize = 0.0, zDomainSize = 0.0;
   Real alpha = 0.0, beta = 0.0, gamma = 0.0, delta = 0.0;
   Real timeStep = 0.0, finalTime = 0.0;
   bool outputData = false;
   bool verbose = false;
   Index maxIterations = 0;

   TNL::Containers::Vector<Real, Device> ux, aux;
};
