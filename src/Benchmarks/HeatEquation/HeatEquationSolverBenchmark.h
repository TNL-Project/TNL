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
          typename Index = int >
struct HeatEquationSolverBenchmarkBase
{
   static void configSetup( TNL::Config::ConfigDescription& config )
   {
      config.addDelimiter("Benchmark settings:");
      config.addEntry<TNL::String>("id", "Identifier of the run", "unknown");
      config.addEntry<TNL::String>("log-file", "Log file name.", "tnl-benchmark-heat-equation.log");
      config.addEntry<TNL::String>("output-mode", "Mode for opening the log file.", "overwrite");
      config.addEntryEnum("append");
      config.addEntryEnum("overwrite");
      config.addEntry<int>("loops", "Number of benchmarking loops to compute average time of benchmarked problem.", 10);
      config.addEntry<int>("verbose", "Verbose mode.", 1);
      config.addEntry<bool>("write-data", "Write initial condition and final state to a file.", false );

      config.addDelimiter("Problem settings:");
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

      config.addEntry<double>("domain-x-size", "Domain size along x-axis.", 2.0);
      config.addEntry<double>("domain-y-size", "Domain size along y-axis.", 2.0);
      config.addEntry<double>("domain-z-size", "Domain size along z-axis.", 2.0);

      config.addDelimiter( "Initial condition settings delta * ( 1-sign( x^2/alpha + y^2/beta + z^2/gamma - 1)):" );
      config.addEntry< double >( "alpha", "Alpha value in initial condition", 0.25 );
      config.addEntry< double >( "beta",  "Beta value in initial condition",  0.25 );
      config.addEntry< double >( "gamma", "Gamma value in initial condition", 0.25 );
      config.addEntry< double >( "delta", "Delta value in initial condition", 0.25 );

      config.addEntry<double>("sigma", "Sigma in exponential initial condition.", 1.0);

      config.addEntry<double>("time-step", "Time step. By default it is proportional to one over space step square.", 0.0 );
      config.addEntry<double>("final-time", "Final time of the simulation.", 0.01);
      config.addEntry<int>("max-iterations", "Maximum time iterations.", 0 );
   }

   bool setup( const TNL::Config::ParameterContainer& parameters, int dimension )
   {
      this->implementation = parameters.getParameter< TNL::String >( "implementation" );
      this->logFileName = parameters.getParameter< TNL::String >( "log-file" );
      this->outputMode = parameters.getParameter< TNL::String >( "output-mode" );
      this->writeData = parameters.getParameter< bool >( "write-data" );

      this->minXDimension = parameters.getParameter< int >("min-x-dimension");
      this->maxXDimension = parameters.getParameter< int >("max-x-dimension");
      this->xSizeStepFactor = parameters.getParameter< int >("x-size-step-factor");

      if( xSizeStepFactor <= 1 ) {
         std::cerr << "The value of --x-size-step-factor must be greater than 1." << std::endl;
         return false;
      }

      this->minYDimension = parameters.getParameter< int >("min-y-dimension");
      this->maxYDimension = parameters.getParameter< int >("max-y-dimension");
      this->ySizeStepFactor = parameters.getParameter< int >("y-size-step-factor");

      if( dimension > 1 && ySizeStepFactor <= 1 ) {
         std::cerr << "The value of --y-size-step-factor must be greater than 1." << std::endl;
         return false;
      }

      this->minZDimension = parameters.getParameter< int >("min-z-dimension");
      this->maxZDimension = parameters.getParameter< int >("max-z-dimension");
      this->zSizeStepFactor = parameters.getParameter< int >("z-size-step-factor");

      if( dimension > 2 && zSizeStepFactor <= 1 ) {
         std::cerr << "The value of --z-size-step-factor must be greater than 1." << std::endl;
         return false;
      }

      this->xDomainSize = parameters.getParameter<Real>( "domain-x-size" );
      this->yDomainSize = parameters.getParameter<Real>( "domain-y-size" );
      this->zDomainSize = parameters.getParameter<Real>( "domain-z-size" );
      this->alpha = parameters.getParameter<Real>( "alpha" );
      this->beta = parameters.getParameter<Real>( "beta" );
      this->gamma = parameters.getParameter<Real>( "gamma" );
      this->delta = parameters.getParameter<Real>( "delta" );
      this->timeStep = parameters.getParameter<Real>( "time-step" );
      this->finalTime = parameters.getParameter<Real>( "final-time" );
      this->maxIterations = parameters.getParameter< int >( "max-iterations" );

      loops = parameters.getParameter< int >("loops");
      verbose = parameters.getParameter< int >("verbose");

      return true;
   }

protected:

   TNL::String implementation, logFileName, outputMode;
   bool writeData;

   Index minXDimension, maxXDimension, xSizeStepFactor;
   Index minYDimension, maxYDimension, ySizeStepFactor;
   Index minZDimension, maxZDimension, zSizeStepFactor;

   Real xDomainSize = 0.0, yDomainSize = 0.0, zDomainSize = 0.0;
   Real alpha = 0.0, beta = 0.0, gamma = 0.0, delta = 0.0;

   Real timeStep = 0.0, finalTime = 0.0;
   bool outputData = false;
   bool verbose = false;
   Index maxIterations = 0;

   int loops;
};

template< int Dimension = 2,
          typename Real = double,
          typename Device = TNL::Devices::Host,
          typename Index = int >
struct HeatEquationSolverBenchmark;

template< typename Real,
          typename Device,
          typename Index >
struct HeatEquationSolverBenchmark< 1, Real, Device, Index > : public HeatEquationSolverBenchmarkBase< Real, Index >
{
   static constexpr int Dimension = 1;
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;

   virtual TNL::String scheme() = 0;

   virtual void init( const Index xSize ) = 0;

   virtual void exec( const Index xSize ) = 0;

   virtual bool writeGnuplot( const std::string &filename, const Index xSize ) const = 0;

   void init( const Index xSize, VectorType& ux, VectorType& aux )
   {
      ux.setSize( xSize );
      aux.setSize( xSize );

      ux = 0;
      aux = 0;

      const Real hx = this->xDomainSize / (Real) xSize;
      auto uxView = ux.getView();
      auto xDomainSize_ = this->xDomainSize;
      auto delta_ = this->delta;
      auto alpha_ = this->alpha;
      auto init = [=] __cuda_callable__( Index i ) mutable
      {
         auto x = i * hx - xDomainSize_ / 2.0;
         uxView[i] = delta_ * ( 1.0 - TNL::sign( x*x / alpha_ - 1.0 ) );
      };
      TNL::Algorithms::ParallelFor<Device>::exec( 1, xSize - 1, init );
   }

   template< typename Vector >
   bool writeGnuplot( const std::string &filename,
                      const Vector& u,
                      const Index xSize ) const
   {
      std::ofstream out(filename, std::ios::out);
      if( !out.is_open() )
         return false;
      const Real hx = this->xDomainSize / (Real) xSize;
      for( Index i = 0; i < xSize; i++)
         out << i * hx - this->xDomainSize / 2. << " "
            << u.getElement(  i ) << std::endl;
      return out.good();
   }

   bool runBenchmark( const TNL::Config::ParameterContainer& parameters )
   {
      if( ! HeatEquationSolverBenchmarkBase< Real, Index >::setup( parameters, Dimension ) )
         return false;

      auto mode = std::ios::out;
      if( this->outputMode == "append" )
         mode |= std::ios::app;

      std::ofstream logFile( this->logFileName.getString(), mode );
      TNL::Benchmarks::Benchmark<> benchmark(logFile, this->loops, this->verbose);

      // write global metadata into a separate file
      std::map< std::string, std::string > metadata = TNL::Benchmarks::getHardwareMetadata();
      TNL::Benchmarks::writeMapAsJson( metadata, this->logFileName, ".metadata.json" );

      auto precision = TNL::getType<Real>();
      TNL::String device;
      if( std::is_same< Device, TNL::Devices::Sequential >::value )
         device = "sequential";
      if( std::is_same< Device, TNL::Devices::Host >::value )
         device = "host";
      if( std::is_same< Device, TNL::Devices::Cuda >::value )
         device = "cuda";

      std::cout << "Heat equation benchmark in " << Dimension << "D : scheme = " << this->scheme() << ", precision = " << precision << ", device = " << device << std::endl;

      for( Index xSize = this->minXDimension; xSize <= this->maxXDimension; xSize *= this->xSizeStepFactor ) {
         benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns( {
            { "precision", precision },
            { "dimension", TNL::convertToString( 1 ) },
            { "xSize", TNL::convertToString( xSize ) },
            { "ySize", TNL::convertToString(     1 ) },
            { "zSize", TNL::convertToString(     1 ) },
            { "implementation", this->implementation }
         }));

         benchmark.setDatasetSize( xSize );
         this->init( xSize );
         if( this->writeData ) {
               TNL::String fileName = TNL::String( "initial-" )
                  + this->scheme() + "-"
                  + precision + "-"
                  + this->implementation + "-"
                  + TNL::convertToString( xSize) + ".gplt";
            writeGnuplot( fileName.data(), xSize );
         }
         auto lambda = [&]() { this->exec( xSize ); };
         benchmark.time<Device>(device, lambda );
         if( this->writeData ) {
               TNL::String fileName = TNL::String( "final-" )
                  + this->scheme() + "-"
                  + precision + "-"
                  + this->implementation + "-"
                  + TNL::convertToString( xSize) + ".gplt";
            writeGnuplot( fileName.data(), xSize );
         }
      }
      return true;
   }
};

template< typename Real,
          typename Device,
          typename Index >
struct HeatEquationSolverBenchmark< 2, Real, Device, Index > : public HeatEquationSolverBenchmarkBase< Real, Index >
{
   static constexpr int Dimension = 2;
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;

   virtual TNL::String scheme() = 0;

   virtual void init( const Index xSize, const Index ySize ) = 0;

   virtual void exec( const Index xSize, const Index ySize ) = 0;

   virtual bool writeGnuplot( const std::string &filename, const Index xSize, const Index ySize ) const = 0;

   void init( const Index xSize, const Index ySize, VectorType& ux, VectorType& aux )
   {
      ux.setSize( xSize * ySize );
      aux.setSize( xSize * ySize );

      ux = 0;
      aux = 0;

      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;

      auto uxView = ux.getView();
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

   template< typename Vector >
   bool writeGnuplot( const std::string &filename,
                      const Vector& u,
                      const Index xSize, const Index ySize ) const
   {
      std::ofstream out(filename, std::ios::out);
      if( !out.is_open() )
         return false;
      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;
      for( Index j = 0; j < ySize; j++) {
         for( Index i = 0; i < xSize; i++)
            out << i * hx - this->xDomainSize / 2. << " "
               << j * hy - this->yDomainSize / 2. << " "
               << u.getElement( j * xSize + i ) << std::endl;
         out << std::endl;
      }
      return out.good();
   }

   bool runBenchmark( const TNL::Config::ParameterContainer& parameters )
   {
      if( ! HeatEquationSolverBenchmarkBase< Real, Index >::setup( parameters, Dimension ) )
         return false;

      auto mode = std::ios::out;
      if( this->outputMode == "append" )
         mode |= std::ios::app;
      std::ofstream logFile( this->logFileName.getString(), mode );
      TNL::Benchmarks::Benchmark<> benchmark(logFile, this->loops, this->verbose);

      // write global metadata into a separate file
      std::map< std::string, std::string > metadata = TNL::Benchmarks::getHardwareMetadata();
      TNL::Benchmarks::writeMapAsJson( metadata, this->logFileName, ".metadata.json" );


      auto precision = TNL::getType<Real>();
      TNL::String device;
      if( std::is_same< Device, TNL::Devices::Sequential >::value )
         device = "sequential";
      if( std::is_same< Device, TNL::Devices::Host >::value )
         device = "host";
      if( std::is_same< Device, TNL::Devices::Cuda >::value )
         device = "cuda";

      std::cout << "Heat equation benchmark in " << Dimension << "D : scheme = " << this->scheme() << ", precision = " << precision << ", device = " << device << std::endl;

      for( Index xSize = this->minXDimension; xSize <= this->maxXDimension; xSize *= this->xSizeStepFactor ) {
         for( Index ySize = this->minYDimension; ySize <= this->maxYDimension; ySize *= this->ySizeStepFactor ) {
            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns( {
               { "precision", precision },
               { "dimension", TNL::convertToString( 2 ) },
               { "xSize", TNL::convertToString( xSize ) },
               { "ySize", TNL::convertToString( ySize ) },
               { "zSize", TNL::convertToString(     1 ) },
               { "implementation", this->implementation }
            }));

            benchmark.setDatasetSize( xSize * ySize );
            this->init( xSize, ySize );
            if( this->writeData ) {
               TNL::String fileName = TNL::String( "initial-" )
                  + this->scheme() + "-"
                  + precision + "-"
                  + this->implementation + "-"
                  + TNL::convertToString( xSize) + "-"
                  + TNL::convertToString( ySize ) + ".gplt";
               writeGnuplot( fileName.data(), xSize, ySize );
            }
            auto lambda = [&]() { this->exec( xSize, ySize ); };
            benchmark.time<Device>(device, lambda );
            if( this->writeData ) {
               TNL::String fileName = TNL::String( "final-" )
                  + this->scheme() + "-"
                  + precision + "-"
                  + this->implementation + "-"
                  + TNL::convertToString( xSize) + "-"
                  + TNL::convertToString( ySize ) + ".gplt";
               writeGnuplot( fileName.data(), xSize, ySize );
            }
         }
      }
      return true;
   }
};

template< typename Real,
          typename Device,
          typename Index >
struct HeatEquationSolverBenchmark< 3, Real, Device, Index > : public HeatEquationSolverBenchmarkBase< Real, Index >
{
   static constexpr int Dimension = 3;
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;

   virtual TNL::String scheme() = 0;

   virtual void init( const Index xSize, const Index ySize, const Index zSize ) = 0;

   virtual void exec( const Index xSize, const Index ySize, const Index zSize ) = 0;

   virtual bool writeGnuplot( const std::string& filename,
                              const Index xSize, const Index ySize, const Index zSize,
                              const Index zSlice ) const = 0;

   void init( const Index xSize, const Index ySize, const Index zSize, VectorType& ux, VectorType& aux )
   {
      ux.setSize( xSize * ySize * zSize );
      aux.setSize( xSize * ySize * zSize );

      ux = 0;
      aux = 0;

      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;
      const Real hz = this->zDomainSize / (Real) zSize;

      auto uxView = ux.getView();
      auto xDomainSize_ = this->xDomainSize;
      auto yDomainSize_ = this->yDomainSize;
      auto zDomainSize_ = this->zDomainSize;
      auto delta_ = this->delta;
      auto alpha_ = this->alpha;
      auto beta_ = this->beta;
      auto gamma_ = this->gamma;
      auto init = [=] __cuda_callable__( Index i, Index j, Index k) mutable
      {
         auto index = ( k * ySize + j ) * xSize + i;

         auto x = i * hx - xDomainSize_ / 2.0;
         auto y = j * hy - yDomainSize_ / 2.0;
         auto z = k * hz - zDomainSize_ / 2.0;
         uxView[index] = delta_ * ( 1.0 - TNL::sign( x*x / alpha_ + y*y / beta_ + z*z / gamma_ - 1.0 ) );
      };
      TNL::Algorithms::ParallelFor3D<Device>::exec( 1, 1, 1, xSize - 1, ySize - 1, zSize - 1, init );
   }

   template< typename Vector >
   bool writeGnuplot( const std::string &filename,
                      const Vector& u,
                      const Index xSize, const Index ySize, const Index zSize,
                      const Index zSlice ) const
   {
      std::ofstream out(filename, std::ios::out);
      if( !out.is_open() )
         return false;
      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;
      for( Index j = 0; j < ySize; j++) {
         for( Index i = 0; i < xSize; i++)
            out << i * hx - this->xDomainSize / 2. << " "
               << j * hy - this->yDomainSize / 2. << " "
               << u.getElement( ( zSlice * ySize  + j ) * xSize + i ) << std::endl;
         out << std::endl;
      }
      return out.good();
   }

   bool runBenchmark( const TNL::Config::ParameterContainer& parameters )
   {
      if( ! HeatEquationSolverBenchmarkBase< Real, Index >::setup( parameters, Dimension ) )
         return false;

      auto mode = std::ios::out;
      if( this->outputMode == "append" )
         mode |= std::ios::app;
      std::ofstream logFile( this->logFileName.getString(), mode );
      TNL::Benchmarks::Benchmark<> benchmark(logFile, this->loops, this->verbose);

      // write global metadata into a separate file
      std::map< std::string, std::string > metadata = TNL::Benchmarks::getHardwareMetadata();
      TNL::Benchmarks::writeMapAsJson( metadata, this->logFileName, ".metadata.json" );


      auto precision = TNL::getType<Real>();
      auto scheme = parameters.getParameter< TNL::String >( "scheme" );
      TNL::String device;
      if( std::is_same< Device, TNL::Devices::Sequential >::value )
         device = "sequential";
      if( std::is_same< Device, TNL::Devices::Host >::value )
         device = "host";
      if( std::is_same< Device, TNL::Devices::Cuda >::value )
         device = "cuda";

      std::cout << "Heat equation benchmark in " << Dimension << "D : scheme = " << this->scheme() << ", precision = " << precision << ", device = " << device << std::endl;

      for( Index xSize = this->minXDimension; xSize <= this->maxXDimension; xSize *= this->xSizeStepFactor ) {
         for( Index ySize = this->minYDimension; ySize <= this->maxYDimension; ySize *= this->ySizeStepFactor ) {
            for( Index zSize = this->minZDimension; zSize <= this->maxZDimension; zSize *= this->zSizeStepFactor ) {
               benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns( {
                  { "precision", precision },
                  { "scheme", scheme },
                  { "dimension", TNL::convertToString( 3 ) },
                  { "xSize", TNL::convertToString( xSize ) },
                  { "ySize", TNL::convertToString( ySize ) },
                  { "zSize", TNL::convertToString( zSize ) },
                  { "implementation", this->implementation }
               }));

               benchmark.setDatasetSize( xSize * ySize * zSize );
               this->init( xSize, ySize, zSize );
               if( this->writeData ) {
                  TNL::String fileName = TNL::String( "initial-" )
                     + this->scheme() + "-"
                     + precision + "-"
                     + this->implementation + "-"
                     + TNL::convertToString( xSize) + "-"
                     + TNL::convertToString( ySize ) + "-"
                     + TNL::convertToString( zSize ) + ".gplt";
                  writeGnuplot( fileName.data(), xSize, ySize, zSize, zSize/2 );
               }
               auto lambda = [&]() { this->exec( xSize, ySize, zSize ); };
               benchmark.time<Device>(device, lambda );
               if( this->writeData ) {
                  TNL::String fileName = TNL::String( "final-" )
                     + this->scheme() + "-"
                     + precision + "-"
                     + this->implementation + "-"
                     + TNL::convertToString( xSize) + "-"
                     + TNL::convertToString( ySize) + "-"
                     + TNL::convertToString( zSize ) + ".gplt";
                  writeGnuplot( fileName.data(), xSize, ySize, zSize, zSize/2 );
               }
            }
         }
      }
      return true;
   }
};
