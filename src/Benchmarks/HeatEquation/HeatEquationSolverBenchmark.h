// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Benchmarks/Devices.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Containers/Vector.h>

template< typename Real = double, typename Index = int >
struct HeatEquationSolverBenchmarkBase
{
   static void
   configSetup( TNL::Config::ConfigDescription& config )
   {
      config.addDelimiter( "Benchmark settings:" );
      config.addEntry< bool >( "write-data", "Write initial condition and final state to a file.", false );

      config.addDelimiter( "Problem settings:" );
      config.addEntry< int >( "dimension", "Dimension of the benchmark problem.", 2 );
      config.addEntry< int >( "min-x-dimension", "Minimum dimension over x axis used in the benchmark.", 100 );
      config.addEntry< int >( "max-x-dimension", "Maximum dimension over x axis used in the benchmark.", 200 );
      config.addEntry< int >(
         "x-size-step-factor",
         "Factor determining the dimension grows over x axis. First size is min-x-dimension and each "
         "following size is stepFactor*previousSize, up to max-x-dimension.",
         2 );

      config.addEntry< int >( "min-y-dimension", "Minimum dimension over y axis used in the benchmark.", 100 );
      config.addEntry< int >( "max-y-dimension", "Maximum dimension over y axis used in the benchmark.", 200 );
      config.addEntry< int >(
         "y-size-step-factor",
         "Factor determining the dimension grows over y axis. First size is min-y-dimension and each "
         "following size is stepFactor*previousSize, up to max-y-dimension.",
         2 );

      config.addEntry< int >( "min-z-dimension", "Minimum dimension over z axis used in the benchmark.", 100 );
      config.addEntry< int >( "max-z-dimension", "Maximum dimension over z axis used in the benchmark.", 200 );
      config.addEntry< int >(
         "z-size-step-factor",
         "Factor determining the dimension grows over z axis. First size is min-z-dimension and each "
         "following size is stepFactor*previousSize, up to max-z-dimension.",
         2 );

      config.addEntry< double >( "domain-x-size", "Domain size along x-axis.", 2.0 );
      config.addEntry< double >( "domain-y-size", "Domain size along y-axis.", 2.0 );
      config.addEntry< double >( "domain-z-size", "Domain size along z-axis.", 2.0 );

      config.addDelimiter( "Initial condition settings delta * exp( -( x^2/alpha + y^2/beta + z^2/gamma ) ):" );
      config.addEntry< double >( "alpha", "Alpha value in initial condition", 0.02 );
      config.addEntry< double >( "beta", "Beta value in initial condition", 0.02 );
      config.addEntry< double >( "gamma", "Gamma value in initial condition", 0.02 );
      config.addEntry< double >( "delta", "Delta value in initial condition", 0.25 );

      config.addEntry< double >( "time-step", "Time step. By default it is proportional to one over space step square.", 0.0 );
      config.addEntry< double >( "final-time", "Final time of the simulation.", 0.01 );
      config.addEntry< int >( "max-iterations", "Maximum time iterations.", 0 );
      config.addEntry< bool >(
         "check-solution",
         "Compute the maximal difference from the exact solution of the heat equation in the whole space "
         "after one extra run and add it to the log.",
         false );
   }

   /**
    * \brief Returns the initial condition `delta * exp( -( x^2/alpha + y^2/beta + z^2/gamma ) )`.
    *
    * The same function is used by all implementations, so that their results can be compared.
    */
   __cuda_callable__
   static Real
   getInitialCondition( Real x, Real y, Real z, Real alpha, Real beta, Real gamma, Real delta )
   {
      return delta * TNL::exp( -( x * x / alpha + y * y / beta + z * z / gamma ) );
   }

   /**
    * \brief Returns the exact solution of the heat equation in the whole space with the initial condition
    * \ref getInitialCondition at the time `t`.
    *
    * The solution is the product of one-dimensional Gaussian functions, each of them spreads independently:
    * `delta * prod_i sqrt( a_i / ( a_i + 4t ) ) * exp( -x_i^2 / ( a_i + 4t ) )` for `a_i` = `alpha`, `beta`, `gamma`.
    * It approximates the solution with the homogeneous Dirichlet boundary conditions while the function is negligible
    * on the boundary.
    */
   double
   getExactSolution( int dimension, double x, double y, double z, double t ) const
   {
      const double coordinates[ 3 ] = { x, y, z };
      const double a[ 3 ] = { (double) alpha, (double) beta, (double) gamma };
      double result = delta;
      for( int i = 0; i < dimension; i++ )
         result *= std::sqrt( a[ i ] / ( a[ i ] + 4.0 * t ) )
                 * std::exp( -coordinates[ i ] * coordinates[ i ] / ( a[ i ] + 4.0 * t ) );
      return result;
   }

   /**
    * \brief Returns the time reached by the time loop of the solvers with the time step `tau`.
    *
    * The loop is the same as in the solvers, including the accumulation of the time in the precision `Real`.
    */
   Real
   getReachedTime( Real tau ) const
   {
      Real time = 0;
      Index iterations = 0;
      while( time < finalTime && ( ! maxIterations || iterations < maxIterations ) ) {
         time += tau;
         iterations++;
      }
      return time;
   }

   /**
    * \brief Returns the maximal difference of the solution `u` from the exact solution \ref getExactSolution.
    *
    * The time is the time reached by the solvers with their default time step (one tenth of the smallest
    * space step squared) or with the time step given by the user.
    */
   template< typename Vector >
   double
   getMaxError( const Vector& u, int dimension, Index xSize, Index ySize, Index zSize ) const
   {
      const TNL::Containers::Vector< Real, TNL::Devices::Host, Index > hostU( u );
      const Real hx = xDomainSize / (Real) xSize;
      const Real hy = yDomainSize / (Real) ySize;
      const Real hz = zDomainSize / (Real) zSize;
      Real hMin = hx;
      if( dimension > 1 )
         hMin = std::min( hMin, hy );
      if( dimension > 2 )
         hMin = std::min( hMin, hz );
      const Real tau = timeStep ? timeStep : 0.1 * hMin * hMin;
      const double t = getReachedTime( tau );
      double error = 0;
      for( Index k = 0; k < zSize; k++ )
         for( Index j = 0; j < ySize; j++ )
            for( Index i = 0; i < xSize; i++ ) {
               const double x = i * hx - xDomainSize / 2.0;
               const double y = dimension > 1 ? j * hy - yDomainSize / 2.0 : 0.0;
               const double z = dimension > 2 ? k * hz - zDomainSize / 2.0 : 0.0;
               const double value = hostU[ ( k * ySize + j ) * xSize + i ];
               error = std::max( error, std::abs( value - getExactSolution( dimension, x, y, z, t ) ) );
            }
      return error;
   }

   bool
   setup( const TNL::Config::ParameterContainer& parameters, int dimension )
   {
      this->implementation = parameters.getParameter< TNL::String >( "implementation" );
      this->writeData = parameters.getParameter< bool >( "write-data" );

      this->minXDimension = parameters.getParameter< int >( "min-x-dimension" );
      this->maxXDimension = parameters.getParameter< int >( "max-x-dimension" );
      this->xSizeStepFactor = parameters.getParameter< int >( "x-size-step-factor" );

      if( xSizeStepFactor <= 1 ) {
         std::cerr << "The value of --x-size-step-factor must be greater than 1." << std::endl;
         return false;
      }

      this->minYDimension = parameters.getParameter< int >( "min-y-dimension" );
      this->maxYDimension = parameters.getParameter< int >( "max-y-dimension" );
      this->ySizeStepFactor = parameters.getParameter< int >( "y-size-step-factor" );

      if( dimension > 1 && ySizeStepFactor <= 1 ) {
         std::cerr << "The value of --y-size-step-factor must be greater than 1." << std::endl;
         return false;
      }

      this->minZDimension = parameters.getParameter< int >( "min-z-dimension" );
      this->maxZDimension = parameters.getParameter< int >( "max-z-dimension" );
      this->zSizeStepFactor = parameters.getParameter< int >( "z-size-step-factor" );

      if( dimension > 2 && zSizeStepFactor <= 1 ) {
         std::cerr << "The value of --z-size-step-factor must be greater than 1." << std::endl;
         return false;
      }

      this->xDomainSize = parameters.getParameter< Real >( "domain-x-size" );
      this->yDomainSize = parameters.getParameter< Real >( "domain-y-size" );
      this->zDomainSize = parameters.getParameter< Real >( "domain-z-size" );
      this->alpha = parameters.getParameter< Real >( "alpha" );
      this->beta = parameters.getParameter< Real >( "beta" );
      this->gamma = parameters.getParameter< Real >( "gamma" );
      this->delta = parameters.getParameter< Real >( "delta" );
      this->timeStep = parameters.getParameter< Real >( "time-step" );
      this->finalTime = parameters.getParameter< Real >( "final-time" );
      this->maxIterations = parameters.getParameter< int >( "max-iterations" );
      this->checkSolution = parameters.getParameter< bool >( "check-solution" );

      return true;
   }

protected:
   TNL::String implementation;
   bool writeData;
   bool checkSolution = false;

   Index minXDimension, maxXDimension, xSizeStepFactor;
   Index minYDimension, maxYDimension, ySizeStepFactor;
   Index minZDimension, maxZDimension, zSizeStepFactor;

   Real xDomainSize = 0.0, yDomainSize = 0.0, zDomainSize = 0.0;
   Real alpha = 0.0, beta = 0.0, gamma = 0.0, delta = 0.0;

   Real timeStep = 0.0, finalTime = 0.0;
   bool outputData = false;
   Index maxIterations = 0;
};

template< int Dimension = 2, typename Real = double, typename Device = TNL::Devices::Host, typename Index = int >
struct HeatEquationSolverBenchmark;

template< typename Real, typename Device, typename Index >
struct HeatEquationSolverBenchmark< 1, Real, Device, Index > : public HeatEquationSolverBenchmarkBase< Real, Index >
{
   static constexpr int Dimension = 1;
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;

   virtual TNL::String
   scheme() = 0;

   virtual void
   init( const Index xSize ) = 0;

   virtual void
   exec( const Index xSize ) = 0;

   //! \brief Copies the current solution to `u`, with the first index changing fastest.
   virtual void
   getSolution( VectorType& u ) const = 0;

   void
   init( const Index xSize, VectorType& ux, VectorType& aux )
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
      auto init = [ = ] __cuda_callable__( Index i ) mutable
      {
         auto x = i * hx - xDomainSize_ / 2.0;
         uxView[ i ] = HeatEquationSolverBenchmarkBase< Real, Index >::getInitialCondition( x, 0, 0, alpha_, 1, 1, delta_ );
      };
      TNL::Algorithms::parallelFor< Device >( 1, xSize - 1, init );
   }

   template< typename Vector >
   bool
   writeGnuplot( const std::string& filename, const Vector& u, const Index xSize ) const
   {
      std::ofstream out( filename, std::ios::out );
      if( ! out.is_open() )
         return false;
      const Real hx = this->xDomainSize / (Real) xSize;
      for( Index i = 0; i < xSize; i++ )
         out << i * hx - this->xDomainSize / 2. << " " << u.getElement( i ) << std::endl;
      return out.good();
   }

   bool
   writeGnuplot( const std::string& filename, const Index xSize ) const
   {
      VectorType u;
      getSolution( u );
      const TNL::Containers::Vector< Real, TNL::Devices::Host, Index > hostU( u );
      return writeGnuplot( filename, hostU, xSize );
   }

   bool
   runBenchmark( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
   {
      if( ! HeatEquationSolverBenchmarkBase< Real, Index >::setup( parameters, Dimension ) )
         return false;

      auto precision = TNL::getType< Real >();
      auto scheme = parameters.getParameter< TNL::String >( "scheme" );
      const std::string device = TNL::Benchmarks::getDeviceName< Device >();

      std::cout << "Heat equation benchmark in " << Dimension << "D : scheme = " << this->scheme()
                << ", precision = " << precision << ", device = " << device << std::endl;

      for( Index xSize = this->minXDimension; xSize <= this->maxXDimension; xSize *= this->xSizeStepFactor ) {
         TNL::Benchmarks::Benchmark::MetadataColumns metadata(
            { { "precision", precision },
              { "scheme", scheme },
              { "dimension", TNL::convertToString( 1 ) },
              { "x size", TNL::convertToString( xSize ) },
              { "y size", TNL::convertToString( 1 ) },
              { "z size", TNL::convertToString( 1 ) },
              { "implementation", this->implementation } } );
         if( this->checkSolution ) {
            this->init( xSize );
            this->exec( xSize );
            VectorType u;
            getSolution( u );
            metadata.emplace_back( "max error", TNL::convertToString( this->getMaxError( u, Dimension, xSize, 1, 1 ) ) );
         }
         benchmark.setMetadataColumns( metadata );

         benchmark.setDatasetSize( xSize * sizeof( Real ) );
         this->init( xSize );
         if( this->writeData ) {
            TNL::String fileName = TNL::String( "initial-" ) + this->scheme() + "-" + precision + "-" + this->implementation
                                 + "-" + TNL::convertToString( xSize ) + ".gplt";
            writeGnuplot( fileName.data(), xSize );
         }
         auto lambda = [ & ]()
         {
            this->exec( xSize );
         };
         benchmark.time< Device >( "TNL", lambda );
         if( this->writeData ) {
            TNL::String fileName = TNL::String( "final-" ) + this->scheme() + "-" + precision + "-" + this->implementation + "-"
                                 + TNL::convertToString( xSize ) + ".gplt";
            writeGnuplot( fileName.data(), xSize );
         }
      }
      return true;
   }
};

template< typename Real, typename Device, typename Index >
struct HeatEquationSolverBenchmark< 2, Real, Device, Index > : public HeatEquationSolverBenchmarkBase< Real, Index >
{
   static constexpr int Dimension = 2;
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;
   using CoordinatesType = TNL::Containers::StaticArray< Dimension, Index >;

   virtual TNL::String
   scheme() = 0;

   virtual void
   init( const Index xSize, const Index ySize ) = 0;

   virtual void
   exec( const Index xSize, const Index ySize ) = 0;

   //! \brief Copies the current solution to `u`, with the first index changing fastest.
   virtual void
   getSolution( VectorType& u ) const = 0;

   void
   init( const Index xSize, const Index ySize, VectorType& ux, VectorType& aux )
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
      auto delta_ = this->delta;
      auto alpha_ = this->alpha;
      auto beta_ = this->beta;
      auto init = [ = ] __cuda_callable__( const CoordinatesType& idx ) mutable
      {
         const Index& i = idx.x();
         const Index& j = idx.y();
         auto index = j * xSize + i;

         auto x = i * hx - xDomainSize_ / 2.0;
         auto y = j * hy - yDomainSize_ / 2.0;
         uxView[ index ] =
            HeatEquationSolverBenchmarkBase< Real, Index >::getInitialCondition( x, y, 0, alpha_, beta_, 1, delta_ );
      };
      TNL::Algorithms::parallelFor< Device >( CoordinatesType{ 1, 1 }, CoordinatesType{ xSize - 1, ySize - 1 }, init );
   }

   bool
   writeGnuplot( const std::string& filename, const Index xSize, const Index ySize ) const
   {
      VectorType u;
      getSolution( u );
      const TNL::Containers::Vector< Real, TNL::Devices::Host, Index > hostU( u );
      return writeGnuplot( filename, hostU, xSize, ySize );
   }

   template< typename Vector >
   bool
   writeGnuplot( const std::string& filename, const Vector& u, const Index xSize, const Index ySize ) const
   {
      std::ofstream out( filename, std::ios::out );
      if( ! out.is_open() )
         return false;
      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;
      for( Index j = 0; j < ySize; j++ ) {
         for( Index i = 0; i < xSize; i++ )
            out << i * hx - this->xDomainSize / 2. << " " << j * hy - this->yDomainSize / 2. << " "
                << u.getElement( j * xSize + i ) << std::endl;
         out << std::endl;
      }
      return out.good();
   }

   bool
   runBenchmark( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
   {
      if( ! HeatEquationSolverBenchmarkBase< Real, Index >::setup( parameters, Dimension ) )
         return false;

      auto precision = TNL::getType< Real >();
      auto scheme = parameters.getParameter< TNL::String >( "scheme" );
      const std::string device = TNL::Benchmarks::getDeviceName< Device >();

      std::cout << "Heat equation benchmark in " << Dimension << "D : scheme = " << this->scheme()
                << ", precision = " << precision << ", device = " << device << std::endl;

      for( Index xSize = this->minXDimension; xSize <= this->maxXDimension; xSize *= this->xSizeStepFactor ) {
         for( Index ySize = this->minYDimension; ySize <= this->maxYDimension; ySize *= this->ySizeStepFactor ) {
            TNL::Benchmarks::Benchmark::MetadataColumns metadata(
               { { "precision", precision },
                 { "scheme", scheme },
                 { "dimension", TNL::convertToString( 2 ) },
                 { "x size", TNL::convertToString( xSize ) },
                 { "y size", TNL::convertToString( ySize ) },
                 { "z size", TNL::convertToString( 1 ) },
                 { "implementation", this->implementation } } );
            if( this->checkSolution ) {
               this->init( xSize, ySize );
               this->exec( xSize, ySize );
               VectorType u;
               getSolution( u );
               metadata.emplace_back( "max error", TNL::convertToString( this->getMaxError( u, Dimension, xSize, ySize, 1 ) ) );
            }
            benchmark.setMetadataColumns( metadata );

            benchmark.setDatasetSize( xSize * ySize * sizeof( Real ) );
            this->init( xSize, ySize );
            if( this->writeData ) {
               TNL::String fileName = TNL::String( "initial-" ) + this->scheme() + "-" + precision + "-" + this->implementation
                                    + "-" + TNL::convertToString( xSize ) + "-" + TNL::convertToString( ySize ) + ".gplt";
               writeGnuplot( fileName.data(), xSize, ySize );
            }
            auto lambda = [ & ]()
            {
               this->exec( xSize, ySize );
            };
            benchmark.time< Device >( "TNL", lambda );
            if( this->writeData ) {
               TNL::String fileName = TNL::String( "final-" ) + this->scheme() + "-" + precision + "-" + this->implementation
                                    + "-" + TNL::convertToString( xSize ) + "-" + TNL::convertToString( ySize ) + ".gplt";
               writeGnuplot( fileName.data(), xSize, ySize );
            }
         }
      }
      return true;
   }
};

template< typename Real, typename Device, typename Index >
struct HeatEquationSolverBenchmark< 3, Real, Device, Index > : public HeatEquationSolverBenchmarkBase< Real, Index >
{
   static constexpr int Dimension = 3;
   using VectorType = TNL::Containers::Vector< Real, Device, Index >;
   using CoordinatesType = TNL::Containers::StaticArray< Dimension, Index >;

   virtual TNL::String
   scheme() = 0;

   virtual void
   init( const Index xSize, const Index ySize, const Index zSize ) = 0;

   virtual void
   exec( const Index xSize, const Index ySize, const Index zSize ) = 0;

   //! \brief Copies the current solution to `u`, with the first index changing fastest.
   virtual void
   getSolution( VectorType& u ) const = 0;

   bool
   writeGnuplot( const std::string& filename, const Index xSize, const Index ySize, const Index zSize, const Index zSlice )
      const
   {
      VectorType u;
      getSolution( u );
      const TNL::Containers::Vector< Real, TNL::Devices::Host, Index > hostU( u );
      return writeGnuplot( filename, hostU, xSize, ySize, zSize, zSlice );
   }

   void
   init( const Index xSize, const Index ySize, const Index zSize, VectorType& ux, VectorType& aux )
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
      auto init = [ = ] __cuda_callable__( const CoordinatesType& idx ) mutable
      {
         const Index& i = idx[ 0 ];
         const Index& j = idx[ 1 ];
         const Index& k = idx[ 2 ];
         auto index = ( k * ySize + j ) * xSize + i;

         auto x = i * hx - xDomainSize_ / 2.0;
         auto y = j * hy - yDomainSize_ / 2.0;
         auto z = k * hz - zDomainSize_ / 2.0;
         uxView[ index ] =
            HeatEquationSolverBenchmarkBase< Real, Index >::getInitialCondition( x, y, z, alpha_, beta_, gamma_, delta_ );
      };
      TNL::Algorithms::parallelFor< Device >(
         CoordinatesType{ 1, 1, 1 }, CoordinatesType{ xSize - 1, ySize - 1, zSize - 1 }, init );
   }

   template< typename Vector >
   bool
   writeGnuplot(
      const std::string& filename,
      const Vector& u,
      const Index xSize,
      const Index ySize,
      const Index zSize,
      const Index zSlice ) const
   {
      std::ofstream out( filename, std::ios::out );
      if( ! out.is_open() )
         return false;
      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;
      for( Index j = 0; j < ySize; j++ ) {
         for( Index i = 0; i < xSize; i++ )
            out << i * hx - this->xDomainSize / 2. << " " << j * hy - this->yDomainSize / 2. << " "
                << u.getElement( ( zSlice * ySize + j ) * xSize + i ) << std::endl;
         out << std::endl;
      }
      return out.good();
   }

   bool
   runBenchmark( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
   {
      if( ! HeatEquationSolverBenchmarkBase< Real, Index >::setup( parameters, Dimension ) )
         return false;

      auto precision = TNL::getType< Real >();
      auto scheme = parameters.getParameter< TNL::String >( "scheme" );
      const std::string device = TNL::Benchmarks::getDeviceName< Device >();

      std::cout << "Heat equation benchmark in " << Dimension << "D : scheme = " << this->scheme()
                << ", precision = " << precision << ", device = " << device << std::endl;

      for( Index xSize = this->minXDimension; xSize <= this->maxXDimension; xSize *= this->xSizeStepFactor ) {
         for( Index ySize = this->minYDimension; ySize <= this->maxYDimension; ySize *= this->ySizeStepFactor ) {
            for( Index zSize = this->minZDimension; zSize <= this->maxZDimension; zSize *= this->zSizeStepFactor ) {
               TNL::Benchmarks::Benchmark::MetadataColumns metadata(
                  { { "precision", precision },
                    { "scheme", scheme },
                    { "dimension", TNL::convertToString( 3 ) },
                    { "x size", TNL::convertToString( xSize ) },
                    { "y size", TNL::convertToString( ySize ) },
                    { "z size", TNL::convertToString( zSize ) },
                    { "implementation", this->implementation } } );
               if( this->checkSolution ) {
                  this->init( xSize, ySize, zSize );
                  this->exec( xSize, ySize, zSize );
                  VectorType u;
                  getSolution( u );
                  metadata.emplace_back(
                     "max error", TNL::convertToString( this->getMaxError( u, Dimension, xSize, ySize, zSize ) ) );
               }
               benchmark.setMetadataColumns( metadata );

               benchmark.setDatasetSize( xSize * ySize * zSize * sizeof( Real ) );
               this->init( xSize, ySize, zSize );
               if( this->writeData ) {
                  TNL::String fileName = TNL::String( "initial-" ) + this->scheme() + "-" + precision + "-"
                                       + this->implementation + "-" + TNL::convertToString( xSize ) + "-"
                                       + TNL::convertToString( ySize ) + "-" + TNL::convertToString( zSize ) + ".gplt";
                  writeGnuplot( fileName.data(), xSize, ySize, zSize, zSize / 2 );
               }
               auto lambda = [ & ]()
               {
                  this->exec( xSize, ySize, zSize );
               };
               benchmark.time< Device >( "TNL", lambda );
               if( this->writeData ) {
                  TNL::String fileName = TNL::String( "final-" ) + this->scheme() + "-" + precision + "-" + this->implementation
                                       + "-" + TNL::convertToString( xSize ) + "-" + TNL::convertToString( ySize ) + "-"
                                       + TNL::convertToString( zSize ) + ".gplt";
                  writeGnuplot( fileName.data(), xSize, ySize, zSize, zSize / 2 );
               }
            }
         }
      }
      return true;
   }
};
