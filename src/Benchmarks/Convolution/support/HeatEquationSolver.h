
#pragma once

#include "Solver.h"
#include "DummyTask.h"

#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Timer.h>

static std::vector< TNL::String > dimensionIds = { "grid-x-size", "grid-y-size" };
static std::vector< TNL::String > kernelSizeIds = { "kernel-x-size", "kernel-y-size" };
static std::vector< TNL::String > domainIds = { "domain-x-size", "domain-y-size" };
static std::vector< TNL::String > kernelDomainIds = { "kernel-domain-x-size", "kernel-domain-y-size" };

static std::string alphaKey = "alpha";
static std::string betaKey = "beta";
static std::string gammaKey = "gamma";

static std::string timeStepKey = "timeStep";
static std::string startTimeKey = "startTime";
static std::string finalTimeKey = "finalTime";
static std::string outputFilenamePrefix = "outputFilenamePrefix";

template< typename Real = double >
class HeatEquationSolver : public Solver< 2, TNL::Devices::Cuda >
{
public:
   constexpr static int Dimension = 2;
   using Device = TNL::Devices::Cuda;

   using Base = Solver< Dimension, Device >;
   using Vector = TNL::Containers::StaticVector< Dimension, int >;
   using Point = TNL::Containers::StaticVector< Dimension, Real >;
   using DataStore = TNL::Containers::Vector< Real, Device, int >;
   using HostDataStore = TNL::Containers::Vector< Real, TNL::Devices::Host, int >;

   virtual void
   start( const TNL::Config::ParameterContainer& parameters ) const override
   {
      int gridXSize = parameters.getParameter< int >( dimensionIds[ 0 ] );
      int gridYSize = parameters.getParameter< int >( dimensionIds[ 1 ] );

      int kernelXSize = parameters.getParameter< int >( kernelSizeIds[ 0 ] );
      int kernelYSize = parameters.getParameter< int >( kernelSizeIds[ 1 ] );

      Real xDomainSize = parameters.getParameter< Real >( domainIds[ 0 ] );
      Real yDomainSize = parameters.getParameter< Real >( domainIds[ 1 ] );

      Real kernelXDomainSize = parameters.getParameter< Real >( kernelDomainIds[ 0 ] );
      Real kernelYDomainSize = parameters.getParameter< Real >( kernelDomainIds[ 1 ] );

      Real hx = xDomainSize / (Real) gridXSize;
      Real hy = yDomainSize / (Real) gridYSize;

      Point domain = { xDomainSize, yDomainSize };
      Point kernelDomain = { kernelXDomainSize, kernelYDomainSize };
      Point spaceSteps = { hx, hy };

      Vector dimensions = { gridXSize, gridYSize };
      Vector kernelSize = { kernelXSize, kernelYSize };

      DataStore function = prepareFunction( parameters, dimensions, domain, spaceSteps );

      auto filenamePrefix = parameters.getParameter< TNL::String >( outputFilenamePrefix );
      auto initialFilename = filenamePrefix + "_initial.txt";

      if( ! writeGNUPlot( initialFilename, dimensions, spaceSteps, domain, function.getConstView() ) ) {
         std::cout << "Did fail during file write";
         return;
      }

      DataStore result;

      result.setLike( function );
      result = 0;

      auto timeStep = parameters.getParameter< double >( timeStepKey );
      auto startTime = parameters.getParameter< double >( startTimeKey );
      auto finalTime = parameters.getParameter< double >( finalTimeKey );

      int iteration = (startTime / timeStep) + 1;
      int finalIteration = finalTime / timeStep;

      double time = iteration * timeStep;

      for (int i = iteration; i <= finalIteration; i++) {
         printf("Time: %lf\n", time);

         convolve( dimensions, domain, kernelSize, kernelDomain, function.getConstView(), result.getView(), time );

         auto filename = TNL::String("data_") + TNL::convertToString(i) + ".txt";

         if( ! writeGNUPlot( filename, dimensions, spaceSteps, domain, result.getConstView() ) ) {
            std::cout << "Did fail during file write";
            return;
         }

         result = 0;

         time += timeStep;
      }
   }

   virtual TNL::Config::ConfigDescription
   makeInputConfig() const override
   {
      TNL::Config::ConfigDescription config = Base::makeInputConfig();

      config.addDelimiter( "Grid settings:" );
      config.addEntry< int >( dimensionIds[ 0 ], "Grid size along x-axis.", 200 );
      config.addEntry< int >( dimensionIds[ 1 ], "Grid size along y-axis.", 200 );

      config.addDelimiter( "Kernel settings:" );
      config.addEntry< int >( kernelSizeIds[ 0 ], "Kernel size along x-axis.", 3 );
      config.addEntry< int >( kernelSizeIds[ 1 ], "Kernel size along y-axis.", 3 );

      config.addDelimiter( "Problem settings:" );
      config.addEntry< TNL::String >( outputFilenamePrefix, "The prefix in name of the output file", "data" );

      config.addEntry< Real >( domainIds[ 0 ], "Domain size along x-axis.", 4.0 );
      config.addEntry< Real >( domainIds[ 1 ], "Domain size along y-axis.", 4.0 );

      config.addEntry< Real >( kernelDomainIds[ 0 ], "Kernel domain size along x-axis.", 4.0 );
      config.addEntry< Real >( kernelDomainIds[ 1 ], "Kernel domain size along y-axis.", 4.0 );

      config.addDelimiter( "Initial condition settings ( (x^2/alpha + y^2/beta) + gamma)):" );
      config.addEntry< Real >( alphaKey, "Alpha value in initial condition", -0.05 );
      config.addEntry< Real >( betaKey, "Beta value in initial condition", -0.05 );
      config.addEntry< Real >( gammaKey, "Gamma key in initial condition", 15 );

      config.addDelimiter( "Time settings:" );
      config.addEntry< Real >( startTimeKey, "Final time of the simulation.", 0.0);
      config.addEntry< Real >( timeStepKey, "Time step of the simulation.", 0.005);
      config.addEntry< Real >( finalTimeKey, "Final time of the simulation.", 0.36);

      return config;
   }

   DataStore
   prepareFunction( const TNL::Config::ParameterContainer& parameters,
                    const Vector& dimensions,
                    const Point& domain,
                    const Point& spaceSteps ) const
   {
      DataStore function;

      function.resize( dimensions.x() * dimensions.y() );

      auto functionView = function.getView();

      auto xDomainSize = parameters.getParameter< Real >( domainIds[ 0 ] );
      auto yDomainSize = parameters.getParameter< Real >( domainIds[ 1 ] );

      auto alpha = parameters.getParameter< Real >( alphaKey );
      auto beta = parameters.getParameter< Real >( betaKey );
      auto gamma = parameters.getParameter< Real >( gammaKey );

      auto init = [ = ] __cuda_callable__( int i, int j ) mutable
      {
         auto index = j * dimensions.x() + i;

         auto x = i * spaceSteps.x() - domain.x() / 2.;
         auto y = j * spaceSteps.y() - domain.y() / 2.;

         functionView[ index ] = TNL::max((x * x / alpha)  + (y * y / beta) + gamma, 0);
      };

      TNL::Algorithms::ParallelFor2D< Device >::exec( 0, 0, dimensions.x(), dimensions.y(), init );

      return function;
   }

   void
   convolve( const Vector& dimensions,
             const Point& domain,
             const Vector& kernelSize,
             const Point& kernelDomain,
             typename DataStore::ConstViewType input,
             typename DataStore::ViewType result,
             const Real time ) const
   {
      DataStore kernel;
      kernel.resize(kernelSize.x() * kernelSize.y());

      auto kernelView = kernel.getView();
      auto domainSpaceSteps = Point(domain.x() / dimensions.x(), domain.y() / dimensions.y());
      auto kernelSpaceSteps = Point(kernelDomain.x() / (kernelSize.x() - 1), kernelDomain.y() / (kernelSize.y() - 1));

      auto init = [ = ] __cuda_callable__( int i, int j ) mutable {
         auto index = j * kernelSize.x() + i;

         auto x = i * kernelSpaceSteps.x() - kernelDomain.x() / 2.;
         auto y = j * kernelSpaceSteps.y() - kernelDomain.y() / 2.;

         // The space step is given by the function domain
         // However, because the kernel is limited to 31x31 size
         // The user can specify it custom kernel domain from which values are taken
         kernelView[ index ] = domainSpaceSteps.x() * domainSpaceSteps.y() * ( (Real)1 / ( (Real)4 * M_PI * time ) ) * exp( - ( pow(x, 2.) + pow(y, 2.)  ) / ( (Real)4 * time ) );
      };

      TNL::Algorithms::ParallelFor2D< Device >::exec( 0, 0, kernelSize.x(), kernelSize.y(), init );

      // std::cout << std::endl << std::endl << std::endl;

      for (int i = 0; i < kernelSize.x(); i++) {
         for (int j = 0; j < kernelSize.y(); j++) {
            auto index = j * kernelSize.x() + i;

            printf("%lf ", kernelView.getElement(index));
         }

         printf("\n");
      }

      auto kernelConstView = kernel.getConstView();

      DummyTask<int, Real, Dimension, Device>::exec(dimensions, kernelSize, input, result, kernelConstView, 0);
   }

   bool
   writeGNUPlot( const std::string& filename,
                 const Vector& dimensions,
                 const Point& spaceSteps,
                 const Point& domain,
                 const typename DataStore::ConstViewType& map ) const
   {
      std::ofstream out( filename, std::ios::out );

      if( ! out.is_open() )
         return false;

      for( int j = 0; j < dimensions.y(); j++ )
         for( int i = 0; i < dimensions.x(); i++ )
            out << i * spaceSteps.x() - domain.x() / 2. << " "
                << j * spaceSteps.y() - domain.y() / 2. << " "
                << map.getElement( j * dimensions.x() + i ) << std::endl;

      return out.good();
   }
};
