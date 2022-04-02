
#pragma once

#include "Benchmark.h"
#include "DummyTask.h"

static std::vector< TNL::String > minDimensionIds = { "min-x-dimension", "min-y-dimension", "min-z-dimension" };
static std::vector< TNL::String > dimensionIds = { "x-dimension", "y-dimension", "z-dimension" };
static std::vector< TNL::String > maxDimensionIds = { "max-x-dimension", "max-y-dimension", "max-z-dimension" };
static std::vector< TNL::String > minKernelSizeIds = { "min-kernel-width", "min-kernel-height", "min-kernel-depth" };
static std::vector< TNL::String > kernelSizeIds = { "x-kernelSize", "y-kernelSize", "z-kernelSize" };
static std::vector< TNL::String > maxKernelSizeIds = { "max-kernel-width", "max-kernel-height", "max-kernel-depth" };

template< int Dimension, typename Device >
class DummyBenchmark : public Benchmark< Dimension, Device >
{
public:
   using Vector = TNL::Containers::StaticVector< Dimension, int >;
   using DataStore = TNL::Containers::Array< int, Device, float >;
   using Benchmark = Base::Benchmark;
   using Base = Benchmark< Dimension, Device >;

   virtual void
   start( const Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters ) const override
   {
      Vector start;
      Vector end;
      Vector minKernelSize;
      Vector maxKernelSize;

      for( int i = 0; i < Dimension; i++ ) {
         start[ i ] = parameters.getParameter< int >( minDimensionIds[ i ] );
         end[ i ] = parameters.getParameter< int >( maxDimensionIds[ i ] );
         minKernelSize[ i ] = parameters.getParameter< int >( minKernelSizeIds[ i ] );
         maxKernelSizeIds[ i ] = parameters.getParameter< int >( maxKernelSizeIds[ i ] );

         TNL_ASSERT_GT( start[ i ], 1, "Start dimension must be positive integer" );
         TNL_ASSERT_GT( end[ i ], start[ i ], "End dimension must be greater than start dimension" );

         TNL_ASSERT_GE( minKernelSize[ i ], 1, "Minimal kernel size must be a positive number" );
         TNL_ASSERT_EQ( minKernelSize[ i ] % 2, 1, "Minimal kernel size must be odd" );
         TNL_ASSERT_GT( end[ i ], start[ i ], "End kernel size must be greater than start kernel size" );
      }

      int dimensionStep = parameters.getParameter< int >( "dimension-step" );
      int kernelStep = parameters.getParameter< int >( "kernel-step" );

      TNL_ASSERT_GT( dimensionStep, 1, "Dimension step must be a positive number" );
      TNL_ASSERT_GT( kernelStep, 0, "Kernel step must be a positive number" );
      TNL_ASSERT_EQ( kernelStep % 2, 0, "Kernel step must be even" );

      time( benchmark, start, end, dimensionStep, minKernelSize, maxKernelSize, kernelStep );
   }

   virtual void
   time( Benchmark& bencmark,
         const Vector& minDimension,
         const Vector& maxDimension,
         const int dimensionStep,
         const Vector& minKernelSize,
         const Vector& maxKernelSize,
         const int kernelStep ) const
   {
      Vector currentDimension = minDimension;
      Vector currentKernelSize;

      do {
         currentKernelSize = minKernelSize;

         do {
            time( benchmark, currentDimension, currentKernelSize );

            currentKernelSize[ 0 ] += kernelStep;

            for( size_t i = 0; i < currentKernelSize.getSize() - 1; i++ ) {
               if( currentKernelSize[ i ] >= maxKernelSize[ i ] ) {
                  currentKernelSize[ i ] = minKernelSize[ i ];
                  maxKernelSize[ i + 1 ] += kernelStep;
               }
            }
         } while( currentKernelSize < maxKernelSize );

         currentDimension[ 0 ] *= dimensionStep;

         for( size_t i = 0; i < currentDimension.getSize() - 1; i++ ) {
            if( currentDimension[ i ] >= maxDimension[ i ] ) {
               currentDimension[ i ] = minDimension[ i ];
               maxDimension[ i ] = maxDimension[ i ];
            }
         }

      } while( currentDimension < maxDimension );
   }

   void
   timeConvolution( Benchmark& benchmark, const Vector& dimension, const Vector& kernelSize ) const
   {
      auto device = TNL::getType< Device >();

      Benchmark::MetadataColumns columns = {};

      size_t elementsCount = 1;
      size_t kernelElementsCount = 1;

      for( size_t i = 0; i < dimension.getSize(); i++ ) {
         elementsCount *= dimension[ i ];
         kernelElementsCount *= kernelSize[ i ];

         columns.insert( { dimensionIds[ i ], dimension[ i ] } );
         columns.insert( { kernelSizeIds[ i ], kernelSize[ i ] } );
      }

      benchmark.setDatasetSize( ( elementsCount * 4 ) / 1.e9, 1.0 );

      // Setup input data
      DataStore input, result, kernel;

      input.resize( elementsCount );
      result.resize( elementsCount );
      kernel.resize( kernelSize );

      input = 1;
      result = 1;
      kernel = 1;

      auto inputView = input.getView();
      auto resultView = result.getView();
      auto kernelView = kernel.getView();

      auto measure = [ & ]()
      {
         DummyTask<Dimension, Device>::exec(dimension, kernelSize, inputView, resultView, kernelView);
      };

      benchmark.time< Device >( device, measure );
   }

   TNL::Config::ConfigDescription
   makeInputConfig() const override
   {
      auto config = Base::makeInputConfig();

      config.addDelimiter( "Grid dimension settings:" );

      for( int i = 0; i < Dimension; i++ )
         config.addEntry< int >( minDimensionIds[ i ], minDimensionIds[ i ], 512 );

      for( int i = 0; i < Dimension; i++ )
         config.addEntry< int >( maxDimensionIds[ i ], maxDimensionIds[ i ], 512 );

      config.addEntry< int >( "dimension-step", "Step of kernel increase by which dimension is multiplied (must be even)", 2 );

      config.addDelimiter( "Kernel settings:" );

      for( int i = 0; i < Dimension; i++ )
         config.addEntry< int >( minKernelSizeIds[ i ], minKernelSizeIds[ i ] + " (odd) :", 1 );

      for( int i = 0; i < Dimension; i++ )
         config.addEntry< int >( minKernelSizeIds[ i ], minKernelSizeIds[ i ] + " (odd) :", 11 );

      config.addEntry< int >( "kernel-step", "Step of kernel increase which is added to kernel (must be even)", 2 );

      return config;
   }
};
