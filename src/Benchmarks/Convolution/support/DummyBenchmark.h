
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
   using DataStore = TNL::Containers::Array< float, Device, int >;
   using Base = Benchmark< Dimension, Device >;
   using TNLBenchmark = typename Base::TNLBenchmark;

   virtual void
   start( TNLBenchmark& benchmark, const TNL::Config::ParameterContainer& parameters ) const override
   {
      Vector dimension;
      Vector minKernelSize;
      Vector maxKernelSize;

      for( int i = 0; i < Dimension; i++ ) {
         dimension[ i ] = parameters.getParameter< int >( dimensionIds[ i ] );
         minKernelSize[ i ] = parameters.getParameter< int >( minKernelSizeIds[ i ] );
         maxKernelSize[ i ] = parameters.getParameter< int >( maxKernelSizeIds[ i ] );

         TNL_ASSERT_GE( minKernelSize[ i ], 1, "Minimal kernel size must be a positive number" );
         TNL_ASSERT_EQ( minKernelSize[ i ] % 2, 1, "Minimal kernel size must be odd" );
         TNL_ASSERT_GT( maxKernelSize[ i ], minKernelSize[ i ], "End kernel size must be greater than start kernel size" );
      }

      int kernelStep = parameters.getParameter< int >( "kernel-step" );

      TNL_ASSERT_GT( kernelStep, 0, "Kernel step must be a positive number" );
      TNL_ASSERT_EQ( kernelStep % 2, 0, "Kernel step must be even" );

      TNL::String id = parameters.getParameter< TNL::String >( "id" );

      time( id, benchmark, dimension, minKernelSize, maxKernelSize, kernelStep );
   }

   virtual void
   time( const TNL::String& id,
         TNLBenchmark& benchmark,
         const Vector& dimension,
         const Vector& minKernelSize,
         const Vector& maxKernelSize,
         const int kernelStep ) const
   {
      Vector currentKernelSize = minKernelSize;

      do {
         timeConvolution( id, benchmark, dimension, currentKernelSize );

         currentKernelSize[ 0 ] += kernelStep;

         for( size_t i = 0; i < currentKernelSize.getSize() - 1; i++ ) {
            if( currentKernelSize[ i ] >= maxKernelSize[ i ] ) {
               currentKernelSize[ i ] = minKernelSize[ i ];
               currentKernelSize[ i + 1 ] += kernelStep;
            }
         }
      } while( currentKernelSize < maxKernelSize );
   }

   void
   timeConvolution( const TNL::String& id, TNLBenchmark& benchmark, const Vector& dimension, const Vector& kernelSize ) const
   {
      auto device = TNL::getType< Device >();

      typename TNLBenchmark::MetadataColumns columns = { { "id", id } };

      size_t elementsCount = 1;
      size_t kernelElementsCount = 1;

      for( size_t i = 0; i < dimension.getSize(); i++ ) {
         elementsCount *= dimension[ i ];
         kernelElementsCount *= kernelSize[ i ];

         columns.push_back( { dimensionIds[ i ], TNL::convertToString( dimension[ i ] ) } );
         columns.push_back( { kernelSizeIds[ i ], TNL::convertToString( kernelSize[ i ] ) } );
      }

      benchmark.setDatasetSize( ( elementsCount * 4 ) / 1.e9, 1.0 );
      benchmark.setMetadataColumns( columns );

      // Setup input data
      DataStore input, result, kernel;

      input.resize( elementsCount );
      result.resize( elementsCount );
      kernel.resize( kernelElementsCount );

      input = 1;
      result = 1;
      kernel = 1;

      auto inputView = input.getView();
      auto resultView = result.getView();
      auto kernelView = kernel.getView();

      auto measure = [ & ]()
      {
         DummyTask< int, float, Dimension, Device >::exec( dimension, kernelSize, inputView, resultView, kernelView );
      };

      benchmark.template time< Device >( device, measure );
   }

   TNL::Config::ConfigDescription
   makeInputConfig() const override
   {
      TNL::Config::ConfigDescription config = Base::makeInputConfig();

      config.addDelimiter( "Grid dimension settings:" );

      for( int i = 0; i < Dimension; i++ )
         config.addEntry< int >( dimensionIds[ i ], dimensionIds[ i ], 16 );

      config.addDelimiter( "Kernel settings:" );

      for( int i = 0; i < Dimension; i++ )
         config.addEntry< int >( minKernelSizeIds[ i ], minKernelSizeIds[ i ] + " (odd) :", 1 );

      for( int i = 0; i < Dimension; i++ )
         config.addEntry< int >( maxKernelSizeIds[ i ], maxKernelSizeIds[ i ] + " (odd) :", 11 );

      config.addEntry< int >( "kernel-step", "Step of kernel increase which is added to kernel (must be even)", 2 );

      return config;
   }
};
