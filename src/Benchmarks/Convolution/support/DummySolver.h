
#pragma once

#include "Solver.h"
#include "DummyTask.h"

static std::vector< TNL::String > dimensionIds = { "x-dimension", "y-dimension", "z-dimension" };
static std::vector< TNL::String > kernelSizeIds = { "x-kernel-size", "y-kernel-size", "z-kernel-size" };

template< int Dimension, typename Device >
class DummySolver : public Solver< Dimension, Device >
{
public:
   using Base = Solver< Dimension, Device >;
   using Vector = TNL::Containers::StaticVector< Dimension, int >;
   using DataStore = TNL::Containers::Array< float, Device, int >;

   virtual void
   start( const TNL::Config::ParameterContainer& parameters ) const override
   {
      Vector dimensions;
      Vector kernelSize;

      for( int i = 0; i < Dimension; i++ ) {
         dimensions[ i ] = parameters.getParameter< int >( dimensionIds[ i ] );
         kernelSize[ i ] = parameters.getParameter< int >( kernelSizeIds[ i ] );

         TNL_ASSERT_GT( dimensions[ i ], 1, "Start dimension must be positive integer" );

         TNL_ASSERT_GE( kernelSize[ i ], 1, "Minimal kernel size must be a positive number" );
         TNL_ASSERT_EQ( kernelSize[ i ] % 2, 1, "Minimal kernel size must be odd" );
      }

      launchConvolution( dimensions, kernelSize );
   }

   void
   launchConvolution( const Vector& dimension, const Vector& kernelSize ) const
   {
      DataStore input, result, kernel;

      size_t elementsCount = 1;
      size_t kernelElementsCount = 1;

      for( size_t i = 0; i < (size_t) dimension.getSize(); i++ ) {
         elementsCount *= dimension[ i ];
         kernelElementsCount *= kernelSize[ i ];
      }

      input.resize( elementsCount );
      result.resize( elementsCount );
      kernel.resize( kernelElementsCount );

      input = 1;
      result = 1;
      kernel = 1;

      auto inputView = input.getView();
      auto resultView = result.getView();
      auto kernelView = kernel.getView();

      DummyTask<int, float, Dimension, Device>::exec(dimension, kernelSize, inputView, resultView, kernelView);

      TNL::Containers::Array< float, TNL::Devices::Host, int > host(result);

      for (int i = 0; i < host.getSize(); i++)
         TNL_ASSERT_EQ(host[i], kernelElementsCount, "Dummy task always sets volume of kernel");

      std::cout << "Everything is fine" << std::endl;
   }

   virtual TNL::Config::ConfigDescription
   makeInputConfig() const override
   {
      TNL::Config::ConfigDescription config = Base::makeInputConfig();

      config.addDelimiter( "Grid dimension settings:" );

      for( int i = 0; i < Dimension; i++ )
         config.addEntry< int >( dimensionIds[ i ], dimensionIds[ i ], 64 );

      config.addDelimiter( "Kernel settings:" );

      for( int i = 0; i < Dimension; i++ )
         config.addEntry< int >( kernelSizeIds[ i ], kernelSizeIds[ i ] + " (odd) :", 9 );

      return config;
   }
};
