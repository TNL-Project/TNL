
#pragma once

#include <vector>

#include <TNL/Containers/StaticVector.h>
#include <TNL/Containers/Array.h>

template< int Dimension, typename Device >
class Solver
{
public:
   void
   solve( const TNL::Config::ParameterContainer& parameters ) const
   {
      if( ! TNL::Devices::Host::setup( parameters ) || ! TNL::Devices::Cuda::setup( parameters ) )
         return;

      start( parameters );
   }

   virtual void
   start( const TNL::Config::ParameterContainer& parameters ) const
   {
      TNL_ASSERT_TRUE( false, "Should be overriden" );
   }

   virtual TNL::Config::ConfigDescription
   makeInputConfig() const
   {
      TNL::Config::ConfigDescription config;

      config.addEntry< TNL::String >( "device", "Device the computation will run on.", "cuda" );
      config.addEntryEnum< TNL::String >( "all" );
      config.addEntryEnum< TNL::String >( "host" );

#ifdef HAVE_CUDA
      config.addEntryEnum< TNL::String >( "cuda" );
#endif

      config.addDelimiter( "Device settings:" );
      TNL::Devices::Host::configSetup( config );

#ifdef HAVE_CUDA
      TNL::Devices::Cuda::configSetup( config );
#endif

      return config;
   }
};
