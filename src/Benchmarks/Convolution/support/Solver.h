
#pragma once

#include <vector>

#include <TNL/Containers/StaticVector.h>
#include <TNL/Containers/Vector.h>

template< int Dimension, typename Device >
class Solver
{
public:
   void
   solve( const TNL::Config::ParameterContainer& parameters ) const
   {
      if( ! TNL::Devices::Cuda::setup( parameters ) )
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

      config.addDelimiter( "Device settings:" );

#ifdef HAVE_CUDA
      TNL::Devices::Cuda::configSetup( config );
#endif

      return config;
   }
};
