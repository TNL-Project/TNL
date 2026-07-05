#pragma once

#include <cfloat>  //FLT_MAX
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "../Particles.h"

namespace TNL {
namespace ParticleSystem {

/**
 * \brief Extracts the scalar component type of an array's ValueType.
 *
 * Yields \e T for scalar arrays and the underlying real type for
 * \e StaticVector< N, T > arrays.
 */
template< typename T >
struct ReadComponentType
{
   using type = T;
};

template< typename T, int Size >
struct ReadComponentType< Containers::StaticVector< Size, T > >
{
   using type = T;
};

template< typename T >
using ReadComponentType_t = typename ReadComponentType< T >::type;

/// Checks if an array's ValueType is a StaticVector.
template< typename T >
struct ReadIsStaticVector : std::false_type
{};

template< typename T, int Size >
struct ReadIsStaticVector< Containers::StaticVector< Size, T > > : std::true_type
{};

template< typename T >
inline constexpr bool ReadIsStaticVector_v = ReadIsStaticVector< T >::value;

// Custom Particles config to read data always in doubles
// This is used in case that particle positions use special real type which is incompatible with Readers
// space dimension and Cell indexer need to be passed from the user defined config
template< int spaceDim, typename CellIndexer >
class ParticleSystemToReadParticleSystem
{
public:
   using GlobalIndexType = long int;
   using LocalIndexType = long int;
   using CellIndexType = long int;
   using RealType = double;

   static constexpr int spaceDimension = spaceDim;

   using UseWithDomainDecomposition = std::false_type;
   using CellIndexerType = CellIndexer;
};

template< typename ParticlesConfig, typename Reader >
class ReadParticles
{
public:
   using GlobalIndexType = typename ParticlesConfig::GlobalIndexType;

   ReadParticles( const std::string& inputFileName,
                  const GlobalIndexType& numberOfParticles,
                  const GlobalIndexType numberOfAllocatedParticles )
   : reader( inputFileName ),
     numberOfParticles( numberOfParticles ),
     numberOfAllocatedParticles( numberOfAllocatedParticles )
   {
      reader.detectParticleSystem();
   }

   template< typename PointArray >
   void
   readParticles( PointArray& particles )
   {
      // Custom Particle config to read data
      using ParticleSystemToReadData = typename ParticleSystem::Particles<
         ParticleSystemToReadParticleSystem< ParticlesConfig::spaceDimension, typename ParticlesConfig::CellIndexerType >,
         Devices::Host >;

      ParticleSystemToReadData particlesToRead( numberOfParticles, numberOfParticles, 0. );
      reader.template loadParticle< ParticleSystemToReadData >( particlesToRead );

      PointArray pointsLoaded( numberOfParticles );
      pointsLoaded = particlesToRead.getPoints();
      pointsLoaded.resize( numberOfAllocatedParticles, FLT_MAX );
      particles = pointsLoaded;
   }

   /**
    * \brief Reads a particle variable from the file into \e array.
    *
    * Dispatches on the array's ValueType: scalar arrays copy the flat buffer
    * directly, vector arrays read the 3 components stored per particle in the
    * legacy VTK format and keep the first N (dropping unused components for 2D).
    *
    * \tparam Array type of the target array.
    * \param array output array (resized to numberOfAllocatedParticles, unused
    *    tail padded with FLT_MAX).
    * \param name is the name of the data array in the file.
    */
   template< typename Array >
   void
   readParticleVariable( Array& array, const std::string& name )
   {
      using ValueType = typename Array::ValueType;
      using CompType = ReadComponentType_t< ValueType >;
      const std::size_t pointsInFile = reader.getNumberOfPoints();

      if constexpr( ReadIsStaticVector_v< ValueType > ) {
         // vector path: the legacy VTK format stores 3 components per vector.
         // Read the flat buffer and reshape, keeping the first N components
         // (N = ValueType::getSize(), i.e. the simulation space dimension).
         std::vector< CompType > flat = std::get< std::vector< CompType > >( reader.readPointData( name ) );

         using HostArray = typename Array::
            template Self< std::remove_const_t< typename Array::ValueType >, Devices::Host, typename Array::IndexType >;
         HostArray hostArray( pointsInFile );

         for( std::size_t i = 0; i < pointsInFile; i++ ) {
            ValueType v;
            for( int j = 0; j < ValueType::getSize(); j++ )
               v[ j ] = flat[ 3 * i + j ];
            hostArray[ i ] = v;
         }

         hostArray.resize( numberOfAllocatedParticles, FLT_MAX );
         array = hostArray;
      }
      else {
         // scalar path: the flat buffer maps one-to-one onto the array
         Array arrayLoaded( pointsInFile );
         arrayLoaded = std::get< std::vector< CompType > >( reader.readPointData( name ) );
         arrayLoaded.resize( numberOfAllocatedParticles, FLT_MAX );
         array = arrayLoaded;
      }
   }

protected:
   Reader reader;
   GlobalIndexType numberOfParticles;
   GlobalIndexType numberOfAllocatedParticles;
};

}  //namespace ParticleSystem
}  //namespace TNL
