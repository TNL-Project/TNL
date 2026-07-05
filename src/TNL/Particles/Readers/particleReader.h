#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <variant>

#include <TNL/Containers/StaticVector.h>
#include <TNL/Exceptions/NotImplementedError.h>
#include <TNL/Particles/Particles.h>

#include "VTKTraits.h"

namespace TNL {
namespace ParticleSystem {
namespace Readers {

struct ParticleReaderError : public std::runtime_error
{
   ParticleReaderError( const std::string& readerName, const std::string& msg )
   : std::runtime_error( readerName + " error: " + msg )
   {}
};

/**
 * \brief Extracts the scalar component type of an array's ValueType.
 *
 * Yields \e T for scalar arrays and the underlying real type for
 * \e StaticVector< N, T > arrays.
 */
template< typename T >
struct ComponentType
{
   using type = T;
};

template< typename T, int Size >
struct ComponentType< Containers::StaticVector< Size, T > >
{
   using type = T;
};

template< typename T >
using ComponentType_t = typename ComponentType< T >::type;

/// Checks if an array's ValueType is a StaticVector.
template< typename T >
struct IsStaticVector : std::false_type
{};

template< typename T, int Size >
struct IsStaticVector< Containers::StaticVector< Size, T > > : std::true_type
{};

template< typename T >
inline constexpr bool IsStaticVector_v = IsStaticVector< T >::value;

class ParticleReader
{
public:
   using VariantVector = std::variant< std::vector< std::int8_t >,
                                       std::vector< std::uint8_t >,
                                       std::vector< std::int16_t >,
                                       std::vector< std::uint16_t >,
                                       std::vector< std::int32_t >,
                                       std::vector< std::uint32_t >,
                                       std::vector< std::int64_t >,
                                       std::vector< std::uint64_t >,
                                       std::vector< float >,
                                       std::vector< double > >;

   ParticleReader() = default;

   ParticleReader( std::string fileName )
   : fileName( std::move( fileName ) )
   {}

   virtual ~ParticleReader() = default;

   void
   setFileName( const std::string& fileName )
   {
      reset();
      this->fileName = fileName;
   }

   /**
    * \brief Resets the reader to an empty state.
    *
    * Implementations should call \ref resetBase to reset the arrays holding the
    * intermediate particle representation.
    */
   virtual void
   reset()
   {
      resetBase();
   }

   virtual void
   detectParticleSystem() = 0;

   /**
    * \brief Loads particle positions into the given particle system.
    *
    * Mirrors \ref TNL::Meshes::Readers::MeshReader::loadMesh. The particle
    * system's points array is filled from the file's POINTS section.
    */
   template< typename ParticleType >
   void
   loadParticle( ParticleType& particles )
   {
      if( NumberOfPoints == 0 ) {
         particles = ParticleType{};
         return;
      }

      using PointType = typename ParticleType::PointType;
      auto points_view = particles.getPoints().getView();

      visit(
         [ &points_view ]( auto&& array )
         {
            PointType p;
            std::size_t i = 0;
            for( auto c : array ) {
               int dim = i++ % 3;
               if( dim >= PointType::getSize() )
                  continue;
               p[ dim ] = c;
               if( dim == PointType::getSize() - 1 )
                  points_view[ ( i - 1 ) / 3 ] = p;
            }
         },
         pointsArray );
   }

   /**
    * \brief Loads particle positions into the given particle system, handling
    *        host/device transfer and inactive-slot padding.
    *
    * Points are loaded into a host temp (the reader's I/O loop runs on host)
    * and copied into the target particle system. The points array is then
    * resized to the particle system's allocated size, padding inactive slots
    * with the maximum value of \ref ParticleType::RealType so that neighbor
    * search skips them.
    */
   template< typename ParticleType >
   void
   loadParticles( ParticleType& particles )
   {
      using HostParticles = TNL::ParticleSystem::Particles< typename ParticleType::Config, Devices::Host >;
      HostParticles hostTemp( particles.getNumberOfParticles(), particles.getNumberOfParticles(), 0. );
      loadParticle( hostTemp );

      typename ParticleType::PointArrayType points( particles.getNumberOfParticles() );
      points = hostTemp.getPoints();
      points.resize( particles.getNumberOfAllocatedParticles(), std::numeric_limits< typename ParticleType::RealType >::max() );
      particles.getPoints() = points;
   }

   virtual VariantVector
   readPointData( const std::string& arrayName )
   {
      throw Exceptions::NotImplementedError(
         "readPointData is not implemented in the particle reader for this specific file format." );
   }

   /**
    * \brief Reads a particle variable from the file into \e array.
    *
    * Fills the first \ref getNumberOfPoints elements of \e array. The array
    * must be pre-sized by the caller (e.g. via setSize). The array's size is
    * not modified.
    *
    * Dispatches on the array's ValueType: scalar arrays copy the flat buffer
    * directly, vector arrays read the 3 components stored per particle in the
    * legacy VTK format and keep the first N (dropping unused components for 2D).
    *
    * \tparam Array type of the target array.
    * \param array output array (must be pre-sized; only the leading
    *    getNumberOfPoints() entries are written).
    * \param name is the name of the data array in the file.
    */
   template< typename Array >
   void
   readParticleVariable( Array& array, const std::string& name )
   {
      using ValueType = typename Array::ValueType;
      using CompType = ComponentType_t< ValueType >;
      const std::size_t pointsInFile = getNumberOfPoints();

      std::vector< CompType > flat = std::get< std::vector< CompType > >( readPointData( name ) );

      if constexpr( IsStaticVector_v< ValueType > ) {
         for( std::size_t i = 0; i < pointsInFile; i++ ) {
            ValueType v;
            for( int j = 0; j < ValueType::getSize(); j++ )
               v[ j ] = flat[ 3 * i + j ];
            array.setElement( i, v );
         }
      }
      else {
         for( std::size_t i = 0; i < pointsInFile; i++ )
            array.setElement( i, flat[ i ] );
      }
   }

   int
   getSpaceDimension() const
   {
      return spaceDimension;
   }

   std::size_t
   getNumberOfPoints() const
   {
      return NumberOfPoints;
   }

protected:
   // input file name
   std::string fileName;

   // attributes of the particle system
   std::size_t NumberOfPoints;
   int spaceDimension;

   // string representation of ptcs types (forced means specified by the user, otherwise
   // the type detected by detectMesh takes precedence)
   std::string forcedRealType;
   std::string forcedGlobalIndexType;
   std::string forcedLocalIndexType = "short int";  // not stored in any file format

   // points
   VariantVector pointsArray;

   // string representation of each array's value type
   std::string pointsType, connectivityType, offsetsType, typesType;

   void
   resetBase()
   {
      NumberOfPoints = 0;
      spaceDimension = 0;

      pointsArray = {};
      pointsType = connectivityType = offsetsType = typesType = "";
   }
};

}  // namespace Readers
}  // namespace ParticleSystem
}  // namespace TNL
