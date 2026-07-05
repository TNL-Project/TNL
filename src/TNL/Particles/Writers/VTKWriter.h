#pragma once

#include <ostream>

#include <stdexcept>
#include <string>
#include <cstdint>
#include <type_traits>

#include <TNL/Containers/StaticVector.h>
#include <TNL/Endianness.h>

#include "../Readers/VTKTraits.h"

namespace TNL {
namespace ParticleSystem {
namespace Writers {

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

template< typename ParticleSystem >
class VTKWriter
{
   static_assert( ParticleSystem::getParticlesDimension() <= 3, "The VTK format supports only 1D, 2D and 3D meshes." );
   // TODO: check also space dimension when grids allow it

public:
   using GlobalIndexType = std::uint64_t;

   VTKWriter() = delete;

   VTKWriter( std::ostream& str, VTK::FileFormat format = VTK::FileFormat::binary );

   void
   writeMetadata( std::int32_t cycle = -1, double time = -1 );

   template< int EntityDimension = ParticleSystem::getParticlesDimension() >
   void
   writeParticles( const ParticleSystem& particles );

   /**
    * \brief Writes data linked with particles (whole array).
    *
    * Mirrors \ref TNL::Meshes::Writers::VTKWriter::writePointData. The whole
    * \e array is written, the number of particles is taken from the particle
    * system written by \ref writeParticles.
    *
    * \tparam Array type of array holding the data.
    * \param array instance of an array holding the data.
    * \param name is a name of data which will appear in the output file.
    * \param numberOfComponents is number of components of the data for each
    *    particle (1 for scalars, 3 for vectors - the legacy VTK format always
    *    stores 3 components per vector, unused dimensions are zero-padded).
    */
   template< typename Array >
   void
   writePointData( const Array& array, const std::string& name, int numberOfComponents = 1 );

   /**
    * \brief Writes a slice of particle data (active particles only).
    *
    * Writes the first \e numberOfParticles entries out of \e array. This
    * overload serves SPH simulations where the array is allocated larger than
    * the number of active particles and only the leading range carries
    * meaningful data.
    *
    * \tparam Array type of array holding the data.
    * \param array instance of an array holding the data.
    * \param name is a name of data which will appear in the output file.
    * \param numberOfParticles is the number of particles to write from the
    *    start of the array.
    * \param numberOfComponents is number of components of the data for each
    *    particle (1 for scalars, 3 for vectors).
    */
   template< typename Array >
   void
   writePointData( const Array& array,
                   const std::string& name,
                   const GlobalIndexType numberOfParticles,
                   int numberOfComponents = 1 );

   /**
    * \brief Writes a DataArray (SCALARS or VECTORS) - whole array.
    *
    * Dispatches on the array's ValueType: StaticVector elements are serialized
    * component-by-component (zero-padded to 3 for the legacy VTK format), scalar
    * elements are written directly. This mirrors the unified \ref writePointData
    * interface of the Mesh writers - a single method serves both scalars and
    * vectors.
    */
   template< typename Array >
   void
   writeDataArray( const Array& array, const std::string& name, int numberOfComponents = 1 );

   /**
    * \brief Writes a DataArray (SCALARS or VECTORS) - slice of active particles.
    */
   template< typename Array >
   void
   writeDataArray( const Array& array,
                   const std::string& name,
                   const GlobalIndexType numberOfParticles,
                   int numberOfComponents = 1 );

protected:
   void
   writePoints( const ParticleSystem& particles );

   void
   writePointsTemp( const ParticleSystem& particles );

   void
   writeHeader();

   std::ostream str;

   VTK::FileFormat format;

   // number of cells (in the VTK sense) written to the file
   std::uint64_t cellsCount = 0;

   // number of points written to the file
   std::uint64_t pointsCount = 0;

   // indicator if the header has been written
   bool headerWritten = false;

   // number of data arrays written in each section
   int cellDataArrays = 0;
   int pointDataArrays = 0;

   // indicator of the current section
   VTK::DataType currentSection = VTK::DataType::CellData;

   template< typename T >
   void
   writeValue( VTK::FileFormat format, std::ostream& str, T value );
};

}  // namespace Writers
}  // namespace ParticleSystem
}  // namespace TNL

#include "VTKWriter.hpp"

