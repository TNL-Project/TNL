#pragma once

#include <ostream>

#include <stdexcept>
#include <string>
#include <cstdint>
#include <type_traits>

#include <TNL/Containers/StaticVector.h>
#include <TNL/Endianness.h>
#include <TNL/Matrices/StaticMatrix.h>

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

template< typename Value, std::size_t Rows, std::size_t Columns, typename Permutation >
struct ComponentType< Matrices::StaticMatrix< Value, Rows, Columns, Permutation > >
{
   using type = Value;
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

/// Checks if an array's ValueType is a StaticMatrix.
template< typename T >
struct IsStaticMatrix : std::false_type
{};

template< typename Value, std::size_t Rows, std::size_t Columns, typename Permutation >
struct IsStaticMatrix< Matrices::StaticMatrix< Value, Rows, Columns, Permutation > > : std::true_type
{};

template< typename T >
inline constexpr bool IsStaticMatrix_v = IsStaticMatrix< T >::value;

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
    * \brief Writes data linked with particles (active particles only).
    *
    * Writes the first \ref pointsCount entries of \e array, where
    * \ref pointsCount is the number of particles logged by the prior
    * \ref writeParticles call. The array may be allocated larger than
    * \ref pointsCount (e.g. SPH simulations where arrays are sized to
    * \c numberOfAllocatedParticles); only the leading active range is
    * written.
    *
    * The VTK section header (SCALARS / VECTORS / TENSORS) is selected
    * automatically from the array's ValueType - scalar types yield SCALARS,
    * \ref Containers::StaticVector yields VECTORS, \ref Matrices::StaticMatrix
    * yields TENSORS. The legacy VTK format zero-pads vectors to 3 and tensors
    * to 3x3 = 9 components.
    *
    * \tparam Array type of array holding the data.
    * \param array instance of an array holding the data.
    * \param name is a name of data which will appear in the output file.
    */
   template< typename Array >
   void
   writePointData( const Array& array, const std::string& name );

   /**
    * \brief Writes a slice of particle data with an explicit count.
    *
    * Writes the first \e numberOfParticles entries out of \e array. Use this
    * overload when the count differs from \ref pointsCount (e.g. mass-node
    * data in multiresolution simulations). The VTK section header is selected
    * automatically from the array's ValueType, as in the count-less overload.
    */
   template< typename Array >
   void
   writePointData( const Array& array, const std::string& name, const GlobalIndexType numberOfParticles );

   /**
    * \brief Writes a DataArray (SCALARS / VECTORS / TENSORS) - whole array.
    *
    * Dispatches on the array's ValueType: StaticMatrix elements are serialized
    * as TENSORS (zero-padded to 3x3), StaticVector elements as VECTORS
    * (zero-padded to 3), scalar elements as SCALARS.
    */
   template< typename Array >
   void
   writeDataArray( const Array& array, const std::string& name );

   /**
    * \brief Writes a DataArray (SCALARS / VECTORS / TENSORS) - slice of
    *        active particles.
    */
   template< typename Array >
   void
   writeDataArray( const Array& array, const std::string& name, const GlobalIndexType numberOfParticles );

protected:
   void
   writePoints( const ParticleSystem& particles );

   void
   writePointsTemp( const ParticleSystem& particles );

   void
   writeHeader();

   std::ostream str;

   VTK::FileFormat format;

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

