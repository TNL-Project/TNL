// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/Writers/VTKWriter.h>
#include <TNL/Meshes/Writers/detail/VTKOffsetsCountGetter.h>
#include <TNL/Meshes/Writers/detail/VTKMeshEntitiesWriter.h>

namespace TNL::Meshes::Writers {

template< typename Mesh >
VTKWriter< Mesh >::VTKWriter( std::ostream& str, VTK::FileFormat format )
: str( str.rdbuf() ),
  format( format )
{
   if( format != VTK::FileFormat::ascii && format != VTK::FileFormat::binary )
      throw std::domain_error( "The Legacy VTK file formats support only ASCII and BINARY formats." );
}

template< typename Mesh >
void
VTKWriter< Mesh >::writeMetadata( int cycle, double time )
{
   if( ! headerWritten )
      writeHeader();

   int n_metadata = 0;
   if( cycle >= 0 )
      ++n_metadata;
   if( time >= 0 )
      ++n_metadata;
   if( n_metadata > 0 )
      str << "FIELD FieldData " << n_metadata << "\n";
   if( cycle >= 0 ) {
      str << "CYCLE 1 1 int\n";
      detail::writeValue( format, str, cycle );
      str << "\n";
   }
   if( time >= 0 ) {
      str << "TIME 1 1 double\n";
      detail::writeValue( format, str, time );
      str << "\n";
   }
}

template< typename Mesh >
template< int EntityDimension >
void
VTKWriter< Mesh >::writeEntities( const Mesh& mesh )
{
   if( ! headerWritten )
      writeHeader();
   writePoints( mesh );

   using EntityType = typename Mesh::template EntityType< EntityDimension >;
   cellsCount = mesh.template getEntitiesCount< EntityType >();
   const std::uint64_t offsetsCount = detail::VTKOffsetsCountGetter< Mesh, EntityDimension >::getOffsetsCount( mesh );

   // legacy VTK files always have fixed integer width, even in the BINARY format
   // - DataFormat version 2.0: 32-bit
   // - DataFormat version 5.1: 64-bit (vtktypeint64)
   str << "\nCELLS " << cellsCount + 1 << " " << offsetsCount << '\n';
   str << "OFFSETS vtktypeint64\n";
   detail::VTKMeshEntitiesWriter< Mesh, EntityDimension >::template writeOffsets< std::int64_t >( mesh, str, format );
   str << "CONNECTIVITY vtktypeint64\n";
   detail::VTKMeshEntitiesWriter< Mesh, EntityDimension >::template writeConnectivity< std::int64_t >( mesh, str, format );

   str << "\nCELL_TYPES " << cellsCount << '\n';
   detail::VTKMeshEntityTypesWriter< Mesh, EntityDimension >::exec( mesh, str, format );
}

template< typename Mesh >
template< typename Array >
void
VTKWriter< Mesh >::writePointData( const Array& array, const std::string& name, const int numberOfComponents )
{
   if( array.getSize() / numberOfComponents != typename Array::IndexType( pointsCount ) )
      throw std::length_error( "Mismatched array size for POINT_DATA section: " + std::to_string( array.getSize() )
                               + " (there are " + std::to_string( pointsCount ) + " points in the file)" );

   // check that we won't start the section second time
   if( currentSection != VTK::DataType::PointData && cellDataArrays * pointDataArrays != 0 )
      throw std::logic_error( "The requested data section is not the current section and it has already been written." );
   currentSection = VTK::DataType::PointData;

   // start the appropriate section if necessary
   if( pointDataArrays == 0 )
      str << "\nPOINT_DATA " << pointsCount << '\n';
   ++pointDataArrays;

   writeDataArray( array, name, numberOfComponents );
}

template< typename Mesh >
template< typename Array >
void
VTKWriter< Mesh >::writeCellData( const Array& array, const std::string& name, const int numberOfComponents )
{
   if( array.getSize() / numberOfComponents != typename Array::IndexType( cellsCount ) )
      throw std::length_error( "Mismatched array size for CELL_DATA section: " + std::to_string( array.getSize() )
                               + " (there are " + std::to_string( cellsCount ) + " cells in the file)" );

   // check that we won't start the section second time
   if( currentSection != VTK::DataType::CellData && cellDataArrays * pointDataArrays != 0 )
      throw std::logic_error( "The requested data section is not the current section and it has already been written." );
   currentSection = VTK::DataType::CellData;

   // start the appropriate section if necessary
   if( cellDataArrays == 0 )
      str << "\nCELL_DATA " << cellsCount << '\n';
   ++cellDataArrays;

   writeDataArray( array, name, numberOfComponents );
}

template< typename Mesh >
template< typename Array >
void
VTKWriter< Mesh >::writeDataArray( const Array& array, const std::string& name, const int numberOfComponents )
{
   // use a host buffer if direct access to the array elements is not possible
   if( std::is_same_v< typename Array::DeviceType, Devices::Cuda > ) {
      using HostArray = typename Array::
         template Self< std::remove_const_t< typename Array::ValueType >, Devices::Host, typename Array::IndexType >;
      HostArray hostBuffer;
      hostBuffer = array;
      writeDataArray( hostBuffer, name, numberOfComponents );
      return;
   }

   if( numberOfComponents != 1 && numberOfComponents != 3 )
      throw std::logic_error( "Unsupported numberOfComponents parameter: " + std::to_string( numberOfComponents ) );

   // data type string identifier - convert from XML type names to legacy
   std::string datatype;
   if( VTK::getTypeName( typename Array::ValueType{} ) == "Int32" ) {
      datatype = "vtktypeint32";
   }
   else if( VTK::getTypeName( typename Array::ValueType{} ) == "Int64" ) {
      datatype = "vtktypeint64";
   }
   else {
      // this should be just float or double
      datatype = getType< typename Array::ValueType >();
      if( datatype != "float" && datatype != "double" )
         throw std::invalid_argument( "found data type which is not implemented in the legacy VTK writer: " + datatype );
   }

   // write DataArray header
   if( numberOfComponents == 1 ) {
      str << "SCALARS " << name << " " << datatype << '\n';
      str << "LOOKUP_TABLE default\n";
   }
   else {
      str << "VECTORS " << name << " " << datatype << '\n';
   }

   using detail::writeValue;
   for( typename Array::IndexType i = 0; i < array.getSize(); i++ ) {
      writeValue( format, str, array[ i ] );
      if( format == VTK::FileFormat::ascii )
         str << "\n";
   }
}

template< typename Mesh >
void
VTKWriter< Mesh >::writePoints( const Mesh& mesh )
{
   using detail::writeValue;
   pointsCount = mesh.template getEntitiesCount< typename Mesh::Vertex >();
   str << "POINTS " << pointsCount << " " << getType< typename Mesh::RealType >() << '\n';
   for( std::uint64_t i = 0; i < pointsCount; i++ ) {
      const auto& vertex = mesh.template getEntity< typename Mesh::Vertex >( i );
      const auto& point = vertex.getPoint();
      for( int j = 0; j < point.getSize(); j++ )
         writeValue( format, str, point[ j ] );
      // VTK needs zeros for unused dimensions
      for( int j = point.getSize(); j < 3; j++ )
         writeValue( format, str, (typename Mesh::PointType::RealType) 0 );
      if( format == VTK::FileFormat::ascii )
         str << "\n";
   }
}

template< typename Mesh >
void
VTKWriter< Mesh >::writeHeader()
{
   str << "# vtk DataFile Version 5.1\n"
       << "TNL DATA\n"
       << ( ( format == VTK::FileFormat::ascii ) ? "ASCII\n" : "BINARY\n" ) << "DATASET UNSTRUCTURED_GRID\n";
   headerWritten = true;
}

}  // namespace TNL::Meshes::Writers
