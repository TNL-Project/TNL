// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <stdexcept>
#include <type_traits>

#include <TNL/Meshes/Traits.h>
#include <TNL/Meshes/Readers/getMeshReader.h>
#include <TNL/Meshes/Writers/VTKWriter.h>
#include <TNL/Meshes/Writers/VTUWriter.h>
#include <TNL/Meshes/Writers/VTIWriter.h>
#include <TNL/Meshes/Writers/PVTIWriter.h>
#include "MeshFunctionGnuplotWriter.h"

namespace TNL::Functions::experimental {

template< typename MeshFunction >
void
readMeshFunction( MeshFunction& function,
                  const std::string& functionName,
                  const std::string& fileName,
                  const std::string& fileFormat = "auto" )
{
   std::shared_ptr< Meshes::Readers::MeshReader > reader = Meshes::Readers::getMeshReader( fileName, fileFormat );
   if( reader == nullptr )
      return;

   reader->detectMesh();

   // load the mesh if the function does not have it yet
   if( function.getMesh() == typename MeshFunction::MeshType{} )
      reader->loadMesh( *function.getMeshPointer() );

   Meshes::Readers::MeshReader::VariantVector data;
   if( function.getEntitiesDimension() == 0 )
      data = reader->readPointData( functionName );
   else if( function.getEntitiesDimension() == function.getMeshDimension() )
      data = reader->readCellData( functionName );
   else
      throw std::runtime_error( "The mesh function with entities dimension " + std::to_string( function.getEntitiesDimension() )
                                + " cannot be read from the file " + fileName );

   visit(
      [ & ]( auto&& array )
      {
         const auto entitiesCount = function.getMesh().template getEntitiesCount< MeshFunction::getEntitiesDimension() >();
         if( array.size() == (std::size_t) entitiesCount )
            Algorithms::copy< typename MeshFunction::VectorType::DeviceType, Devices::Host >(
               function.getData().getData(), array.data(), array.size() );
         else
            throw Exceptions::FileDeserializationError( fileName,
                                                        "mesh function data size does not match the mesh size (expected "
                                                           + std::to_string( entitiesCount ) + ", got "
                                                           + std::to_string( array.size() ) + ")." );
      },
      data );
}

template< typename MeshFunction >
void
readDistributedMeshFunction( Meshes::DistributedMeshes::DistributedMesh< typename MeshFunction::MeshType >& distributedMesh,
                             MeshFunction& function,
                             const std::string& functionName,
                             const std::string& fileName,
                             const std::string& fileFormat = "auto" )
{
   std::shared_ptr< Meshes::Readers::MeshReader > reader = Meshes::Readers::getMeshReader( fileName, fileFormat );
   if( reader == nullptr )
      return;

   reader->detectMesh();

   // load the mesh if it was not loaded yet
   using MeshType = typename MeshFunction::MeshType;
   using DistributedMeshType = Meshes::DistributedMeshes::DistributedMesh< MeshType >;
   if( distributedMesh == DistributedMeshType{} ) {
      if( reader->getMeshType() == "Meshes::DistributedMesh" )
         dynamic_cast< Meshes::Readers::PVTUReader& >( *reader ).loadMesh( distributedMesh );
      else if( reader->getMeshType() == "Meshes::DistributedGrid" )
         dynamic_cast< Meshes::Readers::PVTIReader& >( *reader ).loadMesh( distributedMesh );
      else
         throw std::runtime_error( "Unknown type of a distributed mesh: " + reader->getMeshType() );
   }
   if( function.getMesh() != distributedMesh.getLocalMesh() )
      // FIXME: DistributedMesh does not have a SharedPointer of the local mesh,
      // the interface is fucked up (it should not require us to put SharedPointer everywhere)
      *function.getMeshPointer() = distributedMesh.getLocalMesh();

   Meshes::Readers::MeshReader::VariantVector data;
   if( function.getEntitiesDimension() == 0 )
      data = reader->readPointData( functionName );
   else if( function.getEntitiesDimension() == function.getMeshDimension() )
      data = reader->readCellData( functionName );
   else
      throw std::runtime_error( "The mesh function with entities dimension " + std::to_string( function.getEntitiesDimension() )
                                + " cannot be read from the file " + fileName );

   visit(
      [ & ]( auto&& array )
      {
         const auto entitiesCount = function.getMesh().template getEntitiesCount< MeshFunction::getEntitiesDimension() >();
         if( array.size() == (std::size_t) entitiesCount )
            Algorithms::copy< typename MeshFunction::VectorType::DeviceType, Devices::Host >(
               function.getData().getData(), array.data(), array.size() );
         else
            throw Exceptions::FileDeserializationError( fileName,
                                                        "mesh function data size does not match the mesh size (expected "
                                                           + std::to_string( entitiesCount ) + ", got "
                                                           + std::to_string( array.size() ) + ")." );
      },
      data );
}

// specialization for grids
template< typename MeshFunction >
std::enable_if_t< Meshes::isGrid< typename MeshFunction::MeshType >::value >
writeMeshFunction( const MeshFunction& function,
                   const std::string& functionName,
                   const std::string& fileName,
                   const std::string& fileFormat = "auto" )
{
   std::ofstream file;
   // enable exceptions
   file.exceptions( std::fstream::failbit | std::fstream::badbit | std::fstream::eofbit );
   file.open( fileName );

   namespace fs = std::filesystem;

   std::string format = fileFormat;
   if( format == "auto" ) {
      format = fs::path( fileName ).extension().string();
      if( ! format.empty() )
         // remove dot from the extension
         format = format.substr( 1 );
   }

   if( format == "vti" ) {
      Meshes::Writers::VTIWriter< typename MeshFunction::MeshType > writer( file );
      writer.writeImageData( function.getMesh() );
      if( MeshFunction::getEntitiesDimension() == 0 )
         writer.writePointData( function.getData(), functionName, 1 );
      else
         writer.writeCellData( function.getData(), functionName, 1 );
   }
   else if( format == "gnuplot" || format == "gplt" || format == "plt" )
      MeshFunctionGnuplotWriter< MeshFunction >::write( function, file );
   else {
      if( fileFormat == "auto" )
         throw std::runtime_error( "Unsupported file format detected for file '" + fileName + "'. Detected format: " + format
                                   + ". Supported formats are: 'vti', 'gnuplot', 'gplt' and 'plt'." );
      else
         throw std::invalid_argument( "Invalid fileFormat parameter: '" + fileFormat
                                      + "'. Supported formats are: 'vti', 'gnuplot', 'gplt' and 'plt'." );
   }
}

// specialization for meshes
template< typename MeshFunction >
std::enable_if_t< ! Meshes::isGrid< typename MeshFunction::MeshType >::value >
writeMeshFunction( const MeshFunction& function,
                   const std::string& functionName,
                   const std::string& fileName,
                   const std::string& fileFormat = "auto" )
{
   std::ofstream file;
   // enable exceptions
   file.exceptions( std::fstream::failbit | std::fstream::badbit | std::fstream::eofbit );
   file.open( fileName );

   namespace fs = std::filesystem;

   std::string format = fileFormat;
   if( format == "auto" ) {
      format = fs::path( fileName ).extension().string();
      if( ! format.empty() )
         // remove dot from the extension
         format = format.substr( 1 );
   }

   if( format == "vtk" ) {
      Meshes::Writers::VTKWriter< typename MeshFunction::MeshType > writer( file );
      writer.template writeEntities< MeshFunction::getEntitiesDimension() >( function.getMesh() );
      if( MeshFunction::getEntitiesDimension() == 0 )
         writer.writePointData( function.getData(), functionName, 1 );
      else
         writer.writeCellData( function.getData(), functionName, 1 );
   }
   else if( format == "vtu" ) {
      Meshes::Writers::VTUWriter< typename MeshFunction::MeshType > writer( file );
      writer.template writeEntities< MeshFunction::getEntitiesDimension() >( function.getMesh() );
      if( MeshFunction::getEntitiesDimension() == 0 )
         writer.writePointData( function.getData(), functionName, 1 );
      else
         writer.writeCellData( function.getData(), functionName, 1 );
   }
   else if( format == "gnuplot" || format == "gplt" || format == "plt" )
      MeshFunctionGnuplotWriter< MeshFunction >::write( function, file );
   else {
      if( fileFormat == "auto" )
         throw std::runtime_error( "Unsupported file format detected for file '" + fileName + "'. Detected format: " + format
                                   + ". Supported formats are: 'vtk', 'vtu', 'gnuplot', 'gplt' and 'plt'." );
      else
         throw std::invalid_argument( "Invalid fileFormat parameter: '" + fileFormat
                                      + "'. Supported formats are: 'vtk', 'vtu', 'gnuplot', 'gplt' and 'plt'." );
   }
}

// specialization for grids
template< typename MeshFunction >
std::enable_if_t< Meshes::isGrid< typename MeshFunction::MeshType >::value, bool >
writeDistributedMeshFunction(
   const Meshes::DistributedMeshes::DistributedMesh< typename MeshFunction::MeshType >& distributedMesh,
   const MeshFunction& function,
   const std::string& functionName,
   const std::string& fileName,
   const std::string& fileFormat = "auto" )
{
   namespace fs = std::filesystem;

   std::string format = fileFormat;
   if( format == "auto" ) {
      format = fs::path( fileName ).extension().string();
      if( ! format.empty() )
         // remove dot from the extension
         format = format.substr( 1 );
   }

   if( format == "pvti" ) {
      const MPI::Comm& communicator = distributedMesh.getCommunicator();
      std::ofstream file;
      if( communicator.rank() == 0 )
         file.open( fileName );

      using PVTI = Meshes::Writers::PVTIWriter< typename MeshFunction::MeshType >;
      PVTI pvti( file );
      // TODO: write metadata: step and time
      pvti.writeImageData( distributedMesh );
      // TODO
      // if( distributedMesh.getGhostLevels() > 0 ) {
      //   pvti.template writePPointData< std::uint8_t >( Meshes::VTK::ghostArrayName() );
      //   pvti.template writePCellData< std::uint8_t >( Meshes::VTK::ghostArrayName() );
      //}
      if( function.getEntitiesDimension() == 0 )
         pvti.template writePPointData< typename MeshFunction::RealType >( functionName );
      else
         pvti.template writePCellData< typename MeshFunction::RealType >( functionName );
      const std::string subfilePath = pvti.addPiece( fileName, distributedMesh );

      // create a .vti file for local data
      // TODO: write metadata: step and time
      using Writer = Meshes::Writers::VTIWriter< typename MeshFunction::MeshType >;
      std::ofstream subfile( subfilePath );
      Writer writer( subfile );
      // NOTE: passing the local mesh to writeImageData does not work correctly, just like meshFunction->write(...)
      //       (it does not write the correct extent of the subdomain - globalBegin is only in the distributed grid)
      // NOTE: globalBegin and globalEnd here are without overlaps
      writer.writeImageData( distributedMesh.getGlobalGrid().getOrigin(),
                             distributedMesh.getGlobalBegin() - distributedMesh.getLowerOverlap(),
                             distributedMesh.getGlobalBegin() + distributedMesh.getLocalSize()
                                + distributedMesh.getUpperOverlap(),
                             distributedMesh.getGlobalGrid().getSpaceSteps() );
      if( function.getEntitiesDimension() == 0 )
         writer.writePointData( function.getData(), functionName );
      else
         writer.writeCellData( function.getData(), functionName );
      // TODO
      // if( mesh.getGhostLevels() > 0 ) {
      //   writer.writePointData( mesh.vtkPointGhostTypes(), Meshes::VTK::ghostArrayName() );
      //   writer.writeCellData( mesh.vtkCellGhostTypes(), Meshes::VTK::ghostArrayName() );
      //}
   }
   else {
      if( fileFormat == "auto" )
         throw std::runtime_error( "Unsupported file format detected for file '" + fileName + "'. Detected format: " + format
                                   + ". Supported formats are: 'pvti'." );
      else
         throw std::invalid_argument( "Invalid fileFormat parameter: '" + fileFormat + "'. Supported formats are: 'pvti'." );
   }
}

// TODO: specialization of writeDistributedMeshFunction for unstructured mesh

}  // namespace TNL::Functions::experimental
