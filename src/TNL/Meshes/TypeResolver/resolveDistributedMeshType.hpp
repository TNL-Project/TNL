// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <stdexcept>

#include <TNL/Meshes/TypeResolver/resolveDistributedMeshType.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>

namespace TNL::Meshes {

template< typename ConfigTag, typename Device, typename Functor >
void
resolveDistributedMeshType( Functor&& functor, const std::string& fileName, const std::string& fileFormat )
{
   std::cout << "Detecting distributed mesh from file " << fileName << " ...\n";

   auto wrapper = [ &functor ]( Readers::MeshReader& reader, auto&& localMesh )
   {
      using LocalMesh = std::decay_t< decltype( localMesh ) >;
      using DistributedMesh = DistributedMeshes::DistributedMesh< LocalMesh >;
      std::forward< Functor >( functor )( reader, DistributedMesh{ std::move( localMesh ) } );
   };

   resolveMeshType< ConfigTag, Device >( wrapper, fileName, fileFormat );
}

template< typename ConfigTag, typename Device, typename Functor >
void
resolveAndLoadDistributedMesh( Functor&& functor,
                               const std::string& fileName,
                               const std::string& fileFormat,
                               const MPI::Comm& communicator )
{
   auto wrapper = [ & ]( Readers::MeshReader& reader, auto&& mesh )
   {
      using MeshType = std::decay_t< decltype( mesh ) >;
      std::cout << "Loading a mesh from the file " << fileName << " ...\n";
      try {
         if( reader.getMeshType() == "Meshes::DistributedMesh" ) {
            auto& pvtu = dynamic_cast< Readers::PVTUReader& >( reader );
            pvtu.setCommunicator( communicator );
            pvtu.loadMesh( mesh );
         }
         else if( reader.getMeshType() == "Meshes::DistributedGrid" ) {
            auto& pvti = dynamic_cast< Readers::PVTIReader& >( reader );
            pvti.setCommunicator( communicator );
            pvti.loadMesh( mesh );
         }
         else
            throw std::runtime_error( "Unknown type of a distributed mesh: " + reader.getMeshType() );
      }
      catch( const Meshes::Readers::MeshReaderError& e ) {
         std::cerr << "Failed to load the mesh from the file " << fileName << ". The error is:\n" << e.what() << '\n';
         throw;
      }
      functor( reader, std::forward< MeshType >( mesh ) );
   };
   resolveDistributedMeshType< ConfigTag, Device >( wrapper, fileName, fileFormat );
}

template< typename Mesh >
void
loadDistributedMesh( DistributedMeshes::DistributedMesh< Mesh >& distributedMesh,
                     const std::string& fileName,
                     const std::string& fileFormat,
                     const MPI::Comm& communicator )
{
   namespace fs = std::filesystem;
   std::string format = fileFormat;
   if( format == "auto" ) {
      format = fs::path( fileName ).extension().string();
      if( ! format.empty() )
         // remove dot from the extension
         format = format.substr( 1 );
   }

   if( format == "pvtu" ) {
      Readers::PVTUReader reader( fileName, communicator );
      reader.loadMesh( distributedMesh );
   }
   else if( format == "pvti" ) {
      Readers::PVTIReader reader( fileName, communicator );
      reader.loadMesh( distributedMesh );
   }
   else {
      if( fileFormat == "auto" )
         throw std::runtime_error( "Unsupported file format detected for file '" + fileName + "'. Detected format: " + format
                                   + ". Supported formats are 'pvtu' and 'pvti'." );
      else
         throw std::invalid_argument( "Invalid fileFormat parameter: '" + fileFormat
                                      + "'. Supported formats are 'pvtu' and 'pvti'." );
   }
}

}  // namespace TNL::Meshes
