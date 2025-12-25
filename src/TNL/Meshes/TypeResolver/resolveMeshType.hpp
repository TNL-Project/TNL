// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>

#include <TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <TNL/Meshes/TypeResolver/GridTypeResolver.h>
#include <TNL/Meshes/TypeResolver/MeshTypeResolver.h>
#include <TNL/Meshes/Readers/getMeshReader.h>

namespace TNL::Meshes {

template< typename ConfigTag, typename Device, typename Functor >
void
resolveMeshType( Functor&& functor,
                 const std::string& fileName,
                 const std::string& fileFormat,
                 const std::string& realType,
                 const std::string& globalIndexType )
{
   std::shared_ptr< Readers::MeshReader > reader = Readers::getMeshReader( fileName, fileFormat );
   if( reader == nullptr )
      return;

   reader->detectMesh();

   if( realType != "auto" )
      reader->forceRealType( realType );
   if( globalIndexType != "auto" )
      reader->forceGlobalIndexType( globalIndexType );

   if( reader->getMeshType() == "Meshes::Grid" || reader->getMeshType() == "Meshes::DistributedGrid" )
      GridTypeResolver< ConfigTag, Device >::run( *reader, functor );
   else if( reader->getMeshType() == "Meshes::Mesh" || reader->getMeshType() == "Meshes::DistributedMesh" )
      MeshTypeResolver< ConfigTag, Device >::run( *reader, functor );
   else {
      throw std::runtime_error( "The mesh type " + reader->getMeshType() + " is not supported." );
   }
}

template< typename ConfigTag, typename Device, typename Functor >
void
resolveAndLoadMesh( Functor&& functor,
                    const std::string& fileName,
                    const std::string& fileFormat,
                    const std::string& realType,
                    const std::string& globalIndexType )
{
   auto wrapper = [ & ]( auto& reader, auto&& mesh ) -> void
   {
      using MeshType = std::decay_t< decltype( mesh ) >;
      std::cout << "Loading a mesh from the file " << fileName << " ...\n";
      try {
         reader.loadMesh( mesh );
      }
      catch( const Meshes::Readers::MeshReaderError& e ) {
         std::cerr << "Failed to load the mesh from the file " << fileName << ". The error is:\n" << e.what() << '\n';
         throw;
      }
      functor( reader, std::forward< MeshType >( mesh ) );
   };
   resolveMeshType< ConfigTag, Device >( wrapper, fileName, fileFormat, realType, globalIndexType );
}

template< typename Mesh >
void
loadMesh( Mesh& mesh, const std::string& fileName, const std::string& fileFormat )
{
   std::cout << "Loading a mesh from the file " << fileName << " ...\n";

   std::shared_ptr< Readers::MeshReader > reader = Readers::getMeshReader( fileName, fileFormat );
   if( reader == nullptr )
      return;

   try {
      reader->loadMesh( mesh );
   }
   catch( const Meshes::Readers::MeshReaderError& e ) {
      std::cerr << "Failed to load the mesh from the file " << fileName << ". The error is:\n" << e.what() << '\n';
      throw;
   }
}

template< typename MeshConfig >
void
loadMesh( Mesh< MeshConfig, Devices::Cuda >& mesh, const std::string& fileName, const std::string& fileFormat )
{
   Mesh< MeshConfig, Devices::Host > hostMesh;
   loadMesh( hostMesh, fileName, fileFormat );
   mesh = hostMesh;
}

}  // namespace TNL::Meshes
