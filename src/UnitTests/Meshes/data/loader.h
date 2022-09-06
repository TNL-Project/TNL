#pragma once

#include <iostream>
#include <string>
#ifdef __APPLE__
#include <filesystem>
#else
#include <experimental/filesystem>
#endif

#ifndef TNL_MESH_TESTS_DATA_DIR
   #error "The TNL_MESH_TESTS_DATA_DIR macro is not defined."
#endif

template< typename MeshType, typename ReaderType >
MeshType loadMeshFromFile( std::string relative_path )
{
#ifdef __APPLE__
   namespace fs = std::__fs::filesystem;
#else
   namespace fs = std::experimental::filesystem;
#endif

   const fs::path full_path = fs::path( TNL_MESH_TESTS_DATA_DIR ) / fs::path( relative_path );
   std::cout << "Reading a mesh from file " << full_path << std::endl;

   MeshType mesh;
   ReaderType reader( full_path );
   reader.loadMesh( mesh );
   return mesh;
}
