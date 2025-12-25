// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <filesystem>
#include <stdexcept>

#include <TNL/Meshes/Readers/NetgenReader.h>
#include <TNL/Meshes/Readers/VTKReader.h>
#include <TNL/Meshes/Readers/VTUReader.h>
#include <TNL/Meshes/Readers/VTIReader.h>
#include <TNL/Meshes/Readers/PVTUReader.h>
#include <TNL/Meshes/Readers/PVTIReader.h>
#include <TNL/Meshes/Readers/FPMAReader.h>

namespace TNL::Meshes::Readers {

inline std::shared_ptr< MeshReader >
getMeshReader( const std::string& fileName, const std::string& fileFormat )
{
   namespace fs = std::filesystem;

   std::string format = fileFormat;
   if( format == "auto" ) {
      format = fs::path( fileName ).extension().string();
      if( ! format.empty() )
         // remove dot from the extension
         format = format.substr( 1 );
   }

   if( format == "ng" )
      return std::make_shared< Readers::NetgenReader >( fileName );
   else if( format == "vtk" )
      return std::make_shared< Readers::VTKReader >( fileName );
   else if( format == "vtu" )
      return std::make_shared< Readers::VTUReader >( fileName );
   else if( format == "vti" )
      return std::make_shared< Readers::VTIReader >( fileName );
   else if( format == "pvtu" )
      return std::make_shared< Readers::PVTUReader >( fileName );
   else if( format == "pvti" )
      return std::make_shared< Readers::PVTIReader >( fileName );
   else if( format == "fpma" )
      return std::make_shared< Readers::FPMAReader >( fileName );

   if( fileFormat == "auto" )
      throw std::runtime_error( "Unsupported file format detected for file '" + fileName + "'. Detected format: " + format
                                + ". Supported formats are 'ng', 'vtk', 'vtu', 'vti', 'pvtu' and 'pvti'." );
   else
      throw std::invalid_argument( "Invalid fileFormat parameter: '" + fileFormat
                                   + "'. Supported formats are 'ng', 'vtk', 'vtu', 'vti', 'pvtu' and 'pvti'." );
}

}  // namespace TNL::Meshes::Readers
