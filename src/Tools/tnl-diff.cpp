// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include "tnl-diff.h"
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>

struct TNLDiffBuildConfigTag
{};

namespace TNL::Meshes::BuildConfigTags {

/****
 * Turn off support for float and long double.
 */
//template<> struct GridRealTag< TNLDiffBuildConfigTag, float > { static constexpr bool enabled = false; };
template<>
struct GridRealTag< TNLDiffBuildConfigTag, long double >
{
   static constexpr bool enabled = false;
};

/****
 * Turn off support for short int and long int indexing.
 */
template<>
struct GridIndexTag< TNLDiffBuildConfigTag, short int >
{
   static constexpr bool enabled = false;
};
template<>
struct GridIndexTag< TNLDiffBuildConfigTag, long int >
{
   static constexpr bool enabled = false;
};

}  // namespace TNL::Meshes::BuildConfigTags

void
setupConfig( Config::ConfigDescription& config )
{
   config.addEntry< std::string >( "mesh", "Input mesh file.", "mesh.vti" );
   config.addEntry< std::string >( "mesh-format", "Mesh file format.", "auto" );
   config.addRequiredList< std::string >( "input-files", "Input files containing the mesh functions to be compared." );
   config.addEntry< std::string >( "mesh-function-name", "Name of the mesh function in the input files.", "f" );
   config.addEntry< std::string >( "output-file", "File for the output data.", "tnl-diff.log" );
   config.addEntry< std::string >(
      "mode",
      "Mode 'couples' compares two subsequent files. Mode 'sequence' compares the input files against the first one. 'halves' "
      "compares the files from the first and the second half of the input files.",
      "couples" );
   config.addEntryEnum< std::string >( "couples" );
   config.addEntryEnum< std::string >( "sequence" );
   config.addEntryEnum< std::string >( "halves" );
   config.addEntry< bool >( "exact-match", "Check if the data are exactly the same.", false );
   config.addEntry< bool >( "write-difference", "Write difference grid function.", false );
   //   config.addEntry< bool >( "write-exact-curve", "Write exact curve with given radius.", false );
   config.addEntry< int >( "edges-skip", "Width of the edges that will be skipped - not included into the error norms.", 0 );
   //   config.addEntry< bool >( "write-graph", "Draws a graph in the Gnuplot format of the dependence of the error norm on t.",
   //   true ); config.addEntry< bool >( "write-log-graph", "Draws a logarithmic graph in the Gnuplot format of the dependence
   //   of the error norm on t.", true );
   config.addEntry< double >( "snapshot-period", "The period between consecutive snapshots.", 0.0 );
   config.addEntry< bool >( "verbose", "Sets verbosity.", true );
}

int
main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;
   setupConfig( conf_desc );
   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   const auto meshFile = parameters.getParameter< std::string >( "mesh" );
   const auto meshFileFormat = parameters.getParameter< std::string >( "mesh-format" );

   bool status = true;
   auto wrapper = [ & ]( const auto& reader, auto&& mesh )
   {
      using MeshType = std::decay_t< decltype( mesh ) >;
      status = processFiles< MeshType >( parameters );
   };
   try {
      TNL::Meshes::resolveMeshType< TNLDiffBuildConfigTag, Devices::Host >( wrapper, meshFile, meshFileFormat );
   }
   catch( const std::exception& e ) {
      std::cerr << "Error: " << e.what() << '\n';
      return EXIT_FAILURE;
   }
   return static_cast< int >( ! status );
}
