//! [config]
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>

// Define the tag for the MeshTypeResolver configuration
struct MyConfigTag
{};

namespace TNL::Meshes::BuildConfigTags {

template<>
struct MeshCellTopologyTag< MyConfigTag, Topologies::Triangle >
{
   static constexpr bool enabled = true;
};

template<>
struct MeshCellTopologyTag< MyConfigTag, Topologies::Quadrangle >
{
   static constexpr bool enabled = true;
};

}  // namespace TNL::Meshes::BuildConfigTags
//! [config]

//! [task]
// Define the main task/function of the program
template< typename Mesh >
bool
task( const Mesh& mesh, const std::string& inputFileName )
{
   std::cout << "The file '" << inputFileName << "' contains the following mesh: " << TNL::getType< Mesh >() << std::endl;
   return true;
}
//! [task]

//! [main]
int
main( int argc, char* argv[] )
{
   const std::string inputFileName = "example-triangles.vtu";

   bool result = true;
   auto wrapper = [ & ]( auto& reader, auto&& mesh )
   {
      result = task( mesh, inputFileName );
   };
   try {
      TNL::Meshes::resolveAndLoadMesh< MyConfigTag, TNL::Devices::Host >( wrapper, inputFileName, "auto" );
   }
   catch( const std::exception& e ) {
      std::cerr << "Error: " << e.what() << '\n';
      return EXIT_FAILURE;
   }
   return static_cast< int >( ! result );
}
//! [main]
