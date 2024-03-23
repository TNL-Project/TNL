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

   auto wrapper = [ & ]( auto& reader, auto&& mesh ) -> bool
   {
      return task( mesh, inputFileName );
   };
   const bool result = TNL::Meshes::resolveAndLoadMesh< MyConfigTag, TNL::Devices::Host >( wrapper, inputFileName, "auto" );
   return static_cast< int >( ! result );
}
//! [main]
