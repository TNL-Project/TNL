#include <TNL/Meshes/TypeResolver/resolveMeshType.h>

// Define the tag for the MeshTypeResolver configuration
struct MyConfigTag
{};

//! [Configuration example]
namespace TNL::Meshes::BuildConfigTags {

// Create a template specialization of the tag specifying the MeshConfig template to use as the Config parameter for the mesh.
template<>
struct MeshConfigTemplateTag< MyConfigTag >
{
   template< typename Cell,
             int SpaceDimension = Cell::dimension,
             typename Real = double,
             typename GlobalIndex = int,
             typename LocalIndex = short int >
   struct MeshConfig : public DefaultConfig< Cell, SpaceDimension, Real, GlobalIndex, LocalIndex >
   {
      static constexpr bool
      subentityStorage( int entityDimension, int subentityDimension )
      {
         return subentityDimension == 0 && entityDimension >= Cell::dimension - 1;
      }
   };
};

}  // namespace TNL::Meshes::BuildConfigTags
//! [Configuration example]

namespace TNL::Meshes::BuildConfigTags {

// disable all grids
template< int Dimension, typename Real, typename Device, typename Index >
struct GridTag< MyConfigTag, Grid< Dimension, Real, Device, Index > >
{
   static constexpr bool enabled = false;
};

template<>
struct MeshCellTopologyTag< MyConfigTag, Topologies::Triangle >
{
   static constexpr bool enabled = true;
};

}  // namespace TNL::Meshes::BuildConfigTags

int
main( int argc, char* argv[] )
{
   const std::string inputFileName = "example-triangles.vtu";

   auto wrapper = []( auto& reader, auto&& mesh ) -> bool
   {
      return true;
   };
   const bool result = TNL::Meshes::resolveAndLoadMesh< MyConfigTag, TNL::Devices::Host >( wrapper, inputFileName, "auto" );
   return static_cast< int >( ! result );
}
