// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>

namespace TNL::Meshes {

template< typename ConfigTag, typename Device, typename Functor >
void
resolveDistributedMeshType( Functor&& functor, const std::string& fileName, const std::string& fileFormat = "auto" );

template< typename ConfigTag, typename Device, typename Functor >
void
resolveAndLoadDistributedMesh( Functor&& functor,
                               const std::string& fileName,
                               const std::string& fileFormat = "auto",
                               const MPI::Comm& communicator = MPI_COMM_WORLD );

template< typename Mesh >
void
loadDistributedMesh( DistributedMeshes::DistributedMesh< Mesh >& distributedMesh,
                     const std::string& fileName,
                     const std::string& fileFormat = "auto",
                     const MPI::Comm& communicator = MPI_COMM_WORLD );

}  // namespace TNL::Meshes

#include <TNL/Meshes/TypeResolver/resolveDistributedMeshType.hpp>
