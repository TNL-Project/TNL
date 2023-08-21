#pragma once

#include <gtest/gtest.h>

#include <TNL/Functions/BoundaryMeshFunction.h>
#include <TNL/Meshes/Grid.h>

TEST( BoundaryMeshFunctionTest, BasicConstructor )
{
   using Grid = TNL::Meshes::Grid< 2 >;
   TNL::Functions::BoundaryMeshFunction< Grid > boundaryMesh;
}

#include "../main.h"
