
#pragma once

#include "TNL/Meshes/GridDetails/Grid1D.h"

using Grid1D = TNL::Meshes::Grid<1, float, TNL::Devices::Cuda, int>;
using Grid2D = TNL::Meshes::Grid<2, float, TNL::Devices::Cuda, int>;

void check1D() {
   Grid1D grid;

   grid.setDimensions({ 100 });

   auto fn0 = [=] __cuda_callable__(const Grid1D::EntityType<0>&entity, int variant) mutable {
      printf("%d %d\n", variant, entity.getCoordinates().x());
      };

   auto fn1 = [=] __cuda_callable__(const Grid1D::EntityType<1>&entity, int variant) mutable {
      printf("%d %d\n", variant, entity.getCoordinates().x());
   };

   grid.forAll<0>(fn0, 0);
   grid.forInterior<0>(fn0, 1);
   grid.forBoundary<0>(fn0, 2);

   grid.forAll<1>(fn1, 3);
   grid.forInterior<1>(fn1, 4);
   grid.forBoundary<1>(fn1, 5);
}

void check2D() {
   Grid2D grid;

   grid.setDimensions({ 10, 10 });

   auto fn0 = [=] __cuda_callable__(const Grid2D::EntityType<0>&entity, int variant) mutable {
      printf("%d %d %d\n", variant, entity.getCoordinates().x(), entity.getCoordinates().y());
   };

   auto fn1 = [=] __cuda_callable__(const Grid2D::EntityType<1>&entity, int variant) mutable {
      printf("%d %d %d\n", variant, entity.getCoordinates().x(), entity.getCoordinates().y());
   };

   auto fn2 = [=] __cuda_callable__(const Grid2D::EntityType<2>&entity, int variant) mutable {
      printf("%d %d %d\n", variant, entity.getCoordinates().x(), entity.getCoordinates().y());
   };

   grid.forAll<0>(fn0, 0);
   grid.forInterior<0>(fn0, 1);
   grid.forBoundary<0>(fn0, 2);

   grid.forAll<1>(fn1, 3);
   grid.forInterior<1>(fn1, 4);
   grid.forBoundary<1>(fn1, 5);

   grid.forAll<2>(fn2, 6);
   grid.forInterior<2>(fn2, 7);
   grid.forBoundary<2>(fn2, 8);
}

int main(int argc, char *argv[]) {
   check2D();
   return 0;
}

