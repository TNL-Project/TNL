
#pragma once

#include "TNL/Meshes/GridDetails/Grid1D.h"

using Grid = TNL::Meshes::Grid<1, float, TNL::Devices::Cuda, int>;

int main(int argc, char *argv[]) {
   Grid grid;

   grid.setDimensions({ 100 });

   auto fn0 = [=] __cuda_callable__(const Grid::EntityType<0>&entity, int variant) mutable {
      printf("%d %d\n", variant, entity.getCoordinates().x());
   };

   auto fn1 = [=] __cuda_callable__(const Grid::EntityType<1>& entity, int variant) mutable {
      printf("%d %d\n", variant, entity.getCoordinates().x());
   };

   grid.forAll<0>(fn0, 0);//, "All 0 dim:");
   grid.forInterior<0>(fn0, 1);//, "Interior 0 dim:");
   grid.forBoundary<0>(fn0, 2);//, "Boundary 0 dim:");


   grid.forAll<1>(fn1, 3);//, "All 1 dim:");
   grid.forInterior<1>(fn1, 4);//, "Interior 1 dim:");
   grid.forBoundary<1>(fn1, 5);//, "Boundary 1 dim:");

   return 0;
}

