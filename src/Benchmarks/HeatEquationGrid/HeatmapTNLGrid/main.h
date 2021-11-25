
#pragma once

#include "TNL/Meshes/GridDetails/Grid2D.h"
#include "../Base/HeatmapSolver.h"

template <typename Real>
template <typename Device>
bool HeatmapSolver<Real>::solve(const HeatmapSolver<Real>::Parameters& params) const {
   using Grid2D = TNL::Meshes::Grid<2, Real, Device, int>;

   Grid2D grid;

   // Grid implementation defines its dimensions in the amount of edges.
   // To align it size to all other benchmarks substract 1
   grid.setDimensions(params.xSize - 1, params.ySize - 1);

   // TODO: - Improve style of access. It is counterintuitive for person, who doesn't know C++ well
   int verticesCount = grid.template getEntitiesCount<0>();

   TNL::Containers::Array<Real, Device> ux(verticesCount), // data at step u
                                        aux(verticesCount);// data at step u + 1

   // Invalidate ux/aux
   ux = 0;
   aux = 0;

   const Real hx = params.xDomainSize / (Real)params.xSize;
   const Real hy = params.yDomainSize / (Real)params.ySize;
   const Real hx_inv = 1 / (hx * hx);
   const Real hy_inv = 1 / (hy * hy);

   auto timestep = params.timeStep ? params.timeStep : std::min(hx * hx, hy * hy);

   auto uxView = ux.getView(),
        auxView = aux.getView();

   auto init = [=] __cuda_callable__(const typename Grid2D::EntityType<0> &entity) mutable {
      auto index = entity.getIndex();

      auto x = entity.getCoordinates().x() * hx - params.xDomainSize / 2;
      auto y = entity.getCoordinates().y() * hy - params.yDomainSize / 2;

      uxView[index] = exp(params.sigma * (x * x + y * y));
   };

   grid.template forInterior<0>(init);

   if (!writeGNUPlot("data.txt", params, ux))
      return false;

   auto next = [=] __cuda_callable__(const typename Grid2D::EntityType<0>&entity) mutable {
      auto index = entity.getIndex();
      auto width = grid.getDimensions().x() + 1;

      auxView[index] = (uxView[index - 1] - 2 * uxView[index] + uxView[index + 1]) * hx_inv +
                        (uxView[index - width] - 2 * uxView[index] + uxView[index + width]) * hy_inv;
   };

   auto update = [=] __cuda_callable__(const typename Grid2D::EntityType<0>&entity) mutable {
      auto index = entity.getIndex();

      uxView[index] += auxView[index] * timestep;
   };

   Real start = 0;

   while (start < params.finalTime) {
      grid.template forInterior<0>(next);
      grid.template forInterior<0>(update);

      start += timestep;
   }

   return writeGNUPlot("data_final.txt", params, ux);

   return false;
}

int main(int argc, char* argv[]) {
   using Real = double;

   auto config = HeatmapSolver<Real>::Parameters::makeInputConfig();

   TNL::Config::ParameterContainer parameters;
   if (!parseCommandLine(argc, argv, config, parameters))
      return EXIT_FAILURE;

   auto device = parameters.getParameter<TNL::String>("device");
   auto params = HeatmapSolver<Real>::Parameters(parameters);

   HeatmapSolver<Real> solver;

   if (device == "host" && !solver.solve<TNL::Devices::Host>(params))
      return EXIT_FAILURE;

#ifdef HAVE_CUDA
   if (device == "cuda" && !solver.solve<TNL::Devices::Cuda>(params))
      return EXIT_FAILURE;
#endif

   return EXIT_SUCCESS;
}


// TODO: - Move to tests
/*
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

   std::cout << "Entities count of 0: dimension" << grid.getEntitiesCount<0>() << std::endl;

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
   return 0;
}
*/
