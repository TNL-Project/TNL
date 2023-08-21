#include <TNL/Matrices/Sandbox/SparseSandboxMatrix.h>

#include <gtest/gtest.h>

// types for which MatrixTest is instantiated
using MatrixTypes = ::testing::Types<
   TNL::Matrices::Sandbox::SparseSandboxMatrix< int, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix >,
   TNL::Matrices::Sandbox::SparseSandboxMatrix< long, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix >,
   TNL::Matrices::Sandbox::SparseSandboxMatrix< float, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix >,
   TNL::Matrices::Sandbox::SparseSandboxMatrix< double, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix >,
   TNL::Matrices::Sandbox::SparseSandboxMatrix< int, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix >,
   TNL::Matrices::Sandbox::SparseSandboxMatrix< long, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix >,
   TNL::Matrices::Sandbox::SparseSandboxMatrix< float, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix >,
   TNL::Matrices::Sandbox::SparseSandboxMatrix< double, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix >
#ifdef __CUDACC__
   ,
   TNL::Matrices::Sandbox::SparseSandboxMatrix< int, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix >,
   TNL::Matrices::Sandbox::SparseSandboxMatrix< long, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix >,
   TNL::Matrices::Sandbox::SparseSandboxMatrix< float, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix >,
   TNL::Matrices::Sandbox::SparseSandboxMatrix< double, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix >,
   TNL::Matrices::Sandbox::SparseSandboxMatrix< int, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix >,
   TNL::Matrices::Sandbox::SparseSandboxMatrix< long, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix >,
   TNL::Matrices::Sandbox::SparseSandboxMatrix< float, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix >,
   TNL::Matrices::Sandbox::SparseSandboxMatrix< double, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix >
#endif
   >;

#include "SparseMatrixVectorProductTest.h"
#include "../main.h"
