#include <TNL/Matrices/DenseMatrix.h>
#include <type_traits>

#ifdef __CUDACC__
#ifdef HAVE_CUTLASS

#include <cutlass/cutlass.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/gemm/device/gemm.h>

template <typename DenseMatrix>
void matrixMultiplicationCutlass(const DenseMatrix& matrix1,
                                 const DenseMatrix& matrix2,
                                 DenseMatrix& resultMatrix) {

   using RealType = typename DenseMatrix::RealType;
   using IndexType = typename DenseMatrix::IndexType;

   // Define the matrix sizes
   IndexType m = matrix1.getRows();
   IndexType n = matrix2.getColumns();
   IndexType k = matrix1.getColumns();

   // Define the element types and layout for column-major order
   using ElementA = RealType;
   using LayoutA = cutlass::layout::ColumnMajor;
   using ElementB = RealType;
   using LayoutB = cutlass::layout::ColumnMajor;
   using ElementC = RealType;
   using LayoutC = cutlass::layout::ColumnMajor;

   // Define the GEMM operation
   using CutlassGemm = cutlass::gemm::device::Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>;
   CutlassGemm gemm_operator;

   typename CutlassGemm::Arguments args(
        {m, n, k},  // Problem size
        {matrix1.getValues().getData(), m}, // Leading dimension for A
        {matrix2.getValues().getData(), k}, // Leading dimension for B
        {resultMatrix.getValues().getData(), m}, // Leading dimension for C
        {resultMatrix.getValues().getData(), m}, // Leading dimension for C
        {1.0, 0.0}   // alpha and beta
   );

   // Launch the GEMM operation
   cutlass::Status status = gemm_operator(args);
   if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("CUTLASS GEMM failed");
   }
}
#endif //HAVE_CUTLASS
#endif
