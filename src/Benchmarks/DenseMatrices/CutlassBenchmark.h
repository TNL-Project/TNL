#include <TNL/Matrices/DenseMatrix.h>
#include <type_traits>

#ifdef __CUDACC__
#ifdef HAVE_CUTLASS

#include <cutlass/cutlass.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/gemm/device/gemm.h>

template <typename RealType, typename DeviceType, typename IndexType>
void matrixMultiplicationCutlass(const TNL::Matrices::DenseMatrix<RealType, DeviceType, IndexType>& matrix1,
                                 const TNL::Matrices::DenseMatrix<RealType, DeviceType, IndexType>& matrix2,
                                 TNL::Matrices::DenseMatrix<RealType, DeviceType, IndexType>& resultMatrix) {
    // Define the matrix sizes
    int m = matrix1.getRows();
    int n = matrix2.getColumns();
    int k = matrix1.getColumns();

    // Define the element types and layout
    using ElementA = RealType;
    using LayoutA = cutlass::layout::RowMajor;
    using ElementB = RealType;
    using LayoutB = cutlass::layout::RowMajor;
    using ElementC = RealType;
    using LayoutC = cutlass::layout::RowMajor;

    // Define the GEMM operation
    using CutlassGemm = cutlass::gemm::device::Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>;
    CutlassGemm gemm_operator;

    typename CutlassGemm::Arguments args(
        {m, n, k},  // Problem size
        {matrix1.getValues().getData(), k},
        {matrix2.getValues().getData(), n},
        {resultMatrix.getValues().getData(), n},
        {resultMatrix.getValues().getData(), n},
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
