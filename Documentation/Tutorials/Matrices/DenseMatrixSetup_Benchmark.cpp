#include <iostream>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Timer.h>

const int testsCount = 5;

template< typename Matrix >
void setElement_on_host( const int matrixSize, Matrix& matrix )
{
   matrix.setDimensions( matrixSize, matrixSize );

   for( int j = 0; j < matrixSize; j++ )
      for( int i = 0; i < matrixSize; i++ )
         matrix.setElement( i, j,  i + j );
}

template< typename Matrix >
void setElement_on_host_and_transfer( const int matrixSize, Matrix& matrix )
{
   using RealType = typename Matrix::RealType;
   using IndexType = typename Matrix::IndexType;
   using HostMatrix = TNL::Matrices::DenseMatrix< RealType, TNL::Devices::Host, IndexType >;
   HostMatrix hostMatrix( matrixSize, matrixSize );

   for( int j = 0; j < matrixSize; j++ )
      for( int i = 0; i < matrixSize; i++ )
         hostMatrix.setElement( i, j,  i + j );
   matrix = hostMatrix;
}

template< typename Matrix >
void setElement_on_device( const int matrixSize, Matrix& matrix )
{
   matrix.setDimensions( matrixSize, matrixSize );

   auto matrixView = matrix.getView();
   auto f = [=] __cuda_callable__ ( const TNL::Containers::StaticArray< 2, int >& i ) mutable {
         matrixView.setElement( i[ 0 ], i[ 1 ],  i[ 0 ] + i[ 1 ] );
   };
   const TNL::Containers::StaticArray< 2, int > begin = { 0, 0 };
   const TNL::Containers::StaticArray< 2, int > end = { matrixSize, matrixSize };
   TNL::Algorithms::parallelFor< typename Matrix::DeviceType >( begin, end, f );
}

template< typename Matrix >
void getRow( const int matrixSize, Matrix& matrix )
{
   matrix.setDimensions( matrixSize, matrixSize );

   auto matrixView = matrix.getView();
   auto f = [=] __cuda_callable__ ( int rowIdx ) mutable {
      auto row = matrixView.getRow( rowIdx );
      for( int i = 0; i < matrixSize; i++ )
         row.setValue( i, rowIdx + i );
   };
   TNL::Algorithms::parallelFor< typename Matrix::DeviceType >( 0, matrixSize, f );
}

template< typename Matrix >
void forElements( const int matrixSize, Matrix& matrix )
{
   matrix.setDimensions( matrixSize, matrixSize );

   auto f = [=] __cuda_callable__ ( int rowIdx, int localIdx, int& columnIdx, float& value ) mutable {
      value = rowIdx + columnIdx;
   };
   matrix.forElements( 0, matrixSize, f );
}

template< typename Device >
void setupDenseMatrix()
{
   std::cout << " Dense matrix test:" << std::endl;
   for( int matrixSize = 16; matrixSize <= 8192; matrixSize *= 2 )
   {
      std::cout << "  Matrix size = " << matrixSize << std::endl;
      TNL::Timer timer;

      std::cout << "   setElement on host: ";
      timer.reset();
      timer.start();
      for( int i = 0; i < testsCount; i++ )
      {
         TNL::Matrices::DenseMatrix< float, Device, int > matrix;
         setElement_on_host( matrixSize, matrix );
      }
      timer.stop();
      std::cout << timer.getRealTime() / ( double ) testsCount << " sec." << std::endl;

      std::cout << "   setElement on device: ";
      timer.reset();
      timer.start();
      for( int i = 0; i < testsCount; i++ )
      {
         TNL::Matrices::DenseMatrix< float, Device, int > matrix;
         setElement_on_device( matrixSize, matrix );
      }
      timer.stop();
      std::cout << timer.getRealTime() / ( double ) testsCount << " sec." << std::endl;

      if( std::is_same< Device, TNL::Devices::Cuda >::value )
      {
         std::cout << "   setElement on host and transfer on GPU: ";
         timer.reset();
         timer.start();
         for( int i = 0; i < testsCount; i++ )
         {
            TNL::Matrices::DenseMatrix< float, Device, int > matrix;
            setElement_on_host_and_transfer( matrixSize, matrix );
         }
         timer.stop();
         std::cout << timer.getRealTime() / ( double ) testsCount << " sec." << std::endl;
      }

      std::cout << "   getRow: ";
      timer.reset();
      timer.start();
      for( int i = 0; i < testsCount; i++ )
      {
         TNL::Matrices::DenseMatrix< float, Device, int > matrix;
         getRow( matrixSize, matrix );
      }
      timer.stop();
      std::cout << timer.getRealTime() / ( double ) testsCount << " sec." << std::endl;

      std::cout << "   forElements: ";
      timer.reset();
      timer.start();
      for( int i = 0; i < testsCount; i++ )
      {
         TNL::Matrices::DenseMatrix< float, Device, int > matrix;
         forElements( matrixSize, matrix );
      }
      timer.stop();
      std::cout << timer.getRealTime() / ( double ) testsCount << " sec." << std::endl;
   }
}


int main( int argc, char* argv[] )
{
   std::cout << "Creating dense matrix on CPU ... " << std::endl;
   setupDenseMatrix< TNL::Devices::Host >();


#ifdef __CUDACC__
   std::cout << "Creating dense matrix on CUDA GPU ... " << std::endl;
   setupDenseMatrix< TNL::Devices::Cuda >();
#endif
}
