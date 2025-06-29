
#define DEBUG 0

template< typename Real, typename Device, typename Index >
Matrix< Real, Device, Index >::Matrix()
{}

template< typename Real, typename Device, typename Index >
Matrix< Real, Device, Index >::Matrix( Index rows, Index columns )
{
   this->setDimensions( rows, columns );
}

template< typename Real, typename Device, typename Index >
Matrix< Real, Device, Index >::Matrix( Matrix< Real, Device, Index >& matrix )
: rows( matrix.getNumRows() ),
  columns( matrix.getNumColumns() )
{
   matrix.getData( this->data );
}

template< typename Real, typename Device, typename Index >
void
Matrix< Real, Device, Index >::reset()
{
   this->rows = 1;
   this->columns = 1;
   this->data.reset();
   this->data.setSize( 1 );
}

template< typename Real, typename Device, typename Index >
void
Matrix< Real, Device, Index >::setDimensions( const Index rows, const Index columns )
{
   TNL_ASSERT( rows > 0 && columns > 0, std::cerr << "Matrix cannot have zero rows nor columns!\n" );
   this->rows = rows;
   this->columns = columns;

   if( std::is_same< Device, TNL::Devices::Host >::value ) {
#if DEBUG
      printf( "We are making host array!\n" );
#endif
      this->data.setSize( rows * columns );
      this->data.setValue( 0 );
   }
#ifdef HAVE_CUDA
   if( std::is_same< Device, TNL::Devices::Cuda >::value ) {
   #if DEBUG
      printf( "We are making cuda array!\n" );
   #endif
      data.setSize( rows * TNL::roundToMultiple( columns, TNL::Cuda::getWarpSize() ) );
      data.setValue( 0 );
   #if DEBUG
      printf( "Cuda array initialized!\n" );
   #endif
   }
#endif  // HAVE_CUDA
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
void
Matrix< Real, Device, Index >::setElement( Index row, Index col, Real value )
{
   TNL_ASSERT( row > -1 && col > -1, std::cerr << "Matrix cannot have egative row nor negative column!" );
   TNL_ASSERT( row < rows && col < columns, std::cerr << "Matrix dosn't have that much rows or columns!" );
   if( std::is_same< Device, TNL::Devices::Host >::value ) {
      data[ row * columns + col ] = value;
   }
#ifdef HAVE_CUDA
   if( std::is_same< Device, TNL::Devices::Cuda >::value ) {
      data[ row * TNL::roundToMultiple( columns, TNL::Cuda::getWarpSize() ) + col ] = value;
   }
#endif
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
Real
Matrix< Real, Device, Index >::getElement( Index row, Index col ) const
{
   TNL_ASSERT( row > -1 && col > -1, std::cerr << "Matrix cannot have egative row nor negative column!" );
   TNL_ASSERT( row < rows && col < columns, std::cerr << "Matrix dosn't have that much rows or columns!" );
   if( std::is_same< Device, TNL::Devices::Host >::value ) {
#if DEBUG
      printf( "On CPU\n" );
#endif
      return data[ row * columns + col ];
   }
#ifdef HAVE_CUDA
   if( std::is_same< Device, TNL::Devices::Cuda >::value ) {
      return this->data[ row * TNL::roundToMultiple( this->columns, TNL::Cuda::getWarpSize() ) + col ];
   }
#endif
   return 1;
}

template< typename Real, typename Device, typename Index >
void
Matrix< Real, Device, Index >::getRow( Index row, Index col, Real* mainRow, Index size )
{
   TNL_ASSERT( row > -1 && col > -1, std::cerr << "Matrix cannot have egative row nor negative column!" );
   TNL_ASSERT( row < rows && col < columns, std::cerr << "Matrix dosn't have that much rows or columns!" );
   if( std::is_same< Device, TNL::Devices::Host >::value ) {
#if DEBUG
      printf( "On CPU\n" );
#endif
      for( int i = 0; i < size - 1; i++ )
         mainRow[ i ] = this->getElement( row, col + i );
   }
#ifdef HAVE_CUDA
   if( std::is_same< Device, TNL::Devices::Cuda >::value ) {
      for( int i = 0; i < size - 1; i++ )
         mainRow[ i ] =
            this->data.getElement( row * TNL::roundToMultiple( this->columns, TNL::Cuda::getWarpSize() ) + col + i );
   }
#endif
}

template< typename Real, typename Device, typename Index >
void
Matrix< Real, Device, Index >::getRowGPU( Index row, Index col, Real* mainRow, Index size )
{
   TNL_ASSERT( row > -1 && col > -1, std::cerr << "Matrix cannot have egative row nor negative column!" );
   TNL_ASSERT( row < rows && col < columns, std::cerr << "Matrix dosn't have that much rows or columns!" );
#ifdef HAVE_CUDA
   if( std::is_same< Device, TNL::Devices::Cuda >::value ) {
      Matrix< Real, TNL::Devices::Cuda, Index >* devMat;
      cudaMalloc( (void**) &devMat, (size_t) sizeof( Matrix< Real, TNL::Devices::Cuda, Index > ) );
      cudaMemcpy( (void*) devMat, (void*) this, sizeof( Matrix< Real, TNL::Devices::Cuda, Index > ), cudaMemcpyHostToDevice );
      TNL_CHECK_CUDA_DEVICE;

      int blockSize = size - 1 > 256 ? 256 : size - 1;
      int gridSize = TNL::roundToMultiple( size - 1, blockSize );
      fillArray< < < gridSize, blockSize > > >( devMat, mainRow, size - 1, row, col );
      cudaDeviceSynchronize();
      TNL_CHECK_CUDA_DEVICE;
      cudaFree( devMat );
      cudaDeviceSynchronize();
      TNL_CHECK_CUDA_DEVICE;
   }
#endif
}

template< typename Real, typename Device, typename Index >
void
Matrix< Real, Device, Index >::setRowGPU( Index row, Index col, Real* mainRow, Index size )
{
   TNL_ASSERT( row > -1 && col > -1, std::cerr << "Matrix cannot have egative row nor negative column!" );
   TNL_ASSERT( row < rows && col < columns, std::cerr << "Matrix dosn't have that much rows or columns!" );
#ifdef HAVE_CUDA
   if( std::is_same< Device, TNL::Devices::Cuda >::value ) {
      Matrix< Real, TNL::Devices::Cuda, Index >* devMat;
      cudaMalloc( (void**) &devMat, (size_t) sizeof( Matrix< Real, TNL::Devices::Cuda, Index > ) );
      cudaMemcpy( (void*) devMat, (void*) this, sizeof( Matrix< Real, TNL::Devices::Cuda, Index > ), cudaMemcpyHostToDevice );
      TNL_CHECK_CUDA_DEVICE;

      int blockSize = size - 1 > 256 ? 256 : size - 1;
      int gridSize = TNL::roundToMultiple( size - 1, blockSize );
      fillRowMatrix< < < gridSize, blockSize > > >( devMat, mainRow, size - 1, row, col );
      cudaDeviceSynchronize();
      TNL_CHECK_CUDA_DEVICE;
      cudaFree( devMat );
      TNL_CHECK_CUDA_DEVICE;
   }
#endif
}

template< typename Real, typename Device, typename Index >
void
Matrix< Real, Device, Index >::setRow( Index row, Index col, Real* mainRow, Index size )
{
   TNL_ASSERT( row > -1 && col > -1, std::cerr << "Matrix cannot have egative row nor negative column!" );
   TNL_ASSERT( row < rows && col < columns, std::cerr << "Matrix dosn't have that much rows or columns!" );
   if( std::is_same< Device, TNL::Devices::Host >::value ) {
#if DEBUG
      printf( "On CPU\n" );
#endif
      for( int i = 0; i < size - 1; i++ )
         this->setElement( row, col + i, mainRow[ i ] );
   }
#ifdef HAVE_CUDA
   if( std::is_same< Device, TNL::Devices::Cuda >::value ) {
      for( int i = 0; i < size - 1; i++ )
         this->data.setElement( row * TNL::roundToMultiple( this->columns, TNL::Cuda::getWarpSize() ) + col + i, mainRow[ i ] );
   }
#endif
}

template< typename Real, typename Device, typename Index >
void
Matrix< Real, Device, Index >::setRow( Index row, Index col, Vector& mainRow )
{
   TNL_ASSERT( row > -1 && col > -1, std::cerr << "Matrix cannot have egative row nor negative column!" );
   TNL_ASSERT( row < rows && col < columns, std::cerr << "Matrix dosn't have that much rows or columns!" );
   if( std::is_same< Device, TNL::Devices::Host >::value ) {
#if DEBUG
      printf( "On CPU\n" );
#endif
      for( int i = 0; i < mainRow.getSize() - 1; i++ )
         this->setElement( row, col + i, mainRow[ i ] );
   }
#ifdef HAVE_CUDA
   if( std::is_same< Device, TNL::Devices::Cuda >::value ) {
      for( int i = 0; i < mainRow.getSize() - 1; i++ )
         this->data.setElement( row * TNL::roundToMultiple( this->columns, TNL::Cuda::getWarpSize() ) + col + i,
                                mainRow.getElement( i ) );
   }
#endif
}

template< typename Real, typename Device, typename Index >
void
Matrix< Real, Device, Index >::getRow( Index row, Index col, Vector& mainRow )
{
   TNL_ASSERT( row > -1 && col > -1, std::cerr << "Matrix cannot have egative row nor negative column!" );
   TNL_ASSERT( row < rows && col < columns, std::cerr << "Matrix dosn't have that much rows or columns!" );
   if( std::is_same< Device, TNL::Devices::Host >::value ) {
#if DEBUG
      printf( "On CPU\n" );
#endif
      for( int i = 0; i < this->getNumColumns() - col; i++ )
         mainRow[ i ] = this->getElement( row, col + i );
   }
#ifdef HAVE_CUDA
   if( std::is_same< Device, TNL::Devices::Cuda >::value ) {
      for( int i = 0; i < this->getNumColumns() - col; i++ )
         mainRow.setElement(
            i, this->data.getElement( row * TNL::roundToMultiple( this->columns, TNL::Cuda::getWarpSize() ) + col + i ) );
   }
#endif
}

template< typename Real, typename Device, typename Index >
template< typename Device1 >
void
Matrix< Real, Device, Index >::getData( TNL::Containers::Vector< Real, Device1, Index >& dataOut )
{
   /*dataOut.setSize( this->data.getSize() );
   for( int i = 0; i < this->data.getSize(); i++ )
     dataOut.setElement( i , this->data.getElement( i ));*/
   dataOut = this->data;
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
void
Matrix< Real, Device, Index >::showMatrix()
{
   for( int i = 0; i < this->rows; i++ ) {
      for( int j = 0; j < this->columns; j++ )
         printf( "%.4f ", this->getElement( i, j ) );
      printf( "\n" );
   }
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
void
Matrix< Real, Device, Index >::swapRows( Index row1, Index row2, Index column )
{
   TNL_ASSERT( row1 > -1 && row1 < rows, std::cerr << "Matrix cannot have egative row1 nor row1 greater than number of rows." );
   TNL_ASSERT( row2 > -1 && row2 < rows, std::cerr << "Matrix cannot have egative row2 nor row2 greater than number of rows." );
   TNL_ASSERT( column > -1 && column < columns,
               std::cerr << "Matrix cannot have egative column nor column greater than number of columns." );
   if( std::is_same< Device, TNL::Devices::Host >::value ) {
      for( int i = column; i < columns; i++ ) {
         Real pom = this->data[ row1 * columns + i ];
         this->data[ row1 * columns + i ] = this->data[ row2 * columns + i ];
         this->data[ row2 * columns + i ] = pom;
      }
   }
#ifdef HAVE_CUDA
   if( std::is_same< Device, TNL::Devices::Cuda >::value ) {
      for( int i = column; i < columns; i++ ) {
         Real pom = this->data[ TNL::roundToMultiple( this->rows, TNL::Cuda::getWarpSize() ) * i + row1 ];
         this->data[ TNL::roundToMultiple( this->rows, TNL::Cuda::getWarpSize() ) * i + row1 ] =
            this->data[ TNL::roundToMultiple( this->rows, TNL::Cuda::getWarpSize() ) * i + row2 ];
         this->data[ TNL::roundToMultiple( this->rows, TNL::Cuda::getWarpSize() ) * i + row2 ] = pom;
      }
   }
#endif
}

template< typename Real, typename Device, typename Index >
template< typename Device2 >
Matrix< Real, Device, Index >&
Matrix< Real, Device, Index >::operator=( Matrix< Real, Device2, Index >& matrix )
{
   this->setDimensions( matrix.getNumRows(), matrix.getNumColumns() );
   if( std::is_same< Device, Device2 >::value ) {
      Vector pom;
      matrix.getData( pom );
      this->data = pom;
   }
#ifdef HAVE_CUDA
   else if( std::is_same< Device, TNL::Devices::Host >::value ) {
      VectorHost pom;
      matrix.getData( pom );

      VectorHost pom1;
      pom1.setSize( this->data.getSize() );

   #ifdef HAVE_OPENMP
      #pragma omp parallel for if( Devices::Host::isOMPEnabled() )
   #endif
      for( int i = 0; i < this->rows; i++ )
         for( int j = 0; j < this->columns; j++ )
            pom1[ i * this->columns + j ] = pom[ i * TNL::roundToMultiple( this->rows, TNL::Cuda::getWarpSize() ) + j ];
      this->data = pom1;
   }
   else if( std::is_same< Device, TNL::Devices::Cuda >::value ) {
      VectorHost pom;
      matrix.getData( pom );

      VectorHost pom1;
      pom1.setSize( this->data.getSize() );
   #ifdef HAVE_OPENMP
      #pragma omp parallel for if( Devices::Host::isOMPEnabled() )
   #endif
      for( int i = 0; i < this->getNumRows(); i++ )
         for( int j = 0; j < this->getNumColumns(); j++ ) {
            pom1[ i * TNL::roundToMultiple( this->columns, TNL::Cuda::getWarpSize() ) + j ] = pom[ i * this->columns + j ];
         }
      this->data = pom1;
   }
#endif  // HAVE_CUDA
   return *this;
}

#ifdef HAVE_CUDA
template< typename Real, typename Index >
__global__
void
fillArray( Matrix< Real, TNL::Devices::Cuda, int >* A, Real* data, Index size, Index row, Index col )
{
   int thread = threadIdx.x + blockIdx.x * blockDim.x;
   if( thread < size ) {
      data[ thread ] = A->getElement( row, col + thread );
   }
}

template< typename Real, typename Index >
__global__
void
fillRowMatrix( Matrix< Real, TNL::Devices::Cuda, int >* A, Real* data, Index size, Index row, Index col )
{
   int thread = threadIdx.x + blockIdx.x * blockDim.x;
   if( thread < size ) {
      A->setElement( row, col + thread, data[ thread ] );
   }
}
#endif
