/* 
 * File:   Matrix.h
 * Author: maty
 *
 * Created on December 29, 2019, 6:18 PM
 */

#include <TNL/Containers/VectorView.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Cuda.h>

template < typename Real = double,
        typename Device = TNL::Devices::Host,
        typename Index = int >
class Matrix
{    
  public:
    typedef Index IndexType;
    typedef Device DeviceType;
    typedef Real RealType;
    typedef TNL::Containers::Vector< Index, Device, Index > CompressedRowLengthsVector;
    typedef TNL::Containers::Vector< Real, Device, Index > Vector;
    typedef TNL::Containers::VectorView< Real, Device, Index > VectorView;
    typedef TNL::Containers::VectorView< Real, TNL::Devices::Host, Index > HostVectorView;
    typedef TNL::Containers::VectorView< Real, TNL::Devices::Cuda, Index > DeviceVectorView;
    
    Matrix();
    Matrix( const Index rows, const Index columns );
    Matrix( Matrix< Real, Device, Index>& matrix );
    
    /**
    * Sets dimension for matrix of rows x columns elements and allocates memory
    * for data array of this matrix.
    *
    * @param number of rows and columns
    * @return void
    */
    void setDimensions( const Index rows, const Index columns );
    
    void setCompressedRowLengths( CompressedRowLengthsVector& rowLengths ){ rowLengths.setValue(this->getNumberOfMatrixElements());}
    
    IndexType getNumberOfMatrixElements(){return this->getNumRows() * this->getNumColumns();}
    /**
    * Classic setElement method.
    * Cannot be called from host if the matrix is on device and also cannot be 
    * called from device if the matrix is on host.
    *
    * @param row, column and value to save on row, column
    * @return void
    */
    __cuda_callable__
    void setElement( Index row, Index col, Real value );
    
    /**
    * Returns element on row and column. Cannot be
    * called from host if the matrix is on device and also cannot be called from
    * device if the matrix is on host.
    *
    * @param row and column
    * @return element on row and column.
    */
    __cuda_callable__ Real getElement( Index row, Index col ) const;
    
    /**
    * Returns ROW on row and starting column. Can be
    * called from host only.
    *
    * @param row and column, mainRow is array with size to be filled with values.
    * @return void.
    */
    void getRow( Index row, Index col, Real* mainRow, Index size );
    
    /**
    * Sets ROW on row and starting column into matrix A. Can be
    * called from host only.
    *
    * @param row and column, mainRow is array with size to be filled with values.
    * @return void.
    */
    void setRow( Index row, Index col, Real* mainRow, Index size );
    
    /**
    * Sets ROW on row and starting column into matrix A. Can be
    * called from host for host and device vector.
    *
    * @param row and column, mainRow is vector with size to be filled with values.
    * @return void.
    */
    void setRow( Index row, Index col, Vector& mainRow );
    /**
    * Returns ROW on row and starting column. Can be
    * called from host if the matrix is on device and also can be called from
    * device if the matrix is on host. Values are saved in vector mainRow
    *
    * @param row and column, mainRow is output vector.
    * @return void
    */
    void getRow( Index row, Index col, Vector& mainRow );
    
    /**
    * Saves data array into array parameter even in cross device function matrix.
    * Data array is huge on device for coalesced memory access condition! 
    *
    * @param Vector eather on host or device, depends where we want to store the data.
    * @return void
    */
    template < typename Device1 >
    void getData( TNL::Containers::Vector< Real, Device1, Index >& data );
    
    /**
    * Shows matrix elements fully. Can be called from host for host matrices or 
    * from device for device matrices! Cannot print cross device function matrix.
    *
    * @param none
    * @return void
    */
    __cuda_callable__
    void showMatrix();
    
    __cuda_callable__
    Index getNumRows() const{return this->rows;}
    
    __cuda_callable__
    Index getNumColumns() const{return this->columns;}
   
    Index getNumNonzeros() { 
      Index nonzeros = 0;
      Matrix< Real, TNL::Devices::Host, Index> m;
      m = *this;
      for( int i = 0; i < m.getNumRows(); i++ )
        for( int j = 0; j < m.getNumColumns(); j++ ) 
          if( m.getElement(i,j) != 0 )
            nonzeros++;
      return nonzeros;
    }
    
    /**
    * Function for switching 2 rows in matrix from column further. E.g. in matrix 5 x 5
    * switch rows row1 = 1 and row2 = 2 from column = 1 to columns = 5. 
    * Can be called from host for host matrix and from device for device matrix only.
    *
    * @param row1 as first row to switch
    * row2 as second row to switch
    * column as starting column till columns
    * @return void
    */
    __cuda_callable__
    void swapRows( Index row1, Index row2, Index column );
    
    /**
    * Cross device operator=, i.e can be used from host to host, device to host,
    * host to device or device to device.
    *
    * @param none
    * @return matrix on device of 'this'
    */
    template < typename Device2 >
    Matrix<Real, Device, Index >& operator=( Matrix< Real, Device2, Index>& matrix );
    
    
    
    private:
      // container for Matrix elements
      Vector data;
      
      // Indexes size of Matrix rows x columns
      Index rows, columns;
};

#include "Matrix_impl.h"

