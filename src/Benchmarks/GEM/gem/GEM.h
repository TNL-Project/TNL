/* 
 * File:   gem.h
 * Author: oberhuber
 *
 * Created on September 28, 2016, 5:30 PM
 */

#pragma once

#include <ostream>

template < typename Device >
class GEMDeviceDependentCode;

template < typename Real,
           typename Device,
           typename Index >
class GEM
{
  public:
    typedef Device DeviceType;
    typedef Matrix< Real, Device, Index > MatrixGEM;
    typedef TNL::Containers::Vector< Real, Device, Index > Array;
    
    
    GEM( MatrixGEM& A,
         Array& b );
    
    bool solve( Array& x, const TNL::String& pivoting, int verbose = 0 );
    
    bool solveWithoutPivoting( Array& x, int verbose = 0 );
    
    bool solveWithPivoting( Array& x, int verbose = 0 );
    
    bool computeLUDecomposition( int verbose = 0 );
    
    bool GEMdevice( Array& x, const TNL::String& pivoting, int verbose );
    
    bool setMatrixVector( MatrixGEM& A, Array& b ){
      this->A = A; this->b = b;
      return true;
    }

  protected: 
    void print( std::ostream& str = std::cout ) const;

    MatrixGEM A;

    Array b;   
    typedef GEMDeviceDependentCode< DeviceType > DeviceDependentCode;
    friend class GEMDeviceDependentCode< DeviceType >;
};

#include "GEM_impl.h"

