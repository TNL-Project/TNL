/* 
 * File:   gem.h
 * Author: oberhuber
 *
 * Created on September 28, 2016, 5:30 PM
 */

#pragma once

#include <ostream>

template < typename Real = double,
           typename Device = TNL::Devices::Host,
           typename Index = int >
class GEM
{
  public:
    
    typedef Matrix< Real, Device, Index > MatrixGEM;
    typedef TNL::Containers::Array< Real, Device, Index > Array;
    
    
    GEM( MatrixGEM& A,
         Array& b );
    
    bool solve( Array& x, int verbose = 0 );
    
    bool solveWithPivoting( Array& x, int verbose = 0 );
    
    bool computeLUDecomposition( int verbose = 0 );
    
  protected:

    void print( std::ostream& str = std::cout ) const;

    MatrixGEM A;

    Array b;
};

#include "GEM_impl.h"

