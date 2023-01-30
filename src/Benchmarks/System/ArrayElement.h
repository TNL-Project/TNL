/***************************************************************************
                          ArrayElement.h  -  description
                             -------------------
    begin                : 2015/02/04
    copyright            : (C) 2015 by Tomáš Oberhuber,
                         :             Milan Lang
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#pragma once

#include <iostream>

template< int Size >
class ArrayElement
{
   public:

      long int& operator[]( int i )
      {
         return data[ i ];
      }

      ArrayElement* next;

      /****
       * long int has the same size as a pointer on both 32 and 64 bits systems.
       */
      long int data[ Size - 1 ];
};

template<>
class ArrayElement< 1 >
{
   public:

      long int& operator[]( int i )
      {
         std::cerr << "Calling of operator [] for ArrayElement with Size = 1 does not make sense." << std::endl;
         abort();
      }

      ArrayElement* next;
	};


/****
 * We do not allow array element with no data.
 */
template<>
class ArrayElement< 0 >
{
};