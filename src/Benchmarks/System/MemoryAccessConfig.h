/***************************************************************************
                          memoryAccessConfig.h  -  description
                             -------------------
    begin                : 2015/02/05
    copyright            : (C) 2015 by Tomá¨ Oberhuber
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

#ifndef memoryAccessConfigH
#define memoryAccessConfigH

class MemoryAccessConfig
{
   public:

   MemoryAccessConfig();
   
   bool parseCommandLineArguments( int argc, char* argv[] );

   void printHelp( const char* programName );

   const char* outputFile;

   const char* testType;

   int sequentialStride, elementSize;

   int initialArraySize,
       maxArraySize,
       elementsPerTest,
       tlbTestBlockSize,
       numThreads;


   bool readTest, writeTest, accessCentralData, interleaving;

   bool verbose;
};

#endif
