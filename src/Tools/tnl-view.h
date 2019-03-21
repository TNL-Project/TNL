/***************************************************************************
                          tnl-view.h  -  description
                             -------------------
    begin                : Jan 21, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNL_VIEW_H_
#define TNL_VIEW_H_

#include <cstdlib>
#include <TNL/FileName.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/String.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Functions/VectorField.h>

#include <TNL/Communicators/NoDistrCommunicator.h>
#include <TNL/Meshes/TypeResolver/TypeResolver.h>

using namespace TNL;

bool getOutputFileName( const String& inputFileName,
                        const String& outputFormat,
                        String& outputFileName )
{
   outputFileName = removeFileNameExtension( inputFileName );
   if( outputFormat == "gnuplot" )
   {
      outputFileName += ".gplt";
      return true;
   }
   if( outputFormat == "vtk" )
   {
      outputFileName += ".vtk";
      return true;
   }
   std::cerr << "Unknown file format " << outputFormat << ".";
   return false;
}


template< typename MeshFunction >
bool writeMeshFunction( const typename MeshFunction::MeshPointer& meshPointer,
                        const String& inputFileName,
                        const Config::ParameterContainer& parameters  )
{

   MeshFunction function( meshPointer );
   std::cout << "Mesh function: " << function.getType() << std::endl;
   try
   {
      function.load( inputFileName );
   }
   catch(...)
   {
      std::cerr << "Unable to load mesh function from a file " << inputFileName << "." << std::endl;
      return false;
   }

   int verbose = parameters. getParameter< int >( "verbose");
   String outputFormat = parameters. getParameter< String >( "output-format" );
   double scale = parameters. getParameter< double >( "scale" );
   String outputFileName;
   if( ! getOutputFileName( inputFileName,
                            outputFormat,
                            outputFileName ) )
      return false;
   if( verbose )
     std::cout << " writing to " << outputFileName << " ... " << std::flush;

   return function.write( outputFileName, outputFormat, scale );
}

template< typename VectorField >
bool writeVectorField( const typename VectorField::FunctionType::MeshPointer& meshPointer,
                       const String& inputFileName,
                       const Config::ParameterContainer& parameters  )
{

   VectorField field( meshPointer );
   std::cout << "VectorField: " << field.getType() << std::endl;
   try
   {
      field.load( inputFileName );
   }
   catch(...)
   {
      std::cerr << "Unable to load vector field from a file " << inputFileName << "." << std::endl;
      return false;
   }

   int verbose = parameters. getParameter< int >( "verbose");
   String outputFormat = parameters. getParameter< String >( "output-format" );
   double scale = parameters. getParameter< double >( "scale" );
   String outputFileName;
   if( ! getOutputFileName( inputFileName,
                            outputFormat,
                            outputFileName ) )
      return false;
   if( verbose )
     std::cout << " writing to " << outputFileName << " ... " << std::flush;

   return field.write( outputFileName, outputFormat, scale );
}


template< typename MeshPointer,
          int EntityDimension,
          typename Real,
          int VectorFieldSize >
bool setMeshFunctionRealType( const MeshPointer& meshPointer,
                              const String& inputFileName,
                              const std::vector< String >& parsedObjectType,
                              const Config::ParameterContainer& parameters  )
{
   if( VectorFieldSize == 0 )
      return writeMeshFunction< Functions::MeshFunction< typename MeshPointer::ObjectType, EntityDimension, Real > >( meshPointer, inputFileName, parameters );
   return writeVectorField< Functions::VectorField< VectorFieldSize, Functions::MeshFunction< typename MeshPointer::ObjectType, EntityDimension, Real > > >( meshPointer, inputFileName, parameters );
}

template< typename MeshPointer,
          int EntityDimension,
          int VectorFieldSize,
          typename = typename std::enable_if< EntityDimension <= MeshPointer::ObjectType::getMeshDimension() >::type >
bool setMeshEntityType( const MeshPointer& meshPointer,
                        const String& inputFileName,
                        const std::vector< String >& parsedObjectType,
                        const Config::ParameterContainer& parameters )
{
   if( parsedObjectType[ 3 ] == "float" )
      return setMeshFunctionRealType< MeshPointer, EntityDimension, float, VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
   if( parsedObjectType[ 3 ] == "double" )
      return setMeshFunctionRealType< MeshPointer, EntityDimension, double, VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
   if( parsedObjectType[ 3 ] == "long double" )
      return setMeshFunctionRealType< MeshPointer, EntityDimension, long double, VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
   if( parsedObjectType[ 3 ] == "int" )
      return setMeshFunctionRealType< MeshPointer, EntityDimension, int, VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
   if( parsedObjectType[ 3 ] == "long int" )
      return setMeshFunctionRealType< MeshPointer, EntityDimension, long int, VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
   if( parsedObjectType[ 3 ] == "bool" )
      return setMeshFunctionRealType< MeshPointer, EntityDimension, bool, VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
   
   std::cerr << "Unsupported arithmetics " << parsedObjectType[ 3 ] << " in mesh function " << inputFileName << std::endl;
   return false;
}

template< typename MeshPointer,
          int EntityDimension,
          int VectorFieldSize,
          typename = typename std::enable_if< ( EntityDimension > MeshPointer::ObjectType::getMeshDimension() ) >::type,
          typename = void >
bool setMeshEntityType( const MeshPointer& meshPointer,
                        const String& inputFileName,
                        const std::vector< String >& parsedObjectType,
                        const Config::ParameterContainer& parameters )
{
   std::cerr << "Unsupported mesh functions entity dimension: " << EntityDimension << "." << std::endl;
   return false;
}

template< int VectorFieldSize,
          typename MeshPointer >
bool setMeshEntityDimension( const MeshPointer& meshPointer,
                             const String& inputFileName,
                             const std::vector< String >& parsedObjectType,
                             const Config::ParameterContainer& parameters )
{
   int meshEntityDimension = atoi( parsedObjectType[ 2 ].getString() );
   switch( meshEntityDimension )
   {
      case 0:
         return setMeshEntityType< MeshPointer, 0, VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
         break;      
      case 1:
         return setMeshEntityType< MeshPointer, 1, VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
         break;
      case 2:
         return setMeshEntityType< MeshPointer, 2, VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
         break;
      case 3:
         return setMeshEntityType< MeshPointer, 3, VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
         break;
      default:
         std::cerr << "Unsupported mesh functions entity dimension: " << meshEntityDimension << "." << std::endl;
         return false;
   }
}

template< typename MeshPointer, int VectorFieldSize = 0 >
bool setMeshFunction( const MeshPointer& meshPointer,
                      const String& inputFileName,
                      const std::vector< String >& parsedObjectType,
                      const Config::ParameterContainer& parameters )
{
   std::cerr << parsedObjectType[ 1 ] << std::endl;
   if( parsedObjectType[ 1 ] != meshPointer->getSerializationType() )
   {
      std::cerr << "Incompatible mesh type for the mesh function " << inputFileName << "." << std::endl;
      return false;
   }
   return setMeshEntityDimension< VectorFieldSize >( meshPointer, inputFileName, parsedObjectType, parameters );
}

template< typename MeshPointer >
bool setVectorFieldSize( const MeshPointer& meshPointer,
                         const String& inputFileName,
                         const std::vector< String >& parsedObjectType,
                         const Config::ParameterContainer& parameters )
{
   int vectorFieldSize = atoi( parsedObjectType[ 1 ].getString() );
   const std::vector< String > parsedMeshFunctionType = parseObjectType( parsedObjectType[ 2 ] );
   if( ! parsedMeshFunctionType.size() )
   {
      std::cerr << "Unable to parse mesh function type  " << parsedObjectType[ 2 ] << " in a vector field." << std::endl;
      return false;
   }
   switch( vectorFieldSize )
   {
      case 1:
         return setMeshFunction< MeshPointer, 1 >( meshPointer, inputFileName, parsedMeshFunctionType, parameters );
      case 2:
         return setMeshFunction< MeshPointer, 2 >( meshPointer, inputFileName, parsedMeshFunctionType, parameters );
      case 3:
         return setMeshFunction< MeshPointer, 3 >( meshPointer, inputFileName, parsedMeshFunctionType, parameters );
   }
   std::cerr << "Unsupported vector field size " << vectorFieldSize << "." << std::endl;
   return false;
}

template< typename MeshPointer, typename Value, typename Real, typename Index, int Dimension >
bool convertObject( const MeshPointer& meshPointer,
                    const String& inputFileName,
                    const std::vector< String >& parsedObjectType,
                    const Config::ParameterContainer& parameters )
{
   int verbose = parameters. getParameter< int >( "verbose");
   String outputFormat = parameters. getParameter< String >( "output-format" );
   String outputFileName;
   if( ! getOutputFileName( inputFileName,
                            outputFormat,
                            outputFileName ) )
      return false;
   if( verbose )
     std::cout << " writing to " << outputFileName << " ... " << std::flush;


   if( parsedObjectType[ 0 ] == "Containers::Vector" )
   {
      using MeshType = typename MeshPointer::ObjectType;
      // FIXME: why is MeshType::GlobalIndexType not the same as Index?
//      Containers::Vector< Value, Devices::Host, Index > vector;
      Containers::Vector< Value, Devices::Host, typename MeshType::GlobalIndexType > vector;
      vector.load( inputFileName );
      Functions::MeshFunction< MeshType, MeshType::getMeshDimension(), Value > mf;
      mf.bind( meshPointer, vector );
      mf.write( outputFileName, outputFormat );
   }
   return true;
}

template< typename MeshPointer, typename Value, typename Real, typename Index >
bool setDimension( const MeshPointer& meshPointer,
                    const String& inputFileName,
                    const std::vector< String >& parsedObjectType,
                    const Config::ParameterContainer& parameters )
{
   int dimensions( 0 );
   if( parsedObjectType[ 0 ] == "Containers::Vector" )
      dimensions = 1;
   switch( dimensions )
   {
      case 1:
         return convertObject< MeshPointer, Value, Real, Index, 1 >( meshPointer, inputFileName, parsedObjectType, parameters );
      case 2:
         return convertObject< MeshPointer, Value, Real, Index, 2 >( meshPointer, inputFileName, parsedObjectType, parameters );
      case 3:
         return convertObject< MeshPointer, Value, Real, Index, 3 >( meshPointer, inputFileName, parsedObjectType, parameters );
   }
   std::cerr << "Cannot convert objects with " << dimensions << " dimensions." << std::endl;
   return false;
}

template< typename MeshPointer, typename Value, typename Real >
bool setIndexType( const MeshPointer& meshPointer,
                   const String& inputFileName,
                   const std::vector< String >& parsedObjectType,
                   const Config::ParameterContainer& parameters )
{
   String indexType;
   if( parsedObjectType[ 0 ] == "Containers::Vector" )
      indexType = parsedObjectType[ 3 ];

   if( indexType == "int" )
      return setDimension< MeshPointer, Value, Real, int >( meshPointer, inputFileName, parsedObjectType, parameters );
   if( indexType == "long-int" )
      return setDimension< MeshPointer, Value, Real, long int >( meshPointer, inputFileName, parsedObjectType, parameters );
   std::cerr << "Unknown index type " << indexType << "." << std::endl;
   return false;
}

template< typename MeshPointer >
bool setTupleType( const MeshPointer& meshPointer,
                   const String& inputFileName,
                   const std::vector< String >& parsedObjectType,
                   const std::vector< String >& parsedValueType,
                   const Config::ParameterContainer& parameters )
{
   int dimensions = atoi( parsedValueType[ 1 ].getString() );
   String dataType = parsedValueType[ 2 ];
   if( dataType == "float" )
      switch( dimensions )
      {
         case 1:
            return setIndexType< MeshPointer, Containers::StaticVector< 1, float >, float >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
         case 2:
            return setIndexType< MeshPointer, Containers::StaticVector< 2, float >, float >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
         case 3:
            return setIndexType< MeshPointer, Containers::StaticVector< 3, float >, float >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
      }
   if( dataType == "double" )
      switch( dimensions )
      {
         case 1:
            return setIndexType< MeshPointer, Containers::StaticVector< 1, double >, double >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
         case 2:
            return setIndexType< MeshPointer, Containers::StaticVector< 2, double >, double >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
         case 3:
            return setIndexType< MeshPointer, Containers::StaticVector< 3, double >, double >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
      }
   if( dataType == "long double" )
      switch( dimensions )
      {
         case 1:
            return setIndexType< MeshPointer, Containers::StaticVector< 1, long double >, long double >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
         case 2:
            return setIndexType< MeshPointer, Containers::StaticVector< 2, long double >, long double >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
         case 3:
            return setIndexType< MeshPointer, Containers::StaticVector< 3, long double >, long double >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
      }
   return false;
}

template< typename MeshPointer >
bool setValueType( const MeshPointer& meshPointer,
                     const String& inputFileName,
                     const std::vector< String >& parsedObjectType,
                     const Config::ParameterContainer& parameters )
{
   String elementType;

   // TODO: Fix this even for arrays
   if( parsedObjectType[ 0 ] == "Containers::Vector" )
      elementType = parsedObjectType[ 1 ];

   if( elementType == "float" )
      return setIndexType< MeshPointer, float, float >( meshPointer, inputFileName, parsedObjectType, parameters );
   if( elementType == "double" )
      return setIndexType< MeshPointer, double, double >( meshPointer, inputFileName, parsedObjectType, parameters );
   if( elementType == "long double" )
      return setIndexType< MeshPointer, long double, long double >( meshPointer, inputFileName, parsedObjectType, parameters );
   if( elementType == "int" )
      return setIndexType< MeshPointer, int, int >( meshPointer, inputFileName, parsedObjectType, parameters );
   if( elementType == "long int" )
      return setIndexType< MeshPointer, long int, long int >( meshPointer, inputFileName, parsedObjectType, parameters );
   if( elementType == "bool" )
      return setIndexType< MeshPointer, bool, bool >( meshPointer, inputFileName, parsedObjectType, parameters );

   const std::vector< String > parsedValueType = parseObjectType( elementType );
   if( ! parsedValueType.size() )
   {
      std::cerr << "Unable to parse object type " << elementType << "." << std::endl;
      return false;
   }
   if( parsedValueType[ 0 ] == "Containers::StaticVector" )
      return setTupleType< MeshPointer >( meshPointer, inputFileName, parsedObjectType, parsedValueType, parameters );

   std::cerr << "Unknown element type " << elementType << "." << std::endl;
   return false;
}

template< typename Mesh >
struct FilesProcessor
{
   static bool run( const Config::ParameterContainer& parameters )
   {
      int verbose = parameters. getParameter< int >( "verbose");
      String meshFile = parameters. getParameter< String >( "mesh" );

      typedef Pointers::SharedPointer<  Mesh > MeshPointer;
      MeshPointer meshPointer;
      
      if( meshFile != "" )
      {
         Meshes::DistributedMeshes::DistributedMesh<Mesh> distributedMesh;
         if( ! Meshes::loadMesh<Communicators::NoDistrCommunicator>( meshFile, *meshPointer, distributedMesh ) )
            return false;
      }

      bool checkOutputFile = parameters. getParameter< bool >( "check-output-file" );
      std::vector< String > inputFiles = parameters. getParameter< std::vector< String > >( "input-files" );
      bool error( false );
   #ifdef HAVE_OPENMP
   #pragma omp parallel for
   #endif
      for( int i = 0; i < (int) inputFiles.size(); i++ )
      {
         if( verbose )
           std::cout << "Processing file " << inputFiles[ i ] << " ... " << std::flush;

         String outputFormat = parameters. getParameter< String >( "output-format" );
         String outputFileName;
         if( ! getOutputFileName( inputFiles[ i ],
                                  outputFormat,
                                  outputFileName ) )
         {
            error = true;
            continue;
         }
         if( checkOutputFile && fileExists( outputFileName ) )
         {
            if( verbose )
              std::cout << " file already exists. Skipping.            \r" << std::flush;
            continue;
         }

         String objectType;
         try
         {
            objectType = getObjectType( inputFiles[ i ] );
         }
         catch(...)
         {
            std::cerr << "unknown object ... SKIPPING!" << std::endl;
            continue;
         }
         
         if( verbose )
           std::cout << objectType << " detected ... ";

         const std::vector< String > parsedObjectType = parseObjectType( objectType );
         if( ! parsedObjectType.size() )
         {
            std::cerr << "Unable to parse object type " << objectType << "." << std::endl;
            error = true;
            continue;
         }
         if( parsedObjectType[ 0 ] == "Containers::Vector" )
            setValueType< MeshPointer >( meshPointer, inputFiles[ i ], parsedObjectType, parameters );
         if( parsedObjectType[ 0 ] == "Functions::MeshFunction" )
            setMeshFunction< MeshPointer >( meshPointer, inputFiles[ i ], parsedObjectType, parameters );
         if( parsedObjectType[ 0 ] == "Functions::VectorField" )
            setVectorFieldSize< MeshPointer >( meshPointer, inputFiles[ i ], parsedObjectType, parameters );
         if( verbose )
            std::cout << "[ OK ].  " << std::endl;
      }
      if( verbose )
        std::cout << std::endl;
      return ! error;
   }
};

#endif /* TNL_VIEW_H_ */
