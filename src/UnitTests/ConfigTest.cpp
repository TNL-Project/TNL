#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Config/parseINIConfigFile.h>
#include <TNL/Exceptions/ConfigError.h>

using namespace TNL;
using namespace TNL::Config;

static const char* TEST_FILE_NAME = "test_ConfigTest.ini";

static ConfigDescription
createConfig()
{
   ConfigDescription config;
   config.addEntry< int >( "int", "Signed integer.", 1 );
   config.addEntry< std::size_t >( "size", "Unsigned integer.", 1 );
   config.addEntry< double >( "real", "Real number.", 1.0 );
   config.addEntry< std::string >( "string", "String.", "a" );
   config.addList< int >( "int-list", "List of signed integers.", { 1 } );
   config.addList< std::size_t >( "size-list", "List of unsigned integers.", { 1 } );
   config.addList< double >( "real-list", "List of real numbers.", { 1.0 } );
   config.addList< std::string >( "string-list", "List of strings.", { "a" } );
   config.addList< std::string >( "enum-list", "List of enumerated strings.", { "x" } );
   config.addEntryEnum( "x" );
   config.addEntryEnum( "y" );
   return config;
}

// Parses the command line given as a list of arguments (without the program name).
static bool
parseArguments( std::vector< std::string > args, const ConfigDescription& config, ParameterContainer& parameters )
{
   std::string program = "ConfigTest";
   std::vector< char* > argv{ program.data() };
   for( auto& arg : args )
      argv.push_back( arg.data() );
   return parseCommandLine( static_cast< int >( argv.size() ), argv.data(), config, parameters, false );
}

// Writes the content into the test INI file and parses it.
static ParameterContainer
parseINI( const std::string& content, const ConfigDescription& config )
{
   std::ofstream( TEST_FILE_NAME ) << content;
   try {
      ParameterContainer parameters = parseINIConfigFile( TEST_FILE_NAME, config );
      std::remove( TEST_FILE_NAME );
      return parameters;
   }
   catch( ... ) {
      std::remove( TEST_FILE_NAME );
      throw;
   }
}

TEST( ConfigTest, commandLineValues )
{
   const ConfigDescription config = createConfig();
   ParameterContainer parameters;
   ASSERT_TRUE( parseArguments(
      { "--int",
        "-3",
        "--size",
        "5",
        "--int-list",
        "-1",
        "2",
        "--size-list",
        "3",
        "4",
        "--real-list",
        "0.5",
        "1.5",
        "--string-list",
        "a",
        "b",
        "--enum-list",
        "x",
        "y" },
      config,
      parameters ) );
   EXPECT_EQ( parameters.getParameter< int >( "int" ), -3 );
   EXPECT_EQ( parameters.getParameter< std::size_t >( "size" ), 5 );
   EXPECT_EQ( parameters.getList< int >( "int-list" ), ( std::vector< int >{ -1, 2 } ) );
   EXPECT_EQ( parameters.getList< std::size_t >( "size-list" ), ( std::vector< std::size_t >{ 3, 4 } ) );
   EXPECT_EQ( parameters.getList< double >( "real-list" ), ( std::vector< double >{ 0.5, 1.5 } ) );
   EXPECT_EQ( parameters.getList< std::string >( "string-list" ), ( std::vector< std::string >{ "a", "b" } ) );
   EXPECT_EQ( parameters.getList< std::string >( "enum-list" ), ( std::vector< std::string >{ "x", "y" } ) );
}

TEST( ConfigTest, commandLineRejectsNegativeUnsigned )
{
   const ConfigDescription config = createConfig();
   for( const std::vector< std::string >& args :
        {
           std::vector< std::string >{ "--size", "-1" },
           std::vector< std::string >{ "--size", " -1" },
           std::vector< std::string >{ "--size-list", "1", "-2" },
        } )
   {
      ParameterContainer parameters;
      EXPECT_FALSE( parseArguments( args, config, parameters ) ) << args[ 0 ] << " " << args.back();
   }
}

TEST( ConfigTest, commandLineRejectsInvalidEnumValue )
{
   const ConfigDescription config = createConfig();
   ParameterContainer parameters;
   EXPECT_FALSE( parseArguments( { "--enum-list", "x", "z" }, config, parameters ) );
}

TEST( ConfigTest, iniValues )
{
   const ConfigDescription config = createConfig();
   const ParameterContainer parameters = parseINI(
      "int = -3\n"
      "size = 5\n"
      "real = 0.25\n"
      "string = b\n"
      "int-list = -1,2\n"
      "size-list = 3,4\n"
      "real-list = 0.5,1.5\n"
      "string-list = a,b\n"
      "enum-list = x,y\n",
      config );
   EXPECT_EQ( parameters.getParameter< int >( "int" ), -3 );
   EXPECT_EQ( parameters.getParameter< std::size_t >( "size" ), 5 );
   EXPECT_EQ( parameters.getParameter< double >( "real" ), 0.25 );
   EXPECT_EQ( parameters.getParameter< std::string >( "string" ), "b" );
   EXPECT_EQ( parameters.getList< int >( "int-list" ), ( std::vector< int >{ -1, 2 } ) );
   EXPECT_EQ( parameters.getList< std::size_t >( "size-list" ), ( std::vector< std::size_t >{ 3, 4 } ) );
   EXPECT_EQ( parameters.getList< double >( "real-list" ), ( std::vector< double >{ 0.5, 1.5 } ) );
   EXPECT_EQ( parameters.getList< std::string >( "string-list" ), ( std::vector< std::string >{ "a", "b" } ) );
   EXPECT_EQ( parameters.getList< std::string >( "enum-list" ), ( std::vector< std::string >{ "x", "y" } ) );
}

TEST( ConfigTest, iniDefaultValues )
{
   const ConfigDescription config = createConfig();
   const ParameterContainer parameters = parseINI( "int = 2\n", config );
   EXPECT_EQ( parameters.getParameter< int >( "int" ), 2 );
   EXPECT_EQ( parameters.getParameter< std::size_t >( "size" ), 1 );
   EXPECT_EQ( parameters.getList< std::size_t >( "size-list" ), ( std::vector< std::size_t >{ 1 } ) );
}

TEST( ConfigTest, iniRejectsNegativeUnsigned )
{
   const ConfigDescription config = createConfig();
   EXPECT_THROW( (void) parseINI( "size = -1\n", config ), Exceptions::ConfigError );
   EXPECT_THROW( (void) parseINI( "size-list = 1,-2\n", config ), Exceptions::ConfigError );
}

TEST( ConfigTest, iniRejectsInvalidEnumValue )
{
   const ConfigDescription config = createConfig();
   EXPECT_THROW( (void) parseINI( "enum-list = x,z\n", config ), Exceptions::ConfigError );
}

#include "main.h"
