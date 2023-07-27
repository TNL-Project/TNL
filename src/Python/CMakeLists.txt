find_package( Python 3 COMPONENTS Interpreter )

set( PYTHON_SITE_PACKAGES_DIR lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages )

if( Python_Interpreter_FOUND )
   CONFIGURE_FILE( "__init__.py.in" "__init__.py" )
   INSTALL( FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
                  BenchmarkLogs.py
                  BenchmarkPlots.py
                  LogParser.py
            DESTINATION ${PYTHON_SITE_PACKAGES_DIR}/TNL )
endif()
