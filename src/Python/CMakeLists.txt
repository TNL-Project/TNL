find_package( Python 3 COMPONENTS Interpreter )

set( PYTHON_SITE_PACKAGES_DIR lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages )

if( Python_Interpreter_FOUND )
   configure_file( "__init__.py.in" "__init__.py" )
   install( FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
                  BenchmarkLogs.py
                  BenchmarkPlots.py
            DESTINATION ${PYTHON_SITE_PACKAGES_DIR}/TNL
            COMPONENT headers )
endif()
