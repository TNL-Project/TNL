include(FetchContent)
FetchContent_Declare(
   tinyxml2
   GIT_REPOSITORY https://github.com/leethomason/tinyxml2.git
   GIT_TAG        9.0.0
   FIND_PACKAGE_ARGS NAMES tinyxml2
   SYSTEM
)
FetchContent_MakeAvailable(tinyxml2)
