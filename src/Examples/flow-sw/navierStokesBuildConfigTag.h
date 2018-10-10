#ifndef navierStokesBUILDCONFIGTAG_H_
#define navierStokesBUILDCONFIGTAG_H_

#include <TNL/Solvers/BuildConfigTags.h>

namespace TNL {

class navierStokesBuildConfigTag{};

namespace Solvers {

/****
 * Turn off support for float and long double.
 */
template<> struct ConfigTagReal< navierStokesBuildConfigTag, float > { enum { enabled = false }; };
template<> struct ConfigTagReal< navierStokesBuildConfigTag, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct ConfigTagIndex< navierStokesBuildConfigTag, short int >{ enum { enabled = false }; };
template<> struct ConfigTagIndex< navierStokesBuildConfigTag, long int >{ enum { enabled = false }; };

//template< int Dimension > struct ConfigTagDimension< navierStokesBuildConfigTag, Dimension >{ enum { enabled = ( Dimension == 1 ) }; };

/****
 * Use of Grid is enabled for allowed dimensions and Real, Device and Index types.
 */
/*
template< int Dimension, typename Real, typename Device, typename Index >
   struct ConfigTagMesh< navierStokesBuildConfigTag, Meshes::Grid< Dimension, Real, Device, Index > >
      { enum { enabled = ConfigTagDimension< navierStokesBuildConfigTag, Dimension >::enabled  &&
                         ConfigTagReal< navierStokesBuildConfigTag, Real >::enabled &&
                         ConfigTagDevice< navierStokesBuildConfigTag, Device >::enabled &&
                         ConfigTagIndex< navierStokesBuildConfigTag, Index >::enabled }; };
*/

/****
 * Please, chose your preferred time discretisation  here.
 */
template<> struct ConfigTagTimeDiscretisation< navierStokesBuildConfigTag, ExplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct ConfigTagTimeDiscretisation< navierStokesBuildConfigTag, SemiImplicitTimeDiscretisationTag >{ enum { enabled = false }; };
template<> struct ConfigTagTimeDiscretisation< navierStokesBuildConfigTag, ImplicitTimeDiscretisationTag >{ enum { enabled = false }; };

/****
 * Only the Runge-Kutta-Merson solver is enabled by default.
 */
template<> struct ConfigTagExplicitSolver< navierStokesBuildConfigTag, ExplicitEulerSolverTag >{ enum { enabled = true }; };

} // namespace Solvers

namespace Meshes {
namespace BuildConfigTags {

template< int Dimensions > struct GridDimensionTag< navierStokesBuildConfigTag, Dimensions >{ enum { enabled = ( Dimensions == 1 ) }; };

/****
 * Turn off support for float and long double.
 */
template<> struct GridRealTag< navierStokesBuildConfigTag, float > { enum { enabled = false }; };
template<> struct GridRealTag< navierStokesBuildConfigTag, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct GridIndexTag< navierStokesBuildConfigTag, short int >{ enum { enabled = false }; };
template<> struct GridIndexTag< navierStokesBuildConfigTag, long int >{ enum { enabled = false }; };

} // namespace BuildConfigTags
} // namespace Meshes

} // namespace TNL

#endif /* navierStokesBUILDCONFIGTAG_H_ */
