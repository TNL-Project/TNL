#ifndef FLOWSBUILDCONFIGTAG_H_
#define FLOWSBUILDCONFIGTAG_H_

#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Meshes/BuildConfigTags.h>

namespace TNL {

class flowsBuildConfigTag{};

namespace Solvers {

/****
 * Turn off support for float and long double.
 */
template<> struct ConfigTagReal< flowsBuildConfigTag, float > { enum { enabled = false }; };
template<> struct ConfigTagReal< flowsBuildConfigTag, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct ConfigTagIndex< flowsBuildConfigTag, short int >{ enum { enabled = false }; };
template<> struct ConfigTagIndex< flowsBuildConfigTag, long int >{ enum { enabled = false }; };

//template< int Dimension > struct ConfigTagDimension< flowsBuildConfigTag, Dimension >{ enum { enabled = ( Dimension == 1 ) }; };

/****
 * Use of Grid is enabled for allowed dimensions and Real, Device and Index types.
 */
/*template< int Dimension, typename Real, typename Device, typename Index >
   struct ConfigTagMesh< flowsBuildConfigTag, Meshes::Grid< Dimension, Real, Device, Index > >
      { enum { enabled = ConfigTagDimension< flowsBuildConfigTag, Dimension >::enabled  &&
                         ConfigTagReal< flowsBuildConfigTag, Real >::enabled &&
                         ConfigTagDevice< flowsBuildConfigTag, Device >::enabled &&
                         ConfigTagIndex< flowsBuildConfigTag, Index >::enabled }; };
*/
/****
 * Please, chose your preferred time discretisation  here.
 */
template<> struct ConfigTagTimeDiscretisation< flowsBuildConfigTag, ExplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct ConfigTagTimeDiscretisation< flowsBuildConfigTag, SemiImplicitTimeDiscretisationTag >{ enum { enabled = false }; };
template<> struct ConfigTagTimeDiscretisation< flowsBuildConfigTag, ImplicitTimeDiscretisationTag >{ enum { enabled = false }; };

/****
 * Only the Runge-Kutta-Merson solver is enabled by default.
 */
template<> struct ConfigTagExplicitSolver< flowsBuildConfigTag, ExplicitEulerSolverTag >{ enum { enabled = true }; };

} // namespace Solvers

namespace Meshes {
namespace BuildConfigTags {

template< int Dimensions > struct GridDimensionTag< flowsBuildConfigTag, Dimensions >{ enum { enabled = true }; };

/****
 * Turn off support for float and long double.
 */
template<> struct GridRealTag< flowsBuildConfigTag, float > { enum { enabled = false }; };
template<> struct GridRealTag< flowsBuildConfigTag, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct GridIndexTag< flowsBuildConfigTag, short int >{ enum { enabled = false }; };
template<> struct GridIndexTag< flowsBuildConfigTag, long int >{ enum { enabled = false }; };

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL

#endif /* FLOWSBUILDCONFIGTAG_H_ */
