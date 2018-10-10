#ifndef FLOWSRHS_H_
#define FLOWSRHS_H_

#include <TNL/Functions/Domain.h>

namespace TNL {

template< typename Mesh, typename Real >class flowsRhs
  : public Functions::Domain< Mesh::getMeshDimension(), Functions::MeshDomain > 
 {
   public:

      typedef Mesh MeshType;
      typedef Real RealType;

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         return true;
      }

      template< typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshEntity& entity,
                       const Real& time = 0.0 ) const
      {
         typedef typename MeshEntity::MeshType::PointType PointType;
         PointType v = entity.getCenter();
         return 0.0;
      }
};

} //namespace TNL

#endif /* FLOWSRHS_H_ */
