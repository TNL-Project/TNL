#pragma once

template< typename Index, typename Real >
void makeSymmetric(  std::map< std::pair< Index, Index >, Real >& map )
{
   for( auto& [ key, value ] : map )
   {
      if( map.count( { key.second, key.first } ) == 0 )
         map.insert( { { key.second, key.first }, value } );
   }
}
