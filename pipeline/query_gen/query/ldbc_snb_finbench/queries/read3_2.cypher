/*
Compute the shortest path of transfer edges from a source account to a destination account, 
using only transfers whose timestamps fall within a given time window (greater than one bound 
and smaller than another). Return the number of edges in that shortest path.
*/
MATCH (src:Account{id:129478489286901859}), (dst:Account{id:4702320960928220657}) 
CALL algo.shortestPath( src, dst, { direction: 'PointingRight', relationshipQuery:'transfer', edgeFilter: { timestamp: { smaller_than: 1649289600000, greater_than: 1643673600000 } } } ) 
YIELD nodeCount 
RETURN nodeCount - 1 AS len;