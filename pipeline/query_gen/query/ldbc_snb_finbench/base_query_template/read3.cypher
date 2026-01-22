/*
Compute the shortest path of transfer edges from a source account to a destination account, 
using only transfers whose timestamps fall within a given time window (greater than one bound 
and smaller than another). Return the number of edges in that shortest path.
*/
MATCH (src:Account{id:%d}), (dst:Account{id:%d}) 
CALL algo.shortestPath( src, dst, { direction: 'PointingRight', relationshipQuery:'transfer', edgeFilter: { timestamp: { smaller_than: %d, greater_than: %d } } } ) 
YIELD nodeCount 
RETURN nodeCount - 1 AS len;