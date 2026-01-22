/*
Compute the shortest path of transfer edges from a source account to a destination account, 
using only transfers whose timestamps fall within a given time window (greater than one bound 
and smaller than another). Return the number of edges in that shortest path.
*/
MATCH (src:Account{id:213358032346678831}), (dst:Account{id:244320279784850805}) 
CALL algo.shortestPath( src, dst, { direction: 'PointingRight', relationshipQuery:'transfer', edgeFilter: { timestamp: { smaller_than: 1643760000000, greater_than: 1635724800000 } } } ) 
YIELD nodeCount 
RETURN nodeCount - 1 AS len;