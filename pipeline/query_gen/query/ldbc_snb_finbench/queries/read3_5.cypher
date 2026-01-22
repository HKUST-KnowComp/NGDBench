/*
Compute the shortest path of transfer edges from a source account to a destination account, 
using only transfers whose timestamps fall within a given time window (greater than one bound 
and smaller than another). Return the number of edges in that shortest path.
*/
MATCH (src:Account{id:235313080530109978}), (dst:Account{id:238409305273925764}) 
CALL algo.shortestPath( src, dst, { direction: 'PointingRight', relationshipQuery:'transfer', edgeFilter: { timestamp: { smaller_than: 1642291200000, greater_than: 1638316800000 } } } ) 
YIELD nodeCount 
RETURN nodeCount - 1 AS len;