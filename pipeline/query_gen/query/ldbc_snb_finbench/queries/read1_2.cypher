/*
From a given account, find other accounts reachable through 1–3 transfer steps whose timestamps are in
ascending order and fall within a specified time window. Then keep only those reachable accounts that 
were signed in by a blocked medium during another given timestamp window.
Return the other account’s ID, the transfer distance, and the medium’s ID and type, ordered by distance
and IDs.
*/

MATCH p = (acc:Account {id:201536083324830776})-[e1:transfer *1..3]->(other:Account)<-[e2:signIn]-(medium) 
WHERE isAsc(getMemberProp(e1, 'timestamp'))=true 
  AND head(getMemberProp(e1, 'timestamp')) > 1643673600000 
  AND last(getMemberProp(e1, 'timestamp')) < 1647302400000 
  AND e2.timestamp > 1643673600000 
  AND e2.timestamp < 1647302400000 
  AND medium.isBlocked = true 
RETURN DISTINCT other.id as otherId, length(p)-1 as accountDistance, medium.id as mediumId, medium.type as mediumType 
ORDER BY accountDistance, otherId, mediumId;