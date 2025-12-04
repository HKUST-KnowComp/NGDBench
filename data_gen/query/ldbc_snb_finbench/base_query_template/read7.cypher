
MATCH (mid:Account {id:%d}) 
WITH mid 
OPTIONAL MATCH (mid)-[edge2:transfer]->(dst:Account) 
WHERE edge2.timestamp > %d AND edge2.timestamp < %d AND edge2.amount > %f 
WITH mid, count(distinct dst) as numDst, sum(edge2.amount) as amountDst 
OPTIONAL MATCH (mid)<-[edge1:transfer]-(src:Account) 
WHERE edge1.timestamp > %d AND edge1.timestamp < %d AND edge1.amount > %f 
WITH count(distinct src) as numSrc, sum(edge1.amount) as amountSrc, numDst, amountDst 
RETURN numSrc, numDst, CASE WHEN amountDst=0 THEN -1 ELSE round(1000.0 * amountSrc / amountDst) / 1000 END AS inOutRatio;