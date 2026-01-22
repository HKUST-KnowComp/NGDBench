
MATCH (person:Person {id:6597069766803})-[edge1:own]->(pAcc:Account) -[edge2:transfer]->(compAcc:Account) <-[edge3:own]-(com:Company) 
WHERE edge2.timestamp > 1635724800000 AND edge2.timestamp < 1639353600000 
WITH compAcc.id AS compAccountId, sum(edge2.amount) AS sumEdge2Amount 
RETURN compAccountId, round(sumEdge2Amount * 1000) / 1000 as sumEdge2Amount 
ORDER BY sumEdge2Amount DESC;