
MATCH (person:Person {id:26388279067160})-[edge1:own]->(pAcc:Account) -[edge2:transfer]->(compAcc:Account) <-[edge3:own]-(com:Company) 
WHERE edge2.timestamp > 1619827200000 AND edge2.timestamp < 1622246400000 
WITH compAcc.id AS compAccountId, sum(edge2.amount) AS sumEdge2Amount 
RETURN compAccountId, round(sumEdge2Amount * 1000) / 1000 as sumEdge2Amount 
ORDER BY sumEdge2Amount DESC;