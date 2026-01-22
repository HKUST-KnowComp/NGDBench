
MATCH (n:Account{id:21990232556190}) WITH n 
OPTIONAL MATCH (n)-[e:transfer]->(m:Account) WHERE e.timestamp > 1625097600000 AND e.timestamp < 1629936000000 WITH n, sum(e.amount) as sumEdge1Amount, max(e.amount) as maxEdge1Amount, count(e) as numEdge1 
OPTIONAL MATCH (n)<-[e:transfer]-(m:Account) WHERE e.timestamp > 1625097600000 AND e.timestamp < 1629936000000 WITH sumEdge1Amount, maxEdge1Amount, numEdge1, sum(e.amount) as sumEdge2Amount, max(e.amount) as maxEdge2Amount, count(e) as numEdge2 
RETURN round(sumEdge1Amount * 1000) / 1000 as sumEdge1Amount, CASE WHEN maxEdge1Amount < 0 THEN -1 ELSE round(maxEdge1Amount * 1000) / 1000 END as maxEdge1Amount, numEdge1, round(sumEdge2Amount * 1000) / 1000 as sumEdge2Amount, CASE WHEN maxEdge2Amount < 0 THEN -1 ELSE round(maxEdge2Amount * 1000) / 1000 END as maxEdge2Amount, numEdge2;
