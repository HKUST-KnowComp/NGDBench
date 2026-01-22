
OPTIONAL MATCH (p1:Person {id:1651363200000})-[edge1:invest]->(m1:Company) 
WHERE edge1.timestamp > 1653782400000 AND edge1.timestamp < 1651363200000 
WITH collect(distinct id(m1)) as m1_vids 
OPTIONAL MATCH (p2:Person {id:1653782400000})-[edge2:invest]->(m2:Company) 
WHERE edge2.timestamp > 1651363200000 AND edge2.timestamp < 1653782400000 
WITH collect(distinct id(m2)) as m2_vids, m1_vids 
CALL algo.jaccard(m1_vids, m2_vids) YIELD similarity 
RETURN CASE WHEN similarity = null THEN 0 ELSE round(similarity*1000)/1000 END AS similarity;