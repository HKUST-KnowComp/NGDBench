
OPTIONAL MATCH (p1:Person {id:1635724800000})-[edge1:invest]->(m1:Company) 
WHERE edge1.timestamp > 1638144000000 AND edge1.timestamp < 1635724800000 
WITH collect(distinct id(m1)) as m1_vids 
OPTIONAL MATCH (p2:Person {id:1638144000000})-[edge2:invest]->(m2:Company) 
WHERE edge2.timestamp > 1635724800000 AND edge2.timestamp < 1638144000000 
WITH collect(distinct id(m2)) as m2_vids, m1_vids 
CALL algo.jaccard(m1_vids, m2_vids) YIELD similarity 
RETURN CASE WHEN similarity = null THEN 0 ELSE round(similarity*1000)/1000 END AS similarity;