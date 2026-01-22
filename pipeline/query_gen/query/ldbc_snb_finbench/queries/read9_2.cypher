
MATCH (mid:Account {id:160440736725075484}) 
WITH mid 
OPTIONAL MATCH (mid)-[edge2:repay]->(loan:Loan) WHERE edge2.amount > 0.000000 AND edge2.timestamp > 1617235200000 AND edge2.timestamp < 1620864000000 WITH mid, sum(edge2.amount) AS edge2Amount 
OPTIONAL MATCH (mid)<-[edge1:deposit]-(loan:Loan) WHERE edge1.amount > 0.000000 AND edge1.timestamp > 1617235200000 AND edge1.timestamp < 1620864000000 WITH mid, sum(edge1.amount) AS edge1Amount, edge2Amount 
OPTIONAL MATCH (mid)-[edge4:transfer]->(down:Account) WHERE edge4.amount > 0.000000 AND edge4.timestamp > 1617235200000 AND edge4.timestamp < 1620864000000 WITH mid, edge1Amount, edge2Amount, sum(edge4.amount) AS edge4Amount 
OPTIONAL MATCH (mid)<-[edge3:transfer]-(up:Account) WHERE edge3.amount > 0.000000 AND edge3.timestamp > 1617235200000 AND edge3.timestamp < 1620864000000 
WITH edge1Amount, edge2Amount, sum(edge3.amount) AS edge3Amount, edge4Amount 
RETURN CASE WHEN edge2Amount=0 THEN -1 ELSE round(1000.0 * edge1Amount / edge2Amount) / 1000 END AS ratioRepay, CASE WHEN edge4Amount=0 THEN -1 ELSE round(1000.0 * edge1Amount / edge4Amount) / 1000 END AS ratioDeposit, CASE WHEN edge4Amount=0 THEN -1 ELSE round(1000.0 * edge3Amount / edge4Amount) / 1000 END AS ratioTransfer;