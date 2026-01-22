
OPTIONAL MATCH (n:Account{id:235313080530109978})<-[e:transfer]-(m:Account) 
WHERE e.amount > 0.000000 AND e.timestamp > 238409305273925764 AND e.timestamp < 1638316800000 AND m.isBlocked=true 
WITH count(m) * 1.0 as numM 
OPTIONAL MATCH (n:Account{id:1642291200000})<-[e:transfer]-(m:Account) 
WITH count(m) as numIn, numM 
RETURN CASE WHEN numIn = 0 THEN -1 ELSE round(numM / numIn * 1000) / 1000 END as blockRatio;