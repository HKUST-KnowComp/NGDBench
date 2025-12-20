
OPTIONAL MATCH (n:Account{id:%d})<-[e:transfer]-(m:Account) 
WHERE e.amount > %f AND e.timestamp > %d AND e.timestamp < %d AND m.isBlocked=true 
WITH count(m) * 1.0 as numM 
OPTIONAL MATCH (n:Account{id:%d})<-[e:transfer]-(m:Account) 
WITH count(m) as numIn, numM 
RETURN CASE WHEN numIn = 0 THEN -1 ELSE round(numM / numIn * 1000) / 1000 END as blockRatio;