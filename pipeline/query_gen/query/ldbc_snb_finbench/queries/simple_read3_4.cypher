
OPTIONAL MATCH (n:Account{id:213358032346678831})<-[e:transfer]-(m:Account) 
WHERE e.amount > 0.000000 AND e.timestamp > 244320279784850805 AND e.timestamp < 1635724800000 AND m.isBlocked=true 
WITH count(m) * 1.0 as numM 
OPTIONAL MATCH (n:Account{id:1643760000000})<-[e:transfer]-(m:Account) 
WITH count(m) as numIn, numM 
RETURN CASE WHEN numIn = 0 THEN -1 ELSE round(numM / numIn * 1000) / 1000 END as blockRatio;