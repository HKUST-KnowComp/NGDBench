
MATCH (n:Account{id:%d}) RETURN n.createTime as createTime, n.isBlocked as isBlocked, n.type as type;