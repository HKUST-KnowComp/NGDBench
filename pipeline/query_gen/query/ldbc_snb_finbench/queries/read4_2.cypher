/*
First, identify a direct transfer from a source account to a destination account within a given 
timestamp range. Then find all intermediate “other” accounts that transferred money to the source 
and also transferred money to the destination, each within their own timestamp windows.

For each such intermediate account, count and aggregate the transfers from it to the source (number, 
sum, and max amount) and also the transfers from the destination to it (number, sum, and max amount).
Return these statistics per intermediate account, sorted by the sum of transfers to the source 
(descending), then the sum of transfers from the destination (descending), and then the account ID.
*/
MATCH (src:Account {id:129478489286901859})-[e1:transfer]->(dst:Account {id:291045125918819487}) 
WHERE e1.timestamp > 1643673600000 AND e1.timestamp < 1649289600000 
WITH src, dst.id as dstid 
MATCH (src)<-[e2:transfer]-(other:Account)<-[e3:transfer]-(dst:Account) 
WHERE dst.id=dstid AND e2.timestamp > 1643673600000 AND e2.timestamp < 1649289600000 AND e3.timestamp > 1643673600000 AND e3.timestamp < 1649289600000 
WITH DISTINCT src, other, dst 
MATCH (src)<-[e2:transfer]-(other) 
WHERE e2.timestamp > 1643673600000 AND e2.timestamp < 1649289600000 
WITH src, other, dst, count(e2) as numEdge2, sum(e2.amount) as sumEdge2Amount, max(e2.amount) as maxEdge2Amount 
MATCH (other)<-[e3:transfer]-(dst) 
WHERE e3.timestamp > 1643673600000 AND e3.timestamp < 1649289600000 
WITH other.id as otherId, numEdge2, sumEdge2Amount, maxEdge2Amount, count(e3) as numEdge3, sum(e3.amount) as sumEdge3Amount, max(e3.amount) as maxEdge3Amount 
RETURN otherId, numEdge2, round(sumEdge2Amount * 1000) / 1000 as sumEdge2Amount, round(maxEdge2Amount * 1000) / 1000 as maxEdge2Amount, numEdge3, round(sumEdge3Amount * 1000) / 1000 as sumEdge3Amount, round(maxEdge3Amount * 1000) / 1000 as maxEdge3Amount 
ORDER BY sumEdge2Amount DESC, sumEdge3Amount DESC, otherId ASC;