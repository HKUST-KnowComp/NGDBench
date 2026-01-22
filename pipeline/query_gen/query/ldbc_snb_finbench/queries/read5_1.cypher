/*
From a given person, take each account they own and find all accounts reachable through 1–3 transfer 
steps whose timestamps are in strict ascending order and fall within a specified time window.
For each valid path, return the sequence of account IDs, keeping only paths that contain no 
repeated accounts, and sort them by path length in descending order.
*/
MATCH (person:Person {id:13194139533470})-[e1:own]->(src:Account) 
WITH src 
MATCH p=(src)-[e2:transfer*1..3]->(dst:Account) 
WHERE isAsc(getMemberProp(e2, 'timestamp'))=true 
  AND head(getMemberProp(e2, 'timestamp')) > 1638316800000 
  AND last(getMemberProp(e2, 'timestamp')) < 1640736000000 
WITH DISTINCT getMemberProp(nodes(p), 'id') as path, length(p) as len 
ORDER BY len DESC 
WHERE hasDuplicates(path)=false 
RETURN path;