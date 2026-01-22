## Basic query

MATCH (n:$LABEL)
RETURN n

MATCH (n:$LABEL {$PROP: $VALUE})-[:$REL_TYPE]->(m) RETURN m

MATCH (a:$LABEL1)-[:$REL1]->(b:$LABEL2)
MATCH (a)-[:$REL2]->(b)
RETURN a, b

MATCH (a:$LABEL1)-[r:$REL_TYPE]-(b:$LABEL2)
WHERE r.$REL_PROP $OP $VALUE
RETURN a

## nested loop pattern
MATCH (a:$LABEL1 {$PROP1: $VALUE})
MATCH (b:$LABEL2 {$PROP2: a.$REF_PROP})
RETURN a, b

MATCH (a:$LABEL1)
WHERE (a)-[:$REL_TYPE]->(:$LABEL2)
RETURN a.$RETURN_PROP


MATCH
  (a:$LABEL1 {$PROP1: $VALUE}),
  (b:$LABEL2)
WHERE NOT (a)-[:$REL_TYPE]->(b)
RETURN b.$RETURN_PROP

MATCH (n:$LABEL)
WHERE (n)-[:$REL1]->(:$L1) OR (n)-[:$REL2]->(:$L2)
RETURN n.$PROP

MATCH (n:$LABEL)
WHERE NOT (n)-[:$REL1]->(:$L1) OR (n)-[:$REL2]->(:$L2)
RETURN n.$PROP

MATCH (n:$LABEL)
WHERE n.$PROP $OP $VAL OR (n)-[:$REL]->(:$L2)
RETURN n.$RET


## Chain pattern 
MATCH (a:$LABEL1)-[:$REL1]->(b:$LABEL2)-[:$REL2]->(c:$LABEL3)-[:$REL3]->(d:$LABEL4)
RETURN a, b, d

MATCH (a:$LABEL1)-[:$REL_TYPE *$MIN_HOPS..$MAX_HOPS]-(b:$LABEL2)
RETURN a, b


MATCH (start:$START_LABEL {$START_PROP: $START_VALUE})
MATCH p = (start) (
    (n1:$L1)<-$D1[:$R1]-$D2(r:$REL_NODE_LABEL)-$D3[:$R2]->(n2:$L2)
    WHERE r.$NODE_PROP = $NODE_VALUE
){$MIN_HOPS..$MAX_HOPS} (end:$END_LABEL)
RETURN p

## Join pattern 

MATCH
  (a:$LABEL1),
  (b:$LABEL2)
RETURN a, b

MATCH
  (a:$LABEL1),
  (b:$LABEL2)
WHERE a.$PROP1 = b.$PROP2
RETURN a, b

## Pipelined pattern


MATCH (a:$LABEL1 {$PROP1: $VALUE})-[:$REL1]->(b:$LABEL2)
WITH b
MATCH (b)-[:$REL2]->(c:$LABEL3)
WHERE c.$FILTER_PROP = $FILTER_VAL
RETURN b.$RET_PROP1, c.$RET_PROP2, c.$RET_PROP3


## Anti Pattern

MATCH (a:$LABEL)
WHERE NOT a.$PROP $OP $VALUE
RETURN a.$RET_PROP

MATCH
  (a:$LABEL1 {$PROP1: $VALUE}),
  (b:$LABEL2)
WHERE NOT (a)-[:$REL_TYPE]->(b)
RETURN b.$RETURN_PROP

MATCH (a:$LABEL)
WHERE NOT (a)-[:$REL1]->(:$LABEL2)
   OR  (a)-[:$REL2]->(:$LABEL3)
RETURN a.$RET_PROP

## union pattern

MATCH (a:$L1)
RETURN a.$P
UNION
MATCH (b:$L2)
RETURN b.$P

## Triadic Pattern

MATCH (a:$LABEL1)-[:$REL]-()-[:$REL]-(b:$LABEL2)
WHERE NOT (a)-[:$REL]-(b)
RETURN b.$PROP

## scan/seek + agg
MATCH (a:$LABEL)
RETURN avg(a.$PROP) AS avg_value

MATCH (g:$GROUP_LABEL)<-[:$REL]-(n:$NODE_LABEL)
RETURN
  g.$GROUP_PROP AS $GROUP_ALIAS,
  collect(n.$NODE_PROP) AS $COLLECT_ALIAS

MATCH (a:$LABEL)
RETURN sum(a.$PROP) AS total_value

MATCH (a:$LABEL)
RETURN min(a.$PROP) AS min_value

MATCH (a:$LABEL)
RETURN max(a.$PROP) AS max_value

MATCH
  (a:$LABEL1 {$PROP1: $VALUE}),
  (b:$LABEL2)
WHERE NOT (a)-[:$REL]->(b)
RETURN b.$RET_PROP

## scan/seek expand + agg

MATCH (n:$LABEL {$PROP: $VALUE})-[:$REL_TYPE]->(m)
RETURN count(m) AS cnt

MATCH (n:$LABEL {$PROP: $VALUE})-[:$REL_TYPE]->(m:$L2)
RETURN avg(m.$PROP2) AS avg_value

MATCH (n:$LABEL {$PROP: $VALUE})-[:$REL_TYPE]->(m)
RETURN sum(m.$PROP2) AS total

## scan/seek + expand into
MATCH (a:$LABEL1)-[:$REL1]->(b:$LABEL2)
MATCH (a)-[:$REL2]->(b)
RETURN count(*) AS cnt

MATCH (a:$LABEL1)-[:$REL1]->(b:$LABEL2)
MATCH (a)-[:$REL2]->(b)
RETURN a, count(b) AS cnt

## scan/seek + filter
MATCH (a:$LABEL1)-[r:$REL_TYPE]-(b:$LABEL2)
WHERE r.$REL_PROP $OP $VALUE
RETURN count(a) AS cnt

MATCH (a:$LABEL1)-[r:$REL_TYPE]-(b:$LABEL2)
WHERE r.$REL_PROP $OP $VALUE
RETURN a, count(b) AS cnt

MATCH (a:$LABEL1)-[r:$REL_TYPE]-(b:$LABEL2)
WHERE r.$REL_PROP $OP $VALUE
RETURN avg(r.$REL_PROP), max(r.$REL_PROP)

## Nested loop + avg
MATCH (a:$LABEL1 {$PROP1: $VALUE})
MATCH (b:$LABEL2 {$PROP2: a.$REF_PROP})
RETURN count(b) AS cnt

MATCH (a:$LABEL1 {$PROP1: $VALUE})
MATCH (b:$LABEL2 {$PROP2: a.$REF_PROP})
RETURN a, collect(b.$P) AS bs

MATCH (a:$LABEL1)
WHERE (a)-[:$REL_TYPE]->(:$LABEL2)
RETURN count(a) AS cnt

MATCH (a:$LABEL1)
WHERE (a)-[:$REL_TYPE]->(:$LABEL2)
RETURN collect(a.$RETURN_PROP) AS values

MATCH (a:$LABEL1 {$PROP1: $VALUE}),
      (b:$LABEL2)
WHERE NOT (a)-[:$REL_TYPE]->(b)
RETURN count(b) AS cnt

MATCH (n:$LABEL)
WHERE (n)-[:$REL1]->(:$L1) OR (n)-[:$REL2]->(:$L2)
RETURN count(n) AS cnt

MATCH (n:$LABEL)
WHERE NOT (n)-[:$REL1]->(:$L1)
   OR (n)-[:$REL2]->(:$L2)
RETURN count(n)

MATCH (n:$LABEL)
WHERE n.$PROP $OP $VAL OR (n)-[:$REL]->(:$L2)
RETURN n.$RET

MATCH (n:$LABEL)
WHERE n.$PROP $OP $VAL OR (n)-[:$REL]->(:$L2)
RETURN avg(n.$NUM_PROP)

## Chain pattern  + avg
MATCH (a:$LABEL1)-[:$REL1]->(b:$LABEL2)-[:$REL2]->(c:$LABEL3)-[:$REL3]->(d:$LABEL4)
RETURN count(d) AS cnt

MATCH (a:$LABEL1)-[:$REL1]->(b:$LABEL2)-[:$REL2]->(c:$LABEL3)-[:$REL3]->(d:$LABEL4)
RETURN a, count(d) AS cnt

MATCH (a:$LABEL1)-[:$REL1]->(b:$LABEL2)-[:$REL2]->(c:$LABEL3)-[:$REL3]->(d:$LABEL4)
RETURN avg(d.$NUM_PROP)

MATCH (a:$LABEL1)-[:$REL1]->(b:$LABEL2)-[:$REL2]->(c:$LABEL3)-[:$REL3]->(d:$LABEL4)
RETURN a, collect(d.$PROP) AS ds

MATCH (a:$LABEL1)-[:$REL_TYPE *$MIN_HOPS..$MAX_HOPS]-(b:$LABEL2)
RETURN count(b) AS cnt

MATCH (a:$LABEL1)-[:$REL_TYPE *$MIN_HOPS..$MAX_HOPS]-(b:$LABEL2)
RETURN a, count(b) AS cnt

MATCH p = (a:$LABEL1)-[:$REL_TYPE *$MIN_HOPS..$MAX_HOPS]-(b:$LABEL2)
RETURN avg(length(p)) AS avg_len

MATCH (start:$START_LABEL {$START_PROP: $START_VALUE})
MATCH p = (start) (
    (n1:$L1)<-[:$R1]-(r:$REL_NODE_LABEL)-[:$R2]->(n2:$L2)
    WHERE r.$NODE_PROP = $NODE_VALUE
){$MIN_HOPS..$MAX_HOPS} (end:$END_LABEL)
RETURN count(p) AS path_count

MATCH (start:$START_LABEL {$START_PROP: $START_VALUE})
MATCH p = (start) (...){$MIN_HOPS..$MAX_HOPS} (end:$END_LABEL)
RETURN avg(length(p)) AS avg_len

## join + avg
MATCH (a:$LABEL1)
WITH a
ORDER BY a.$PROP1
LIMIT $K1

MATCH (b:$LABEL2)
WITH a, b
ORDER BY b.$PROP2
LIMIT $K2
RETURN count(*) AS pair_count

MATCH
  (a:$LABEL1),
  (b:$LABEL2)
WHERE a.$PROP1 = b.$PROP2
RETURN count(*) AS cnt

MATCH
  (a:$LABEL1),
  (b:$LABEL2)
WHERE a.$PROP1 = b.$PROP2
RETURN a, collect(b.$PROP) AS bs

MATCH
  (a:$LABEL1),
  (b:$LABEL2)
WHERE a.$PROP1 = b.$PROP2
WITH a, count(b) AS cnt
WHERE cnt > $K
RETURN a


## pipeline + avg
MATCH (a:$LABEL1 {$PROP1: $VALUE})-[:$REL1]->(b:$LABEL2)
WITH b
MATCH (b)-[:$REL2]->(c:$LABEL3)
WHERE c.$FILTER_PROP = $FILTER_VAL
RETURN count(*) AS total_cnt

MATCH (a:$LABEL1 {$PROP1: $VALUE})-[:$REL1]->(b:$LABEL2)
WITH b
MATCH (b)-[:$REL2]->(c:$LABEL3)
WHERE c.$FILTER_PROP = $FILTER_VAL
RETURN b, count(c) AS cnt

MATCH (a:$LABEL1 {$PROP1: $VALUE})-[:$REL1]->(b:$LABEL2)
WITH b
MATCH (b)-[:$REL2]->(c:$LABEL3)
WHERE c.$FILTER_PROP = $FILTER_VAL
WITH b, count(c) AS cnt
WHERE cnt > $THRESHOLD
RETURN b, cnt

## anti avg
MATCH (a:$LABEL)
WHERE NOT a.$PROP $OP $VALUE
RETURN count(a) AS cnt

MATCH (a:$LABEL)
WHERE NOT a.$PROP $OP $VALUE
RETURN collect(a.$RET_PROP) AS vals

MATCH
  (a:$LABEL1 {$PROP1: $VALUE}),
  (b:$LABEL2)
WHERE NOT (a)-[:$REL_TYPE]->(b)
RETURN count(b) AS cnt

MATCH
  (a:$LABEL1 {$PROP1: $VALUE}),
  (b:$LABEL2)
WHERE NOT (a)-[:$REL_TYPE]->(b)
RETURN a, count(b) AS cnt

MATCH (a:$LABEL)
WHERE NOT (a)-[:$REL1]->(:$LABEL2)
   OR  (a)-[:$REL2]->(:$LABEL3)
RETURN count(a) AS cnt

MATCH (a:$LABEL)
WHERE NOT (a)-[:$REL1]->(:$LABEL2)
   OR  (a)-[:$REL2]->(:$LABEL3)
RETURN collect(a.$RET_PROP) AS vals

## union avg

MATCH (a:$LABEL1)-[:$REL1]->(b:$LABEL2)
WITH a, $AGG_FUNC1(b.$NUM_PROP1) AS metric1
WHERE metric1 $OP1 $THRESHOLD1
RETURN a.$RET_PROP1 AS EntityName,
       '$CATEGORY1' AS Category,
       metric1 AS ValueScore
UNION ALL
MATCH (c:$LABEL3)<-[:$REL2]-(d:$LABEL4)
WITH c, $AGG_FUNC2(d.$NUM_PROP2) AS metric2
WHERE metric2 $OP2 $THRESHOLD2
RETURN c.$RET_PROP2 AS EntityName,
       '$CATEGORY2' AS Category,
       metric2 AS ValueScore


MATCH (a:$L1)
RETURN count(a.$P) AS cnt
UNION
MATCH (b:$L2)
RETURN count(b.$P) AS cnt


MATCH (p:Person)
RETURN count(p.name) AS cnt
UNION
MATCH (c:Company)
RETURN count(c.name) AS cnt

## triadic avg
MATCH (a:$LABEL1)-[:$REL]-()-[:$REL]-(b:$LABEL2)
WHERE NOT (a)-[:$REL]-(b)
WITH a, count(DISTINCT b) AS cnt
RETURN a.$PROP, cnt
