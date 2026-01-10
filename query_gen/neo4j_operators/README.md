# Neo4j Cypher 操作符文档

来源: https://neo4j.com/docs/cypher-manual/25/planning-and-tuning/operators/operators-detail/

## 目录结构

### Aggregation operators (4 个操作符)

文件夹: `aggregation-operators/`

- [Eager Aggregation](aggregation-operators/query-plan-eager-aggregation.html)
- [Ordered Aggregation](aggregation-operators/query-plan-ordered-aggregation.html)
- [Node Count From Count Store](aggregation-operators/query-plan-node-count-from-count-store.html)
- [Relationship Count From Count Store](aggregation-operators/query-plan-relationship-count-from-count-store.html)

### Data modification operators (21 个操作符)

文件夹: `data-modification-operators/`

- [Create](data-modification-operators/query-plan-create.html)
- [Delete](data-modification-operators/query-plan-delete.html)
- [Detach Delete](data-modification-operators/query-plan-detach-delete.html)
- [Merge](data-modification-operators/query-plan-merge.html)
- [Merge IntoIntroduced in 2025.11](data-modification-operators/query-plan-merge-into.html)
- [Merge Unique NodeIntroduced in 2025.11](data-modification-operators/query-plan-merge-unique-node.html)
- [Locking Merge](data-modification-operators/query-plan-locking-merge.html)
- [Lock NodesIntroduced in 2025.09](data-modification-operators/query-plan-lock-nodes.html)
- [Foreach](data-modification-operators/query-plan-foreach.html)
- [SubqueryForeach](data-modification-operators/query-plan-subquery-foreach.html)
- [TransactionForeach](data-modification-operators/query-plan-transaction-foreach.html)
- [Set Labels](data-modification-operators/query-plan-set-labels.html)
- [Remove Labels](data-modification-operators/query-plan-remove-labels.html)
- [Set Node Properties From Map](data-modification-operators/query-plan-set-node-properties-from-map.html)
- [Set Relationship Properties From Map](data-modification-operators/query-plan-set-relationship-properties-from-map.html)
- [Set Property](data-modification-operators/query-plan-set-property.html)
- [Set Properties](data-modification-operators/query-plan-set-properties.html)
- [Load CSV](data-modification-operators/query-plan-load-csv.html)
- [Eager](data-modification-operators/query-plan-eager.html)
- [Assert Same Node](data-modification-operators/query-plan-assert-same-node.html)
- [Assert Same Relationship](data-modification-operators/query-plan-assert-same-relationship.html)

### Hash joins in general (3 个操作符)

文件夹: `execution-plans-operators-hash-join-general/`

- [Node Hash Join](execution-plans-operators-hash-join-general/query-plan-node-hash-join.html)
- [Value Hash Join](execution-plans-operators-hash-join-general/query-plan-value-hash-join.html)
- [Node Left/Right Outer Hash Join](execution-plans-operators-hash-join-general/query-plan-node-left-right-outer-hash-join.html)

### Filter, order, and projection operators (12 个操作符)

文件夹: `filter-order-projection-operators/`

- [Empty Result](filter-order-projection-operators/query-plan-empty-result.html)
- [Produce Results](filter-order-projection-operators/query-plan-produce-results.html)
- [Filter](filter-order-projection-operators/query-plan-filter.html)
- [Empty Row](filter-order-projection-operators/query-plan-empty-row.html)
- [Cache Properties](filter-order-projection-operators/query-plan-cache-properties.html)
- [Projection](filter-order-projection-operators/query-plan-projection.html)
- [Project Endpoints](filter-order-projection-operators/query-plan-project-endpoints.html)
- [Distinct](filter-order-projection-operators/query-plan-distinct.html)
- [Ordered Distinct](filter-order-projection-operators/query-plan-ordered-distinct.html)
- [Procedure Call](filter-order-projection-operators/query-plan-procedure-call.html)
- [Unwind](filter-order-projection-operators/query-plan-unwind.html)
- [Partitioned Unwind](filter-order-projection-operators/query-plan-partitioned-unwind.html)

### Leaf operators (scans and seeks) (59 个操作符)

文件夹: `leaf-operators/`

- [All Nodes Scan](leaf-operators/query-plan-all-nodes-scan.html)
- [Partitioned All Nodes Scan](leaf-operators/query-plan-partitioned-all-nodes-scan.html)
- [Node By Label Scan](leaf-operators/query-plan-node-by-label-scan.html)
- [Partitioned Node By Label Scan](leaf-operators/query-plan-partitioned-node-by-label-scan.html)
- [Intersection Node By Labels Scan](leaf-operators/query-plan-intersection-node-by-labels-scan.html)
- [Partitioned Intersection Node By Labels Scan](leaf-operators/query-plan-partitioned-intersection-node-by-labels-scan.html)
- [Subtraction Node By Labels Scan](leaf-operators/query-plan-subtraction-node-by-labels-scan.html)
- [Partitioned Subtraction Node By Labels Scan](leaf-operators/query-plan-partitioned-subtraction-node-by-labels-scan.html)
- [Union Node By Labels Scan](leaf-operators/query-plan-union-node-by-labels-scan.html)
- [Partitioned Union Node By Labels Scan](leaf-operators/query-plan-partitioned-union-node-by-labels-scan.html)
- [Node Index Contains Scan](leaf-operators/query-plan-node-index-contains-scan.html)
- [Node Index Ends With Scan](leaf-operators/query-plan-node-index-ends-with-scan.html)
- [Node Index Scan](leaf-operators/query-plan-node-index-scan.html)
- [Partitioned Node Index Scan](leaf-operators/query-plan-partitioned-node-index-scan.html)
- [Node By ElementId Seek](leaf-operators/query-plan-node-by-elementid-seek.html)
- [Node By Id Seek](leaf-operators/query-plan-node-by-id-seek.html)
- [Node Index Seek](leaf-operators/query-plan-node-index-seek.html)
- [Partitioned Node Index Seek](leaf-operators/query-plan-partitioned-node-index-seek.html)
- [Node Unique Index Seek](leaf-operators/query-plan-node-unique-index-seek.html)
- [Multi Node Index Seek](leaf-operators/query-plan-multi-node-index-seek.html)
- [Asserting Multi Node Index Seek](leaf-operators/query-plan-asserting-multi-node-index-seek.html)
- [Node Index Seek By Range](leaf-operators/query-plan-node-index-seek-by-range.html)
- [Partitioned Node Index Seek By Range](leaf-operators/query-plan-partitioned-node-index-seek-by-range.html)
- [Node Unique Index Seek By Range](leaf-operators/query-plan-node-unique-index-seek-by-range.html)
- [Directed All Relationships Scan](leaf-operators/query-plan-directed-all-relationships-scan.html)
- [Partitioned Directed All Relationships Scan](leaf-operators/query-plan-partitioned-directed-all-relationships-scan.html)
- [Undirected All Relationships Scan](leaf-operators/query-plan-undirected-all-relationships-scan.html)
- [Partitioned Undirected All Relationships Scan](leaf-operators/query-plan-partitioned-undirected-all-relationships-scan.html)
- [Directed Relationship Type Scan](leaf-operators/query-plan-directed-relationship-type-scan.html)
- [Partitioned Directed Relationship Type Scan](leaf-operators/query-plan-partitioned-directed-relationship-types-scan.html)
- [Undirected Relationship Type Scan](leaf-operators/query-plan-undirected-relationship-type-scan.html)
- [Partitioned Undirected Relationship Type Scan](leaf-operators/query-plan-partitioned-undirected-relationship-type-scan.html)
- [Directed Union Relationship Types Scan](leaf-operators/query-plan-directed-union-relationship-types-scan.html)
- [Partitioned Directed Union Relationship Types Scan](leaf-operators/query-plan-partitioned-directed-union-relationship-types-scan.html)
- [Undirected Union Relationship Types Scan](leaf-operators/query-plan-undirected-union-relationship-types-scan.html)
- [Partitioned Undirected Union Relationship Types Scan](leaf-operators/query-plan-partitioned-undirected-union-relationship-types-scan.html)
- [Directed Relationship Index Scan](leaf-operators/query-plan-directed-relationship-index-scan.html)
- [Partitioned Directed Relationship Index Scan](leaf-operators/query-plan-partitioned-directed-relationship-index-scan.html)
- [Undirected Relationship Index Scan](leaf-operators/query-plan-undirected-relationship-index-scan.html)
- [Partitioned Undirected Relationship Index Scan](leaf-operators/query-plan-partitioned-undirected-relationship-index-scan.html)
- [Directed Relationship Index Contains Scan](leaf-operators/query-plan-directed-relationship-index-contains-scan.html)
- [Undirected Relationship Index Contains Scan](leaf-operators/query-plan-undirected-relationship-index-contains-scan.html)
- [Directed Relationship Index Ends With Scan](leaf-operators/query-plan-directed-relationship-index-ends-with-scan.html)
- [Undirected Relationship Index Ends With Scan](leaf-operators/query-plan-undirected-relationship-index-ends-with-scan.html)
- [Directed Relationship Index Seek](leaf-operators/query-plan-directed-relationship-index-seek.html)
- [Partitioned Directed Relationship Index Seek](leaf-operators/query-plan-partitioned-directed-relationship-index-seek.html)
- [Undirected Relationship Index Seek](leaf-operators/query-plan-undirected-relationship-index-seek.html)
- [Partitioned Undirected Relationship Index Seek](leaf-operators/query-plan-partitioned-undirected-relationship-index-seek.html)
- [Directed Relationship By Element Id Seek](leaf-operators/query-plan-directed-relationship-by-element-id-seek.html)
- [Directed Relationship By Id Seek](leaf-operators/query-plan-directed-relationship-by-id-seek.html)
- [Undirected Relationship By Element Id Seek](leaf-operators/query-plan-undirected-relationship-by-element-id-seek.html)
- [Undirected Relationship By Id Seek](leaf-operators/query-plan-undirected-relationship-by-id-seek.html)
- [Directed Relationship Index Seek By Range](leaf-operators/query-plan-directed-relationship-index-seek-by-range.html)
- [Partitioned Directed Relationship Index Seek By Range](leaf-operators/query-plan-partitioned-directed-relationship-index-seek-by-range.html)
- [Undirected Relationship Index Seek By Range](leaf-operators/query-plan-undirected-relationship-index-seek-by-range.html)
- [Partitioned Undirected Relationship Index Seek By Range](leaf-operators/query-plan-partitioned-undirected-relationship-index-seek-by-range.html)
- [Dynamic Label Node LookupIntroduced in 2025.08](leaf-operators/query-plan-dynamic-label-node-lookup.html)
- [Dynamic Directed Relationship Type LookupIntroduced in 2025.08](leaf-operators/query-plan-dynamic-directed-relationship-type-lookup.html)
- [Dynamic Undirected Relationship Type LookupIntroduced in 2025.08](leaf-operators/query-plan-dynamic-undirected-relationship-type-lookup.html)

### Nested loops and join operators (14 个操作符)

文件夹: `nested-loops-join-operators/`

- [Apply](nested-loops-join-operators/query-plan-apply.html)
- [Semi Apply](nested-loops-join-operators/query-plan-semi-apply.html)
- [Anti Semi Apply](nested-loops-join-operators/query-plan-anti-semi-apply.html)
- [Let Semi Apply](nested-loops-join-operators/query-plan-let-semi-apply.html)
- [Let Anti Semi Apply](nested-loops-join-operators/query-plan-let-anti-semi-apply.html)
- [Select Or Semi Apply](nested-loops-join-operators/query-plan-select-or-semi-apply.html)
- [Select Or Anti Semi Apply](nested-loops-join-operators/query-plan-select-or-anti-semi-apply.html)
- [Let Select Or Semi Apply](nested-loops-join-operators/query-plan-let-select-or-semi-apply.html)
- [Let Select Or Anti Semi Apply](nested-loops-join-operators/query-plan-let-select-or-anti-semi-apply.html)
- [Roll Up Apply](nested-loops-join-operators/query-plan-roll-up-apply.html)
- [TransactionApply](nested-loops-join-operators/query-plan-transaction-apply.html)
- [Argument](nested-loops-join-operators/query-plan-argument.html)
- [Argument Tracker](nested-loops-join-operators/query-plan-argument-tracker.html)
- [Cartesian Product](nested-loops-join-operators/query-plan-cartesian-product.html)

### Schema and system operators (13 个操作符)

文件夹: `schema-system-operators/`

- [Create Constraint](schema-system-operators/query-plan-create-constraint.html)
- [Do Nothing If Exists (constraint)](schema-system-operators/query-plan-do-nothing-if-exists-constraint.html)
- [Drop Constraint](schema-system-operators/query-plan-drop-constraint.html)
- [Show Constraints](schema-system-operators/query-plan-show-constraints.html)
- [Create Index](schema-system-operators/query-plan-create-index.html)
- [Do Nothing If Exists (index)](schema-system-operators/query-plan-do-nothing-if-exists-index.html)
- [Drop Index](schema-system-operators/query-plan-drop-index.html)
- [Show Indexes](schema-system-operators/query-plan-show-indexes.html)
- [Show Functions](schema-system-operators/query-plan-show-functions.html)
- [Show Procedures](schema-system-operators/query-plan-show-procedures.html)
- [Show Settings](schema-system-operators/query-plan-show-settings.html)
- [Show Transactions](schema-system-operators/query-plan-show-transactions.html)
- [Terminate Transactions](schema-system-operators/query-plan-terminate-transactions.html)

### Sort and limit operators (7 个操作符)

文件夹: `sort-limit-operators/`

- [Sort](sort-limit-operators/query-plan-sort.html)
- [Partial Sort](sort-limit-operators/query-plan-partial-sort.html)
- [Top](sort-limit-operators/query-plan-top.html)
- [Partial Top](sort-limit-operators/query-plan-partial-top.html)
- [Limit](sort-limit-operators/query-plan-limit.html)
- [Exhaustive Limit](sort-limit-operators/query-plan-exhaustive-limit.html)
- [Skip](sort-limit-operators/query-plan-skip.html)

### Traversal operators (18 个操作符)

文件夹: `traversal-operators/`

- [Anti](traversal-operators/query-plan-anti.html)
- [Optional](traversal-operators/query-plan-optional.html)
- [Expand All](traversal-operators/query-plan-expand-all.html)
- [Expand Into](traversal-operators/query-plan-expand-into.html)
- [Optional Expand All](traversal-operators/query-plan-optional-expand-all.html)
- [Optional Expand Into](traversal-operators/query-plan-optional-expand-into.html)
- [VarLength Expand All](traversal-operators/query-plan-varlength-expand-all.html)
- [VarLength Expand Into](traversal-operators/query-plan-varlength-expand-into.html)
- [VarLength Expand Pruning](traversal-operators/query-plan-varlength-expand-pruning.html)
- [Breadth First VarLength Expand Pruning](traversal-operators/query-plan-breadth-first-varlength-expand-pruning-bfs-all.html)
- [Repeat](traversal-operators/query-plan-repeat.html)
- [Nullify Metadata](traversal-operators/query-plan-nullify-metadata.html)
- [Shortest path](traversal-operators/query-plan-shortest-path.html)
- [StatefulShortestPath(Into)](traversal-operators/query-plan-stateful-shortest-path-into.html)
- [StatefulShortestPath(All)](traversal-operators/query-plan-stateful-shortest-path-all.html)
- [Triadic Selection](traversal-operators/query-plan-triadic-selection.html)
- [Triadic Build](traversal-operators/query-plan-triadic-build.html)
- [Triadic Filter](traversal-operators/query-plan-triadic-filter.html)

### Union operators (1 个操作符)

文件夹: `union-operators/`

- [Union](union-operators/query-plan-union.html)


**总计: 152 个操作符**
