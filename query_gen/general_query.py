from operator_no_neural import *

def Direct_Relation_Template(params):
    # 1. 锁定起始节点 A
    start_nodes = Filter(Scan(type=params.T_start), {params.P_start: params.V_start})
    
    # 2. 遍历边 R，支持对边属性的过滤 (r.attr)
    # params.E_type: 边类型
    # params.E_prop_k, params.E_prop_v: 边的属性过滤键值对
    edge_filter = None
    if params.E_prop_k and params.E_prop_v:
        edge_filter = {params.E_prop_k: params.E_prop_v}
    
    result = Project(
        input = start_nodes,
        edge = params.E_type,
        edge_filter = edge_filter
    )
    return result
# Config: 查找名为"TP53"的基因(A)参与的(R)且置信度大于0.9的通路(B)
# {
#   "T_start": "Gene", "P_start": "name", "V_start": "TP53",
#   "E_type": "PARTICIPATES_IN", "E_prop_k": "confidence", "E_prop_v": ">0.9",
#   "T_target": "Pathway"
# }

def Chain_Hop_Template(params):
    # 1. Start A
    n0 = Filter(Scan(type=params.T1), {params.P1: params.V1})
    
    # 2. Hop 1: A -> B
    n1 = Project(n0, edge=params.E1)
    
    # 3. Hop 2: B -> C
    n2 = Project(n1, edge=params.E2)
    
    # 4. Hop 3: C -> D (可选，如果 params.E3 为空则返回 n2)
    if params.E3:
        n3 = Project(n2, edge=params.E3)
        return n3
    return n2
#  Config: 查药物(A) -> 靶点(B) -> 关联疾病(C) -> 这种疾病的症状(D)
# {
#   "T1": "Drug", "P1": "name", "V1": "Aspirin",
#   "E1": "TARGETS", "T2": "Protein",
#   "E2": "ASSOCIATED_WITH", "T3": "Disease",
#   "E3": "HAS_SYMPTOM", "T4": "Symptom"
# }

def Aggregation_Rank_Template(params):
    # 1. Start A
    start = Filter(Scan(type=params.T1), {params.P1: params.V1})
    
    # 2. Path: A -> ... -> Target
    # 假设是两跳 A -> B -> C
    # 如果指定了方向，使用该方向；否则默认使用 OUT
    direction1 = getattr(params, 'direction1', 'OUT')
    path_mid = Project(start, edge=params.E1, direction=direction1)
    # 使用 TraverseWithPath 获取最后一跳的路径信息（B -> C）
    direction2 = getattr(params, 'direction2', 'OUT')
    paths_b_to_c = TraverseWithPath(path_mid, edge=params.E2, target_type=params.T3, direction=direction2)
    
    # 3. 聚合统计
    # 按照 T3 (C) 分组，统计 T2 (B) 的数量，并降序排列
    result = Aggregate(
        input = GroupBy(paths_b_to_c, key="target_node"), # Group by C
        func = "COUNT",
        target = "source_node", # Count B (incoming path)
        order = "DESC"
    )
    return result
# Config: 给定疾病(A)，通过关联基因(B)，找到相关的药物(C)，按涉及基因数量排序
# {
#   "T1": "Disease", "P1": "name", "V1": "Diabetes",
#   "E1": "ASSOCIATED_GENE", "T2": "Gene",
#   "E2": "TARGETED_BY", "T3": "Drug"
# }

def Shared_Neighbor_Template(params):
    # 1. Start A
    node_a = Filter(Scan(type=params.T_A), {params.P_A: params.V_A})
    
    # 2. A -> B (中间共享点)
    nodes_b = Project(node_a, edge=params.E1)
    
    # 3. B -> C (反向或继续遍历找到 C)
    # 注意：通常需要排除 C == A
    # 使用 TraverseWithPath 获取路径信息（B -> C）
    paths_b_to_c = TraverseWithPath(nodes_b, edge=params.E2, target_type=params.T_C)
    # 过滤掉目标节点是 A 的路径
    filtered_paths = [p for p in paths_b_to_c if p["target_node"] not in node_a]
    
    # 4. 统计相似度 (共享 B 的数量)
    result = Aggregate(
        input = GroupBy(filtered_paths, key="target_node"),
        func = "COUNT",
        target = "source_node", # Count B
        order = "DESC"
    )
    return result
# Config: 查找与用户A(A)购买过相同商品(B)的其他用户(C)，按重合度排序
# {
#   "T_A": "User", "P_A": "id", "V_A": "u_123",
#   "E1": "PURCHASED", "T_B": "Product",
#   "E2": "PURCHASED_BY", "T_C": "User" // 边方向可能需要反转处理
# }

def Conditional_Logic_Template(params):
    # 1. Start A
    start = Filter(Scan(type=params.T_A), {params.P_A: params.V_A})
    
    # 2. Branch 1: Positive Condition (必须连接 C)
    path_c = Project(start, edge=params.E_AC)
    valid_a_c = GetSourceNodes(path_c) # 获取有 C 的 A
    
    # 3. Branch 2: Negative Condition (不能连接 D)
    # 或者 Count Condition (连接 B 的数量 > N)
    path_b = Project(start, edge=params.E_AB)
    
    # 聚合 B，筛选数量 >= N 的 A
    valid_a_b = AggregateFilter(
        input = GroupBy(path_b, key="source_node"),
        func = "COUNT",
        condition = f"> {params.min_count}"
    )
    
    # 4. Intersection (取交集)
    final_a = Intersect(valid_a_c, valid_a_b)
    return final_a
# Config: 找到某作者(A)，他至少写了5篇论文(B)，且与某机构(C)有合作
# {
#   "T_A": "Author", "P_A": "field", "V_A": "AI",
#   "E_AB": "WROTE", "T_B": "Paper", "min_count": 5,
#   "E_AC": "COLLABORATES_WITH", "T_C": "Institution"
# }

def Shortest_Path_Template(params):
    # 给定起点 A 和 终点 B，找出最短路径
    start = Filter(Scan(type=params.T_A), {params.P_A: params.V_A})
    end = Filter(Scan(type=params.T_B), {params.P_B: params.V_B})
    
    paths = FindShortestPath(
        source = start,
        target = end,
        max_depth = params.max_hops,
        edge_types = params.allowed_edges # 可选：只允许走特定类型的边
    )
    return paths

def Critical_Node_Template(params):
    # 1. 找出 A 到 B 的所有路径
    paths = FindAllPaths(
        source = Scan(type=params.T_A),
        target = Scan(type=params.T_B),
        max_depth = params.depth
    )
    
    # 2. 统计路径上出现的所有中间节点
    nodes = ExtractNodes(paths, exclude_start_end=True)
    
    # 3. 聚合计数
    result = Aggregate(
        input = nodes,
        func = "COUNT",
        order = "DESC",
        limit = 5
    )
    return result

def Tree_Expansion_Template(params):
    # 1. Root A -> Node B
    root = Filter(Scan(type=params.T_A), {params.P_A: params.V_A})
    level_1 = Project(root, edge=params.E1)
    
    # 2. Node B -> Leaf C
    level_2 = Project(level_1, edge=params.E2)
    
    # 3. 收集结果：返回结构化的树或列表
    # 输出格式通常是 { "B": node, "children_C": [list of C] }
    result = CollectHierarchy(parent=level_1, child=level_2)
    return result

# Config: 查公司(A) -> 下属部门(B) -> 部门员工(C)
# {
#   "T_A": "Company", "P_A": "name", "V_A": "Google",
#   "E1": "HAS_DEPT", "T_B": "Department",
#   "E2": "EMPLOYS", "T_C": "Employee"
# }

def Branching_Info_Template(params):
    # 1. Start A -> Center B
    start = Filter(Scan(type=params.T_A), {params.P_A: params.V_A})
    center = Project(start, edge=params.E1)
    
    # 2. Branch 1: B -> C
    branch_c = Project(center, edge=params.E2)
    
    # 3. Branch 2: B -> D
    branch_d = Project(center, edge=params.E3)
    
    # 4. 组合结果
    # 返回 { "center": B, "related_C": [...], "related_D": [...] }
    result = MergeBranches(center, branch_c, branch_d)
    return result
# Config: 查电影(A) -> 导演(B)，然后同时获取该导演的其他电影(C)和获得的奖项(D)
# {
#   "T_A": "Movie", "P_A": "name", "V_A": "Inception",
#   "E1": "DIRECTED_BY", "T_B": "Person",
#   "E2": "DIRECTED", "T_C": "Movie",
#   "E3": "WON_AWARD", "T_D": "Award"
# }

def Differential_Pair_Template(params):
    # 1. 锁定公共终点 B (如果 B 已知)
    # 或者从 C 端开始遍历。这里假设从 B 倒推比较高效。
    common_b = Filter(Scan(type=params.T_B), {params.P_B: params.V_B})
    
    # 2. B <- A (反向遍历)
    nodes_a = Project(common_b, edge=params.E_AB, direction="IN")
    
    # 3. A <- C (反向遍历)
    nodes_c = Project(nodes_a, edge=params.E_CA, direction="IN")
    
    # 4. 组装路径并寻找差异
    # 目标：找到 (Path1, Path2) 使得 Path1.B == Path2.B 但 Path1.C != Path2.C
    # 实际执行通常是 GroupBy B, 然后在组内做笛卡尔积筛选
    result = FindDisjointPairs(
        input = nodes_c, 
        anchor = "B", # 锚点
        diff_target = "C" # 需要不同的点
    )
    return result
# Config: 找到两个不同的供应商(C1, C2)，他们通过不同的中间商(A1, A2)，最终向同一个超市(B)供货
# {
#   "T_B": "Supermarket", "P_B": "name", "V_B": "Walmart",
#   "E_AB": "SUPPLIED_BY", "T_A": "Distributor",
#   "E_CA": "SUPPLIED_BY", "T_C": "Supplier"
# }

def Negative_Relation_Template(params):
    # 1. 基础集合：找到满足正向条件的 A (例如：购买了 iPhone 的用户)
    base_nodes = Filter(Scan(type=params.T_A), {params.P_A: params.V_A})
    
    # 如果有正向边限制 (例如：A -> B)，先进行一次遍历
    if params.E_Positive:
        base_nodes = Project(base_nodes, edge=params.E_Positive)
        # 注意：这里我们要保留的是源节点 A，用于后续判断
        base_nodes = GetSourceNodes(base_nodes) 

    # 2. 负向探测：找出这些 A 中，哪些具备"禁止"的边 (A -> C)
    # (例如：找出其中购买了 手机壳 的用户)
    excluded_nodes_path = Project(
        input = base_nodes, 
        edge = params.E_Negative
    )
    nodes_to_exclude = GetSourceNodes(excluded_nodes_path)
    
    # 3. 差集运算 (NOT)：从基础集合中剔除掉具备负向边的节点
    # Result = Base - Excluded
    result = Minus(base_nodes, nodes_to_exclude)
    
    return result
# Config: 寻找购买了“主机”(B) 但没有购买“保修服务”(C) 的客户(A) -> 销售线索
# {
#   "T_A": "Customer", "P_A": "status", "V_A": "active",
#   "E_Positive": "BOUGHT", "T_Pos": "Console",
#   "E_Negative": "BOUGHT", "T_Neg": "Warranty"
# }

def Value_Join_Template(params):
    # 1. 获取左侧集合 (Set A)
    # 例如：最近登录的所有账号
    nodes_a = Filter(Scan(type=params.T_A), {params.P_Filter_A: params.V_Filter_A})
    
    # 2. 获取右侧集合 (Set B)
    # 例如：黑名单中的 IP 地址记录，或者另一批高风险账号
    # (如果是自连接 Self-Join，这里 nodes_b 可以等于 nodes_a)
    nodes_b = Filter(Scan(type=params.T_B), {params.P_Filter_B: params.V_Filter_B})
    
    # 3. 执行 Join 操作
    # 这里的核心不是沿着边走，而是比较属性值：A.key == B.key
    # 这种操作在图数据库中通常比遍历要慢，但在数据清洗和关联挖掘中必不可少
    joined_results = Join(
        left_set = nodes_a,
        right_set = nodes_b,
        join_key_left = params.Key_A,  # 例如: A.last_login_ip
        join_key_right = params.Key_B, # 例如: B.ip_address
        join_type = "INNER" # 只保留匹配上的
    )
    
    return joined_results