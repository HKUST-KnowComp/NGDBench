from operator_no_neural import *
def DM01_Upsert_Template(params):
    # 这是一个“智能接入”模版
    # 它可以处理：新数据 -> 插入；旧数据 -> 更新
    
    result = Upsert_Resolve(
        target_type = params.T_Target,       # 目标类型，如 "User"
        match_keys = {params.Key: params.Val}, # 唯一键，如 {"user_id": "1001"}
        properties = params.Props,           # 要写入的属性，如 {"name": "Alice", "age": 30}
        
        # 冲突解决策略：
        # "OVERWRITE": 覆盖旧属性 (激进更新)
        # "IGNORE": 保持旧属性不动 (保护历史数据)
        # "MERGE": 仅更新非空字段
        strategy = params.Resolution_Strategy 
    )
    return result


def DM02_Link_Template(params):
    # 1. 找到（或创建）源节点 (比如: 新员工)
    # 这里复用了 DM01 的逻辑，确保员工存在
    src_node = Upsert_Resolve(
        target_type = params.T_Src, 
        match_keys = params.Src_Keys,
        properties = params.Src_Props
    )
    
    # 2. 找到目标节点 (比如: 研发部)
    # 这里通常是查询，因为部门应该已经存在
    dst_node = Scan(
        type = params.T_Dst, 
        filter = params.Dst_Filter
    ).first()
    
    # 3. 插入边 (Insert Edge)
    # 只有当两个端点都存在时，才创建关系
    if src_node and dst_node:
        edge = Insert(
            element_type = "Edge",
            source = src_node,
            target = dst_node,
            edge_type = params.T_Edge, # 如 "WORKS_IN"
            properties = params.Edge_Props
        )
        return edge
    else:
        return Error("Target node not found")

def DM03_Batch_Update_Template(params):
    # 1. 圈选出问题数据
    targets = Filter(
        Scan(type=params.T_Target), 
        {params.Filter_Key: params.Filter_Val} # 如: status == null
    )
    
    # 2. 批量更新
    # 这是一个遍历更新的过程
    updated_count = 0
    for node in targets:
        Update(
            target = node,
            set_properties = params.Update_Props # 如: {"status": "unknown"}
        )
        updated_count += 1
        
    return {"status": "success", "affected_rows": updated_count}


def DM05_Cascading_Delete_Template(params):
    # 1. 找到主节点 (用户)
    root_node = Scan(type=params.T_Root, id=params.Root_Id).first()
    
    if not root_node: return
    
    # 2. 找到附属节点/边 (比如: 他的所有登录记录，或者他的所有边)
    # 这里用到了 Project 算子
    related_items = Project(
        input = [root_node],
        edge = params.T_Edge, # 如 "HAS_LOGIN_RECORD"
        direction = "OUT"
    )
    
    # 3. 先删附属 (防止悬挂引用)
    for item in related_items:
        Delete(target = item)
        
    # 4. 最后删主节点
    Delete(target = root_node)
    
    return {"msg": "Account and related data pruned"}