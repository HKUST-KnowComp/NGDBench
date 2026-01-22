from generator.manage_generator import ManageGenerator

# 创建生成器
generator = ManageGenerator(
    uri="bolt://localhost:7694",
    user="neo4j",
    password="fei123456",  # 根据实际数据库密码修改
    template_path="query_template/template_managemet.json"
)

# 初始化（连接数据库并分析schema）
generator.initialize()

# 生成查询样本
results = generator.generate_samples(
    target_count=100,
    operations=["CREATE", "DELETE", "SET", "MERGE"],  # 可选：指定操作类型
    success_per_template=5
)

# 导出结果
generator.export_results("output.json")

# 关闭连接
generator.close()