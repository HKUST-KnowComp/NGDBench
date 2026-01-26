import json
import argparse
import logging
from generator.query_generator import QueryGenerator

logger = logging.getLogger(__name__)

# 默认连接配置（可被命令行参数覆盖）
DEFAULT_NEO4J_URI = "bolt://localhost:7690"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "fei123456"
DEFAULT_DATASET = "mcp"


def test_query_generator(uri: str, user: str, password: str, dataset: str):
    """测试查询生成器"""
    
    
    # 测试生成器
    generator = QueryGenerator(
        uri=uri,
        user=user,
        password=password,
        template_path="query_template/template_mcp1.json",
        dataset=dataset,
    )
    
    try:
        generator.connect()
        generator.initialize()
        
        print(f"\n=== Schema 信息 ===")
        print(f"Labels: {list(generator.schema.labels.keys())}")
        print(f"Relationships: {list(generator.schema.relationships.keys())}")
        print(f"Total nodes: {generator.schema.total_nodes}")
        print(f"Total edges: {generator.schema.total_edges}")
        print(f"Target sample count: {generator.get_target_sample_count()}")
        
        print(f"\n=== Label 属性详情 ===")
        for label, info in generator.schema.labels.items():
            print(f"\n{label}:")
            for prop_name, prop_info in info.properties.items():
                print(f"  - {prop_name}: {prop_info.prop_type.value}, samples: {prop_info.sample_values[:3]}")
        
        print(f"\n=== 开始生成查询 ===")
        output_file = f"query_results_{dataset}.json"
        # 使用实时输出，一边生成一边写入文件
        results = generator.generate_samples(target_count=10000, realtime_output_path=output_file)
        
        print(f"\n=== 生成结果 ===")
        print(f"共生成 {len(results)} 个查询")
        # 显示前几个结果的摘要
        for i, r in enumerate(results[:5]):  # 只显示前5个
            print(f"\n[{i+1}] Template: {r.template_id}")
            print(f"    Query: {r.query[:100]}...")  # 只显示前100个字符
            print(f"    Answer count: {len(r.answer)}")
        if len(results) > 5:
            print(f"\n... 还有 {len(results) - 5} 个查询结果")
        
        print(f"\n所有结果已实时写入文件: {output_file}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        generator.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test QueryGenerator with Neo4j.")
    parser.add_argument(
        "--uri",
        type=str,
        default=DEFAULT_NEO4J_URI,
        help=f"Neo4j URI, 默认: {DEFAULT_NEO4J_URI}",
    )
    parser.add_argument(
        "--user",
        type=str,
        default=DEFAULT_NEO4J_USER,
        help=f"Neo4j 用户名, 默认: {DEFAULT_NEO4J_USER}",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=DEFAULT_NEO4J_PASSWORD, 
        help="Neo4j 密码（命令行传入会出现在历史记录中，注意安全）",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"数据集, 默认: {DEFAULT_DATASET}",
    )
    args = parser.parse_args()
    test_query_generator(
        uri=args.uri,
        user=args.user,
        password=args.password,
        dataset=args.dataset,
    )