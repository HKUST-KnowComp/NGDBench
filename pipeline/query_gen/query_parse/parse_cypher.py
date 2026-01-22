from antlr4 import InputStream, CommonTokenStream
from antlr4_cypher import CypherLexer, CypherParser
import json
import os
import sys

# 支持直接运行和作为模块导入两种方式
try:
    from .visitor import CypherASTVisitor  # 作为包导入
except ImportError:
    # 直接运行时
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    if _current_dir not in sys.path:
        sys.path.insert(0, _current_dir)
    from visitor import CypherASTVisitor

def parse_cypher_to_ast(query: str):
    # 1. 词法 / 语法分析
    input_stream = InputStream(query)
    lexer = CypherLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = CypherParser(stream)

    # 2. 生成 parse tree
    tree = parser.script()

    # 3. 用 Visitor 生成 AST
    visitor = CypherASTVisitor()
    visitor.visit(tree)

    return visitor.ast
    
def main():
    
    query = """
    MATCH (d:Drug {name: \"{drug_name}\"})-[:drug_protein {label: \"target\"}]->(p:Protein)-[:anatomy_protein_present {label: \"expression present\"}]->(a:Anatomy) RETURN DISTINCT p.name AS TargetProtein, a.name AS AnatomicalRegion
        """

    ast = parse_cypher_to_ast(query)

    json_str = json.dumps(ast, indent=2)
    print(json_str)
    
    with open('ast_output.json', 'w', encoding='utf-8') as f:
        f.write(json_str)


if __name__ == '__main__':
    main()