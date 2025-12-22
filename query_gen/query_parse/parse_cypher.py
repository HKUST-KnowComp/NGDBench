from antlr4 import InputStream, CommonTokenStream
from antlr4_cypher import CypherLexer, CypherParser
import json

from visitor import CypherASTVisitor   # 你自己的文件

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
    MATCH (d:Disease {name:\"hepatocellular carcinoma\"})-[:DISEASE_PROTEIN]-(p:Protein)<-[:DRUG_PROTEIN]-(dr:Drug) WHERE NOT (dr)-[:CONTRAINDICATION]-(:Disease) RETURN DISTINCT dr.name
    """

    ast = parse_cypher_to_ast(query)

    json_str = json.dumps(ast, indent=2)
    print(json_str)
    
    with open('ast_output.json', 'w', encoding='utf-8') as f:
        f.write(json_str)


if __name__ == '__main__':
    main()