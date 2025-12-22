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
    MATCH (d1:Disease {name: \"Chediak-Higashi syndrome\"})-[:DISEASE_PHENOTYPE_POSITIVE]-(pheno:Phenotype)-[:DISEASE_PHENOTYPE_POSITIVE]-(d2:Disease) MATCH (d2)-[:EXPOSURE_DISEASE]-(exp:Exposure) RETURN exp.name AS exposure, COUNT(DISTINCT d2) AS associated_diseases ORDER BY associated_diseases DESC
        """

    ast = parse_cypher_to_ast(query)

    json_str = json.dumps(ast, indent=2)
    print(json_str)
    
    with open('ast_output.json', 'w', encoding='utf-8') as f:
        f.write(json_str)


if __name__ == '__main__':
    main()