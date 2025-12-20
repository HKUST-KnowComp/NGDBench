from antlr4_cypher import CypherParserVisitor

class CypherASTVisitor(CypherParserVisitor):
    def __init__(self):
        self.ast = {
            "type": "query",
            "clauses": [],
            "return": [],
            "limit": None
        }
    def visitMatchSt(self, ctx):
        return self._handle_match(ctx, optional=False)

    def visitOptionalMatchSt(self, ctx):
        return self._handle_match(ctx, optional=True)

    def _handle_match(self, ctx, optional):
        clause = {
            "type": "match",
            "optional": optional,
            "patterns": []
        }

        pattern = ctx.patternWhere().pattern()

        for part in pattern.patternPart():
            clause["patterns"].append(self.visit(part))

        self.ast["clauses"].append(clause)
        return None

    def visitPatternPart(self, ctx):
        # p = (a)-[:r]->(b)
        if ctx.symbol():
            path_var = ctx.symbol().getText()
            path = {
                "type": "path",
                "variable": path_var,
                "elements": []
            }

            elem = ctx.patternElem()
            path["elements"].append(self.visit(elem.nodePattern()))

            for chain in elem.patternElemChain():
                path["elements"].append(self.visit(chain.relationshipPattern()))
                path["elements"].append(self.visit(chain.nodePattern()))

            return path

        # 普通 pattern
        return self.visit(ctx.patternElem())

    def visitNodePattern(self, ctx):
        node = {
            "type": "node",
            "variable": None,
            "labels": [],
            "properties": {}
        }

        if ctx.symbol():
            node["variable"] = ctx.symbol().getText()

        if ctx.nodeLabels():
            node["labels"] = [l.getText() for l in ctx.nodeLabels().nodeLabel()]

        if ctx.properties():
            for pair in ctx.properties().mapLit().mapPair():
                key = pair.name().getText()
                value = pair.expression().getText().strip("'")
                node["properties"][key] = value

        return node
    def visitRelationshipPattern(self, ctx):
        rel = {
            "type": "relationship",
            "labels": [],
            "direction": "->"
        }

        detail = ctx.relationDetail()
        if detail and detail.relationshipTypes():
            rel["labels"] = [n.getText() for n in detail.relationshipTypes().name()]

        return rel
    def visitReturnSt(self, ctx):
        projection_body = ctx.projectionBody()
        items = projection_body.projectionItems().projectionItem()
        for item in items:
            self.ast["return"].append(item.getText())

        if projection_body.limitSt():
            self.ast["limit"] = int(projection_body.limitSt().expression().getText())

        return None
