from antlr4_cypher import CypherParserVisitor

class CypherASTVisitor(CypherParserVisitor):
    def __init__(self):
        self.ast = {
            "type": "query",
            "clauses": [],
            "return": {
                "distinct": False,
                "items": [],
                "orderBy": [],
                "limit": None
            }
        }
    def visitMatchSt(self, ctx):
        return self._handle_match(ctx, optional=False)

    def visitOptionalMatchSt(self, ctx):
        return self._handle_match(ctx, optional=True)

    def _handle_match(self, ctx, optional):
        clause = {
            "type": "match",
            "optional": optional,
            "patterns": [],
            "where": None
        }

        pw = ctx.patternWhere()

        # pattern
        pattern = pw.pattern()
        for part in pattern.patternPart():
            clause["patterns"].append(self.visit(part))

        if pw.where():
            clause["where"] = {
                "type": "where",
                "expression": pw.where().expression().getText()
            }

        self.ast["clauses"].append(clause)
        return None

    def visitPatternPart(self, ctx):
        # 如果有 path variable: p = (...)
        path = self.visit(ctx.patternElem())

        if ctx.symbol():
            path["variable"] = ctx.symbol().getText()

        return path

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
            node["labels"] = [n.getText() for n in ctx.nodeLabels().name()]

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
        body = ctx.projectionBody()

        if body.DISTINCT():
            self.ast["return"]["distinct"] = True

        for item in body.projectionItems().projectionItem():
            expr = item.expression().getText()
            alias = None

            if item.AS():
                alias = item.symbol().getText()

            self.ast["return"]["items"].append({
                "expr": expr,
                "alias": alias
            })

        if body.orderSt():
            for item in body.orderSt().orderItem():
                direction = "ASC"

                if item.DESC():
                    direction = "DESC"
                elif item.ASC():
                    direction = "ASC"

                self.ast["return"]["orderBy"].append({
                    "expr": item.expression().getText(),
                    "direction": direction
                })


        if body.limitSt():
            self.ast["return"]["limit"] = int(
                body.limitSt().expression().getText()
            )

        return None

    def visitPatternElem(self, ctx):
        path = {
            "type": "path",
            "variable": None,
            "elements": []
        }

        # 起始 node
        path["elements"].append(self.visit(ctx.nodePattern()))

        # chain: (rel, node)*
        for chain in ctx.patternElemChain():
            path["elements"].append(self.visit(chain.relationshipPattern()))
            path["elements"].append(self.visit(chain.nodePattern()))

        return path
