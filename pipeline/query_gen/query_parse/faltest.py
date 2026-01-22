import os
from falkordb import FalkorDB

# 从环境变量获取连接参数，如果没有设置则使用默认值
# Docker 容器设置的密码是 "falkordb"
host = os.getenv('FALKORDB_HOST', 'localhost')
port = int(os.getenv('FALKORDB_PORT', '6379'))
password = os.getenv('FALKORDB_PASSWORD', 'falkordb')

# 创建客户端，使用密码连接
client = FalkorDB(host=host, port=port, password=password)
graph = client.select_graph('imdb')
query = '''\
MATCH (other:Person)
WHERE not other.age > 50
RETURN other.name
'''
# result = graph.profile(query)
# for line in result:
#     print(line)
result = graph.explain(query)
print(result)