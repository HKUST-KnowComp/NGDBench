import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
from collections import defaultdict
import re

class Neo4jOperatorCrawler:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.categories = defaultdict(list)  # 存储 {大类: [小类列表]}
        
    def fetch_page(self, url):
        """获取页面内容"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            response.encoding = 'utf-8'
            return response.text
        except Exception as e:
            print(f"获取页面失败 {url}: {e}")
            return None
    
    def parse_page(self):
        """解析页面，提取大类和小类"""
        html = self.fetch_page(self.base_url)
        if not html:
            return
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # 找到主内容区域
        content = soup.find('article') or soup.find('main') or soup.find('div', class_='content')
        
        if not content:
            print("未找到主内容区域")
            return
        
        current_category = None
        current_category_id = None
        
        # 遍历所有元素，找到大类标题和小类标题
        for element in content.find_all(['h2', 'h3', 'h4']):
            elem_id = element.get('id', '')
            
            # 检查是否是大类标题（包含id且不以query-plan-开头）
            if elem_id and not elem_id.startswith('query-plan-'):
                current_category = element.get_text(strip=True)
                current_category_id = elem_id
                if current_category_id not in self.categories:
                    self.categories[current_category_id] = {
                        'name': current_category,
                        'operators': []
                    }
                print(f"\n找到大类: {current_category} (id: {current_category_id})")
            
            # 检查是否是小类标题（id以query-plan-开头）
            elif elem_id and elem_id.startswith('query-plan-') and current_category_id:
                operator_id = elem_id
                operator_name = element.get_text(strip=True)
                
                if operator_name:
                    # 检查是否已经存在，避免重复添加
                    existing_ids = [op['id'] for op in self.categories[current_category_id]['operators']]
                    if operator_id not in existing_ids:
                        self.categories[current_category_id]['operators'].append({
                            'id': operator_id,
                            'name': operator_name,
                            'anchor': f'#{operator_id}'
                        })
                        print(f"  - {operator_name} ({operator_id})")
    
    def extract_operator_content(self, soup, operator_id):
        """提取特定操作符的内容"""
        # 找到操作符的标题元素
        operator_header = soup.find(id=operator_id)
        if not operator_header:
            return None
        
        # 收集内容直到下一个相同级别的标题
        content_parts = []
        current = operator_header
        header_level = operator_header.name  # h2, h3, h4等
        
        # 添加标题本身
        content_parts.append(str(operator_header))
        
        # 遍历后续兄弟元素
        for sibling in operator_header.find_next_siblings():
            # 如果遇到同级或更高级别的标题，停止
            if sibling.name and sibling.name.startswith('h'):
                sibling_level = int(sibling.name[1])
                current_level = int(header_level[1])
                if sibling_level <= current_level:
                    break
            
            content_parts.append(str(sibling))
        
        return '\n'.join(content_parts)
    
    def save_operator_content(self, category_id, category_name, operator, soup, base_dir='neo4j_operators'):
        """保存操作符内容到文件"""
        # 创建分类文件夹
        category_dir = os.path.join(base_dir, self.sanitize_filename(category_id))
        os.makedirs(category_dir, exist_ok=True)
        
        # 提取操作符内容
        content = self.extract_operator_content(soup, operator['id'])
        
        if content:
            # 保存为HTML文件
            filename = os.path.join(category_dir, f"{self.sanitize_filename(operator['id'])}.html")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"<!DOCTYPE html>\n<html>\n<head>\n")
                f.write(f"<meta charset='utf-8'>\n")
                f.write(f"<title>{operator['name']}</title>\n")
                f.write(f"<style>body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 20px auto; padding: 0 20px; }}</style>\n")
                f.write(f"</head>\n<body>\n")
                f.write(f"<p><strong>类别:</strong> {category_name}</p>\n")
                f.write(f"<p><strong>来源:</strong> <a href='{self.base_url}{operator['anchor']}'>{self.base_url}{operator['anchor']}</a></p>\n")
                f.write(f"<hr>\n")
                f.write(content)
                f.write(f"\n</body>\n</html>")
            
            print(f"  已保存: {operator['name']}")
        else:
            print(f"  未找到内容: {operator['name']}")
    
    def sanitize_filename(self, name):
        """清理文件名，移除非法字符"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, '_')
        return name.strip()
    
    def crawl_all(self):
        """执行完整爬取流程"""
        print("开始爬取Neo4j操作符文档...")
        print(f"目标URL: {self.base_url}\n")
        
        # 解析页面结构
        self.parse_page()
        
        if not self.categories:
            print("未找到任何操作符分类，请检查页面结构")
            return
        
        # 获取完整页面用于提取内容
        html = self.fetch_page(self.base_url)
        soup = BeautifulSoup(html, 'html.parser')
        
        # 统计信息
        print("\n" + "="*60)
        print("操作符统计:")
        print("="*60)
        total = 0
        for category_id, category_data in sorted(self.categories.items()):
            count = len(category_data['operators'])
            total += count
            print(f"{category_data['name']} ({category_id}): {count} 个操作符")
        print(f"\n总计: {total} 个操作符")
        print("="*60 + "\n")
        
        # 保存所有操作符内容
        for category_id, category_data in self.categories.items():
            print(f"\n正在保存分类 [{category_data['name']}]:")
            for operator in category_data['operators']:
                self.save_operator_content(
                    category_id,
                    category_data['name'],
                    operator,
                    soup
                )
                time.sleep(0.1)  # 轻微延时
        
        # 保存统计信息
        self.save_statistics()
        
        print("\n爬取完成!")
    
    def save_statistics(self, base_dir='neo4j_operators'):
        """保存统计信息到文件"""
        stats_file = os.path.join(base_dir, 'statistics.txt')
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("Neo4j Cypher 操作符统计\n")
            f.write("="*60 + "\n\n")
            
            total = 0
            for category_id, category_data in sorted(self.categories.items()):
                count = len(category_data['operators'])
                total += count
                f.write(f"{category_data['name']} ({category_id}): {count} 个操作符\n")
                f.write("-" * 40 + "\n")
                for op in category_data['operators']:
                    f.write(f"  - {op['name']} (id: {op['id']})\n")
                f.write("\n")
            
            f.write("="*60 + "\n")
            f.write(f"总计: {total} 个操作符\n")
        
        print(f"\n统计信息已保存到: {stats_file}")
        
        # 同时保存为README
        readme_file = os.path.join(base_dir, 'README.md')
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write("# Neo4j Cypher 操作符文档\n\n")
            f.write(f"来源: {self.base_url}\n\n")
            f.write("## 目录结构\n\n")
            
            for category_id, category_data in sorted(self.categories.items()):
                count = len(category_data['operators'])
                f.write(f"### {category_data['name']} ({count} 个操作符)\n\n")
                f.write(f"文件夹: `{category_id}/`\n\n")
                for op in category_data['operators']:
                    f.write(f"- [{op['name']}]({category_id}/{op['id']}.html)\n")
                f.write("\n")
            
            f.write(f"\n**总计: {sum(len(cat['operators']) for cat in self.categories.values())} 个操作符**\n")
        
        print(f"README已保存到: {readme_file}")


if __name__ == "__main__":
    url = "https://neo4j.com/docs/cypher-manual/25/planning-and-tuning/operators/operators-detail/"
    
    crawler = Neo4jOperatorCrawler(url)
    crawler.crawl_all()