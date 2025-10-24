import json
import os
from collections import Counter
from psycopg2.extras import execute_batch, Json
import psycopg2


class MemoryDataProcessor:
    def __init__(self, db_config):
        """
        初始化数据库连接

        Args:
            db_config: 数据库连接配置
            graph_name: 图数据库名称
        """
        self.db_config = db_config
        self.graph_name = db_config.get('graph_name')
        print("fff:",db_config.get('graph_name'))
        self.connection = None

    def connect(self):
        """连接数据库"""
        try:
            self.connection = psycopg2.connect(
                host=self.db_config["host"],
                port=self.db_config["port"],
                user=self.db_config["user"],
                password=self.db_config["password"],
                database=self.db_config["database"]
            )
            print("✅ 数据库连接成功")
            return True
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
            return False

    def disconnect(self):
        """断开数据库连接"""
        if self.connection:
            self.connection.close()
            print("✅ 数据库连接已关闭")

    def extract_nodes_simple(self, file_path):
        """从 JSON 文件提取 id 和 properties 的简洁版本"""
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"❌ 错误：文件 '{file_path}' 不存在")
                return []

            # 首先尝试用 utf-8-sig 读取（处理 BOM）
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as file:
                    data = json.load(file)
                print("✅ 使用 utf-8-sig 编码成功读取文件")
            except json.JSONDecodeError:
                # 如果 utf-8-sig 失败，尝试用 utf-8
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                    print("✅ 使用 utf-8 编码成功读取文件")
                except json.JSONDecodeError as e:
                    print(f"❌ JSON 解析错误：{e}")
                    return []

            result = []
            tables = data.get('tables', [])

            print(f"📊 找到 {len(tables)} 个表格")

            for i, table in enumerate(tables, 1):
                n_data = table.get('n', {})
                value = n_data.get('value', {})

                # 提取 id 和 properties
                # node_id = value.get('id')
                properties = value.get('properties', {})
                node_id = properties.get('id', {})



                if node_id is not None:
                    # 构建符合插入格式的数据
                    node_data = {
                        "id": str(node_id),  # 转换为字符串
                        "memory": properties.get("memory", ""),
                        "metadata": properties
                    }
                    result.append(node_data)

            print(f"🎯 成功提取 {len(result)} 个节点")
            return result

        except Exception as e:
            print(f"❌ 读取文件时发生错误：{e}")
            return []

    def clean_properties(self, properties):
        """清理 properties，移除不需要的字段"""
        # 移除 embedding 相关字段，这些字段会单独处理
        exclude_fields = [
            "embedding", "embedding_1024", "embedding_768", "embedding_3072",
            "embedding_1024_vector", "embedding_768_vector", "embedding_3072_vector"
        ]

        cleaned = {}
        for key, value in properties.items():
            if key not in exclude_fields:
                cleaned[key] = value

        return cleaned

    def detect_embedding_field(self, embedding):
        """检测 embedding 的维度并返回对应的字段名"""
        if not embedding:
            return None

        if isinstance(embedding, list):
            length = len(embedding)
            if length == 1024:
                return "embedding"
            elif length == 768:
                return "embedding_768"
            elif length == 3072:
                return "embedding_3072"

        return None

    def convert_to_vector(self, embedding):
        """将 embedding 转换为 PostgreSQL 向量格式"""
        if not embedding:
            return None

        try:
            if isinstance(embedding, list):
                # 转换为 PostgreSQL 向量字符串格式: [1,2,3]
                vector_str = "[" + ",".join(map(str, embedding)) + "]"
                return vector_str
            else:
                return None
        except Exception as e:
            print(f"⚠️ 转换向量时出错: {e}")
            return None

    def insert_nodes_to_db(self, nodes, batch_size=1000):
        """将节点数据插入到数据库"""
        if not nodes:
            print("❌ 没有数据可插入")
            return 0, []

        if not self.connection:
            print("❌ 数据库未连接")
            return 0, []

        total_success = 0
        all_errors = []

        # 分批处理
        for i in range(0, len(nodes), batch_size):
            current_batch = nodes[i:i + batch_size]
            batch_success = 0
            batch_errors = []

            print(
                f"🔄 处理批次 {i // batch_size + 1}/{(len(nodes) - 1) // batch_size + 1} ({len(current_batch)} 个节点)")

            try:
                with self.connection.cursor() as cursor:
                    # 准备批量插入数据
                    insert_data_1024 = []
                    insert_data_no_embedding = []

                    for node in current_batch:
                        try:
                            id_ = node["id"]
                            memory_ = node["memory"]
                            metadata = node["metadata"]

                            # 提取 embedding
                            embedding = None
                            for embedding_key in ["embedding_1024", "embedding_768", "embedding_3072", "embedding"]:
                                if embedding_key in metadata and metadata[embedding_key]:
                                    embedding = metadata[embedding_key]
                                    break

                            if isinstance(embedding, str):
                                try:
                                    embedding = json.loads(embedding)
                                except json.JSONDecodeError:
                                    print(f"⚠️ 无法解析embedding字符串: {embedding_key}")
                                    embedding = None

                            # 清理 properties
                            properties = self.clean_properties(metadata)
                            properties["id"] = id_
                            properties["memory"] = memory_

                            # 生成 graph_id 并添加到 properties
                            try:
                                get_graph_id_query = f"""
                                                            SELECT ag_catalog._make_graph_id('{self.graph_name}'::name, 'Memory'::name, %s::text::cstring)
                                                        """
                                cursor.execute(get_graph_id_query, (id_,))
                                graph_id = cursor.fetchone()[0]
                                properties['graph_id'] = str(graph_id)
                            except Exception as e:
                                print(f"⚠️ 生成 graph_id 失败: {e}")
                                properties['graph_id'] = str(id_)  # 备用方案


                            # 根据embedding维度分类
                            field_name = self.detect_embedding_field(embedding)
                            vector_value = self.convert_to_vector(embedding) if field_name else None

                            if field_name == "embedding" and vector_value:
                                insert_data_1024.append((id_, Json(properties), vector_value))
                            else:
                                insert_data_no_embedding.append((id_, Json(properties)))

                        except Exception as e:
                            batch_errors.append(f"ID: {node.get('id', 'unknown')} - {e}")

                    # 批量插入不同维度的数据
                    if insert_data_1024:
                        insert_sql_1024 = f"""
                            INSERT INTO "Memory" (id, properties, embedding)
                            VALUES (ag_catalog._make_graph_id('{self.graph_name}'::name, 'Memory'::name, %s::text::cstring), %s, %s)
                        """
                        execute_batch(cursor, insert_sql_1024, insert_data_1024)
                        batch_success += len(insert_data_1024)
                        print(f"  ✅ 插入 {len(insert_data_1024)} 个带 embedding 的节点")

                    if insert_data_no_embedding:
                        insert_sql_no_embedding = f"""
                            INSERT INTO "Memory" (id, properties)
                            VALUES (ag_catalog._make_graph_id('{self.graph_name}'::name, 'Memory'::name, %s::text::cstring), %s)
                        """
                        execute_batch(cursor, insert_sql_no_embedding, insert_data_no_embedding)
                        batch_success += len(insert_data_no_embedding)
                        print(f"  ✅ 插入 {len(insert_data_no_embedding)} 个无 embedding 的节点")

                    # 提交当前批次
                    self.connection.commit()
                    total_success += batch_success
                    all_errors.extend(batch_errors)

                    print(f"  ✅ 批次完成: {batch_success} 个节点插入成功")

            except Exception as e:
                self.connection.rollback()
                batch_errors.append(f"批次插入失败: {e}")
                all_errors.extend(batch_errors)
                print(f"❌ 批次插入失败: {e}")

        return total_success, all_errors

    def process_file(self, file_path, batch_size):
        """完整处理流程：提取数据并插入数据库"""
        print("🚀 开始处理数据文件...")

        # 1. 提取数据
        nodes = self.extract_nodes_simple(file_path)
        if not nodes:
            return

        # 3. 连接数据库
        if not self.connect():
            return

        try:
            # 4. 插入数据到数据库
            print(f"\n💾 开始插入数据到数据库...")
            success_count, errors = self.insert_nodes_to_db(nodes, batch_size)

            # 5. 显示结果
            print(f"\n🎉 处理完成!")
            print(f"✅ 成功插入: {success_count}/{len(nodes)} 个节点")
            print(f"❌ 错误数量: {len(errors)}")

            if errors:
                print(f"\n📋 错误详情 (前10个):")
                for error in errors[:10]:
                    print(f"  - {error}")
                if len(errors) > 10:
                    print(f"  ... 还有 {len(errors) - 10} 个错误")

        finally:
            # 6. 断开数据库连接
            self.disconnect()


# 使用示例
if __name__ == "__main__":
    # 数据库配置（请根据实际情况修改）
    POLARDB_CONFIG = {
        "host": "xxx",
        "port": 5432,
        "user": "xxx",
        "password": "xxx",
        "database": "xxx",
        # "database": "test_zdy",
        "graph_name": "xxx"
        # "graph_name": "test_zdy_graph"
    }

    # 文件路径
    file_path = "/Users/ccl/Desktop/file/temp/result.json"

    # 创建处理器实例
    processor = MemoryDataProcessor(POLARDB_CONFIG)

    # 处理文件
    processor.process_file(file_path, batch_size=1000)