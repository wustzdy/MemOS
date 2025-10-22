import json
import psycopg2
from psycopg2.extras import Json, execute_batch
import numpy as np
import sys
import os
from datetime import datetime

# PolarDB 配置
POLARDB_CONFIG = {
    "host": "memory.pg.polardb.rds.aliyuncs.com",
    "port": 5432,
    "user": "adimin",
    "password": "Openmem0925",
    # "database": "memtensor_memos",
    "database": "test_zdy",
    # "graph_name": "memtensor_memos_graph"
    "graph_name": "test_zdy_graph"
}


class PolarDBGraph:
    def __init__(self, config):
        self.config = config
        self.connection = psycopg2.connect(
            host=config["host"],
            port=config["port"],
            user=config["user"],
            password=config["password"],
            database=config["database"]
        )
        self.graph_name = config.get("graph_name")
        # 设置自动提交为False，手动控制事务
        self.connection.autocommit = False
        print("✅ PolarDB连接成功")

    def update_graph_id_in_properties(self):
        """更新properties字段，添加graph_id"""
        print("🔄 开始更新properties字段，添加graph_id...")
        start_time = datetime.now()

        try:
            with self.connection.cursor() as cursor:
                # 执行UPDATE语句，将graph_id添加到properties中
                update_sql = f"""
                    UPDATE {self.graph_name}."Memory"
                    SET properties = agtype_concat(properties, agtype_build_map('graph_id', id::text))
                """
                cursor.execute(update_sql)
                updated_count = cursor.rowcount

                self.connection.commit()

                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"✅ 成功更新 {updated_count} 条记录的properties字段，耗时: {elapsed:.2f}秒")
                return updated_count

        except Exception as e:
            self.connection.rollback()
            print(f"❌ 更新properties字段失败: {e}")
            return 0

    def batch_add_nodes_optimized(self, nodes, batch_size=1000):
        """优化版批量插入节点"""
        success_count = 0
        error_count = 0
        total_nodes = len(nodes)

        print(f"🚀 开始处理 {total_nodes} 条记录，批次大小: {batch_size}")
        start_time = datetime.now()

        # 按批次处理
        for batch_start in range(0, total_nodes, batch_size):
            batch_end = min(batch_start + batch_size, total_nodes)
            current_batch = nodes[batch_start:batch_end]

            batch_success = 0
            batch_errors = []

            try:
                with self.connection.cursor() as cursor:

                    # 准备批量插入数据
                    insert_data_1024 = []
                    # insert_data_768 = []
                    # insert_data_3072 = []
                    insert_data_no_embedding = []

                    for node in current_batch:
                        try:
                            id_ = node["id"]
                            memory_ = node["memory"]
                            metadata = node["metadata"]

                            # get_graph_id_query = f"""
                            #                  SELECT ag_catalog._make_graph_id('{self.graph_name}'::name, 'Memory'::name, %s::text::cstring)
                            #              """
                            # cursor.execute(get_graph_id_query, (id_,))
                            # graph_id = cursor.fetchone()[0]
                            # properties['graph_id'] = str(graph_id)

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

                            # 根据embedding维度分类
                            field_name = self.detect_embedding_field(embedding)
                            vector_value = self.convert_to_vector(embedding) if field_name else None

                            if field_name == "embedding" and vector_value:
                                insert_data_1024.append((id_, Json(properties), vector_value))
                            # elif field_name == "embedding_768" and vector_value:
                            #     insert_data_768.append((id_, Json(properties), vector_value))
                            # elif field_name == "embedding_3072" and vector_value:
                            #     insert_data_3072.append((id_, Json(properties), vector_value))
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

                    # if insert_data_768:
                    #     insert_sql_768 = f"""
                    #         INSERT INTO "Memory" (id, properties, embedding_768)
                    #         VALUES (ag_catalog._make_graph_id('{self.graph_name}'::name, 'Memory'::name, %s::text::cstring), %s, %s)
                    #     """
                    #     execute_batch(cursor, insert_sql_768, insert_data_768)
                    #     batch_success += len(insert_data_768)
                    #
                    # if insert_data_3072:
                    #     insert_sql_3072 = f"""
                    #         INSERT INTO "Memory" (id, properties, embedding_3072)
                    #         VALUES (ag_catalog._make_graph_id('{self.graph_name}'::name, 'Memory'::name, %s::text::cstring), %s, %s)
                    #     """
                    #     execute_batch(cursor, insert_sql_3072, insert_data_3072)
                    #     batch_success += len(insert_data_3072)

                    if insert_data_no_embedding:
                        insert_sql_no_embedding = f"""
                            INSERT INTO "Memory" (id, properties)
                            VALUES (ag_catalog._make_graph_id('{self.graph_name}'::name, 'Memory'::name, %s::text::cstring), %s)
                        """
                        execute_batch(cursor, insert_sql_no_embedding, insert_data_no_embedding)
                        batch_success += len(insert_data_no_embedding)

                # 提交当前批次
                self.connection.commit()
                success_count += batch_success
                error_count += len(batch_errors)

                # 进度显示
                elapsed = (datetime.now() - start_time).total_seconds()
                progress = (batch_end / total_nodes) * 100
                estimated_total = (elapsed / batch_end) * total_nodes if batch_end > 0 else 0
                remaining = estimated_total - elapsed

                print(f"📊 进度: {batch_end}/{total_nodes} ({progress:.1f}%) | "
                      f"成功: {success_count} | 失败: {error_count} | "
                      f"已用: {elapsed:.0f}s | 剩余: {remaining:.0f}s")

                # 输出批次错误
                if batch_errors:
                    print(f"❌ 本批次错误: {len(batch_errors)} 条")
                    for i, error in enumerate(batch_errors[:5]):  # 只显示前5个错误
                        print(f"   {i + 1}. {error}")
                    if len(batch_errors) > 5:
                        print(f"   ... 还有 {len(batch_errors) - 5} 个错误")

            except Exception as e:
                self.connection.rollback()
                error_count += len(current_batch)
                print(f"❌ 批次 {batch_start}-{batch_end} 整体失败: {e}")

        total_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ 批量插入完成: 成功 {success_count} 条, 失败 {error_count} 条, 总耗时: {total_time:.2f}秒")

        return success_count, error_count

    def clean_properties(self, props):
        """移除向量字段"""
        vector_keys = {"embedding", "embedding_1024", "embedding_3072", "embedding_768"}
        if not isinstance(props, dict):
            return {}
        return {k: v for k, v in props.items() if k not in vector_keys}

    def detect_embedding_field(self, embedding_list):
        """检测 embedding 维度并返回对应的字段名"""
        if not embedding_list:
            return None
        dim = len(embedding_list)
        # print("---------",dim)
        if dim == 1024:
            return "embedding"
        elif dim == 768:
            return "embedding_768"
        elif dim == 3072:
            return "embedding_3072"
        else:
            print(f"⚠️ 未知 embedding 维度 {dim}，跳过该向量")
            return None

    def convert_to_vector(self, embedding_list):
        """将 embedding 列表转换为向量字符串"""
        if not embedding_list:
            return None
        if isinstance(embedding_list, np.ndarray):
            embedding_list = embedding_list.tolist()
        return "[" + ",".join(str(float(x)) for x in embedding_list) + "]"

    def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            print("🔒 PolarDB连接已关闭")


def getPolarDb():
    """直接创建 PolarDB 图数据库实例"""
    return PolarDBGraph(POLARDB_CONFIG)


def process_metadata(item):
    """处理元数据，提取和转换字段"""
    metadata = {}
    for key, value in item.items():
        if key not in ["id", "memory"]:
            # 类型转换
            if key == "confidence":
                try:
                    metadata[key] = float(value)
                except (ValueError, TypeError):
                    metadata[key] = value
            elif key == "sources" or key == "usage":
                if isinstance(value, str):
                    try:
                        parsed_value = json.loads(value)
                        metadata[key] = [json.dumps(item) for item in parsed_value] if isinstance(parsed_value,
                                                                                                  list) else [
                            json.dumps(parsed_value)]
                    except json.JSONDecodeError:
                        metadata[key] = value
                else:
                    metadata[key] = value
            elif key == "tags":
                if isinstance(value, str):
                    if value.startswith('[') and value.endswith(']'):
                        try:
                            metadata[key] = json.loads(value)
                        except json.JSONDecodeError:
                            metadata[key] = [tag.strip() for tag in value[1:-1].split(',')]
                    else:
                        metadata[key] = value
                else:
                    metadata[key] = value
            else:
                metadata[key] = value
    return metadata


def extract_embedding(item):
    """从数据项中提取embedding"""
    embedding = None
    for embedding_key in ["embedding_1024", "embedding_768", "embedding_3072", "embedding"]:
        if embedding_key in item and item[embedding_key]:
            embedding_value = item[embedding_key]
            if isinstance(embedding_value, str):
                try:
                    embedding = json.loads(embedding_value)
                except json.JSONDecodeError:
                    print(f"⚠️ 无法解析embedding字符串: {embedding_key}")
                    embedding = None
            else:
                embedding = embedding_value
            break
    return embedding


def prepare_nodes_for_insertion(data_list):
    """准备要插入的节点数据"""
    nodes_to_insert = []
    processed_count = 0
    skipped_count = 0

    for item in data_list:
        id_ = item.get("id")
        memory_ = item.get("memory")

        if not id_ or not memory_:
            print(f"⚠️ 跳过无效数据: ID或memory为空")
            skipped_count += 1
            continue

        # 处理元数据
        metadata = process_metadata(item)

        # 处理embedding字段
        embedding = extract_embedding(item)
        if embedding:
            metadata["embedding"] = embedding

        # 构建插入的数据
        nodes_to_insert.append({
            "id": id_,
            "memory": memory_,
            "metadata": metadata
        })
        processed_count += 1

        # 显示进度
        if processed_count % 10000 == 0:
            print(f"📝 已预处理 {processed_count} 条数据")

    print(f"✅ 数据预处理完成: 有效 {processed_count} 条, 跳过 {skipped_count} 条")
    return nodes_to_insert


def insert_data_optimized(data_list, batch_size=1000):
    """优化版数据插入"""
    graph = getPolarDb()

    # 数据预处理
    print("🔄 开始预处理数据...")
    nodes_to_insert = prepare_nodes_for_insertion(data_list)

    if not nodes_to_insert:
        print("⚠️ 没有有效数据需要插入")
        graph.close()
        return 0, 0

    # 使用优化版批量插入
    print("🚀 开始批量插入数据...")
    success_count, error_count = graph.batch_add_nodes_optimized(nodes_to_insert, batch_size)

    graph.close()
    return success_count, error_count



def load_data_from_file(filename):
    """从文件加载数据"""
    print(f"📂 正在加载文件: {filename}")
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"📂 从文件 {filename} 加载了 {len(data)} 条记录")
        return data
    except Exception as e:
        print(f"❌ 加载文件失败: {e}")
        return []

def update_graph():
    print("-----------update_graph[start]")
    graph = getPolarDb()
    graph.update_graph_id_in_properties()
    print("---------update_graph[end]")

def insert_data(conn, data):
    # 记录总开始时间
    total_start_time = datetime.now()


    if not data:
        print("⚠️ 没有数据")
        return

    print(f"🎯 总共需要处理 {len(data)} 条记录")
    success_count, error_count = insert_data_optimized(data, batch_size=1000)

    # 计算总耗时
    total_time = (datetime.now() - total_start_time).total_seconds()
    minutes, seconds = divmod(total_time, 60)
    hours, minutes = divmod(minutes, 60)

    print(f"\n🎉 处理完成!")
    print(f"📊 最终结果:")
    print(f"   ✅ 成功: {success_count} 条")
    print(f"   ❌ 失败: {error_count} 条")
    print(f"   ⏱️  总耗时: {int(hours)}小时{int(minutes)}分钟{seconds:.2f}秒")

def main():
    json_file = r"/Users/ccl/Desktop/file/export13/ceshi/ceshi.json"

    # 记录总开始时间
    total_start_time = datetime.now()

    # 加载数据
    data = load_data_from_file(json_file)
    if not data:
        print("⚠️ 没有数据")
        return

    print(f"🎯 总共需要处理 {len(data)} 条记录")

    # 使用优化版本，设置批次大小为1000
    # 可以根据实际情况调整批次大小：
    # - 网络好：1000-2000
    # - 网络一般：500-1000
    # - 内存有限：200-500
    success_count, error_count = insert_data_optimized(data, batch_size=1000)

    # 计算总耗时
    total_time = (datetime.now() - total_start_time).total_seconds()
    minutes, seconds = divmod(total_time, 60)
    hours, minutes = divmod(minutes, 60)

    print(f"\n🎉 处理完成!")
    print(f"📊 最终结果:")
    print(f"   ✅ 成功: {success_count} 条")
    print(f"   ❌ 失败: {error_count} 条")
    print(f"   ⏱️  总耗时: {int(hours)}小时{int(minutes)}分钟{seconds:.2f}秒")

    if success_count > 0:
        records_per_second = success_count / total_time
        print(f"   🚀 处理速度: {records_per_second:.2f} 条/秒")


if __name__ == "__main__":
    main()