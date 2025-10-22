import json
import psycopg2
from psycopg2.extras import Json
import numpy as np
import sys
import os

# 添加src目录到Python路径
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, src_path)

from memos.configs.graph_db import GraphDBConfigFactory
from memos.graph_dbs.factory import GraphStoreFactory




# DB_CONFIG = {
#     'host': 'xxxxxxx',
#     'port': 5432,
#     'database': 'xxxxx',
#     'user': 'xxxx',
#     'password': 'xxxx'
# }
#
# # 图数据库配置
GRAPH_NAME = 'memtensor_memos_graph'
def getPolarDb():
    config = GraphDBConfigFactory(
        backend="polardb",
        config={
            "host": "memory.pg.polardb.rds.aliyuncs.com",
            "port": 5432,
            "user": "adimin",
            "password": "Openmem0925",
            "db_name": "memtensor_memos",
            "user_name": 'adimin',
            "use_multi_db": True,  # 设置为True，不添加user_name过滤条件
            "auto_create": True,
            "embedding_dimension": 1024,
        },
    )
    graph = GraphStoreFactory.from_config(config)
    return graph

def create_vector_extension(conn):
    with conn.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    print("✅ pgvector 扩展创建成功或已存在")


def create_table(conn):
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS "Memory" (
        id graphid PRIMARY KEY,
        properties agtype,
        embedding vector(1536),
        embedding_1024 vector(1024),
        embedding_768 vector(768),
        imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    with conn.cursor() as cursor:
        cursor.execute(create_table_sql)

        # 尝试添加主键约束（如果不存在）
        try:
            cursor.execute("ALTER TABLE \"Memory\" ADD CONSTRAINT memory_pkey PRIMARY KEY (id);")
            print("✅ 主键约束添加成功")
        except Exception as e:
            print(f"⚠️ 主键约束可能已存在: {e}")

        # 安全地创建索引，检查列是否存在
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_id ON \"Memory\"(id);")
        except Exception as e:
            print(f"⚠️ 创建ID索引时出错: {e}")

        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_properties ON \"Memory\" USING GIN(properties);")
        except Exception as e:
            print(f"⚠️ 创建properties索引时出错: {e}")

        # 只为存在的embedding列创建索引
        for col in ["embedding", "embedding_1024", "embedding_768"]:
            try:
                cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_memory_{col} ON \"Memory\" USING ivfflat ({col} vector_cosine_ops) WITH (lists = 100);")
            except Exception as e:
                print(f"⚠️ 创建{col}索引时出错: {e}")
    conn.commit()
    print("✅ 表和索引创建成功（如果不存在）")


def convert_to_vector(embedding_list):
    if not embedding_list:
        return None
    if isinstance(embedding_list, np.ndarray):
        embedding_list = embedding_list.tolist()
    return "[" + ",".join(str(float(x)) for x in embedding_list) + "]"


def detect_embedding_field(embedding_list):
    if not embedding_list:
        return None
    dim = len(embedding_list)
    if dim == 1024:
        return "embedding"
    elif dim == 3072:
        return "embedding_3072"
    else:
        print(f"⚠️ 未知 embedding 维度 {dim}，跳过该向量")
        return None


def clean_properties(props):
    """移除向量字段"""
    vector_keys = {"embedding", "embedding_1024", "embedding_3072", "embedding_768"}
    if not isinstance(props, dict):
        return {}
    return {k: v for k, v in props.items() if k not in vector_keys}


def find_embedding(item):
    """在多层结构中查找 embedding 向量"""
    for key in ["embedding", "embedding_1024", "embedding_3072", "embedding_768"]:
        if key in item and isinstance(item[key], list):
            return item[key]
        if "metadata" in item and key in item["metadata"]:
            return item["metadata"][key]
        if "properties" in item and key in item["properties"]:
            return item["properties"][key]
    return None


def add_node(conn, id: str, memory: str, metadata: dict, graph_name=None):
    """
    添加单个节点到图数据库

    Args:
        conn: 数据库连接
        id: 节点ID
        memory: 内存内容
        metadata: 元数据字典
        graph_name: 图名称，可选
    """
    # 使用传入的graph_name或默认值
    if graph_name is None:
        graph_name = GRAPH_NAME

    try:
        # 先提取 embedding（在清理properties之前）
        embedding = find_embedding(metadata)
        field_name = detect_embedding_field(embedding)
        vector_value = convert_to_vector(embedding) if field_name else None

        # 提取 properties
        properties = metadata.copy()
        properties = clean_properties(properties)
        properties["id"] = id
        properties["memory"] = memory

        with conn.cursor() as cursor:
            # 先删除现有记录（如果存在）
            delete_sql = f"""
                DELETE FROM "Memory" 
                WHERE id = ag_catalog._make_graph_id('{graph_name}'::name, 'Memory'::name, %s::text::cstring);
            """
            cursor.execute(delete_sql, (id,))

            # 然后插入新记录
            if field_name and vector_value:
                insert_sql = f"""
                                   INSERT INTO "Memory" (id, properties, {field_name})
                                   VALUES (
                                     ag_catalog._make_graph_id('{graph_name}'::name, 'Memory'::name, %s::text::cstring),
                                     %s::text::agtype,
                                     %s::vector
                                   );
                                   """
                cursor.execute(insert_sql, (id, Json(properties), vector_value))
                print(f"✅ 成功插入/更新: {id} ({field_name})")
            else:
                insert_sql = f"""
                                    INSERT INTO "Memory" (id, properties)
                                    VALUES (
                                      ag_catalog._make_graph_id('{graph_name}'::name, 'Memory'::name, %s::text::cstring),
                                      %s::text::agtype
                                    );
                                    """
                cursor.execute(insert_sql, (id, Json(properties)))
                print(f"✅ 成功插入/更新(无向量): {id}")

        conn.commit()
        return True

    except Exception as e:
        conn.rollback()
        print(f"❌ 插入失败 (ID: {id}): {e}")
        return False


def insert_data(conn, data_list, graph_name=None):
    """
    批量插入数据，使用PolarDB的add_node方法

    Args:
        conn: 数据库连接
        data_list: 数据列表
        graph_name: 图名称，可选
    """
    # 创建PolarDB配置
    # config = GraphDBConfigFactory(
    #     backend="polardb",
    #     config={
    #         "host": "xxxxxxx",
    #         "port": 5432,
    #         "user": "xxxx",
    #         "password": "xxxx",
    #         "db_name": "xxxxx",
    #         "user_name": 'xxxx',
    #         "use_multi_db": False,
    #         "auto_create": False,
    #         "embedding_dimension": 1024,
    #     },
    # )
    #
    # # 创建PolarDB实例
    # graph = GraphStoreFactory.from_config(config)
    graph = getPolarDb()
    print("✅ PolarDB连接成功")
    
    success_count = 0
    error_count = 0

    for item in data_list:
        id_ = item.get("id")
        memory_ = item.get("memory")
        
        # 将所有字段作为metadata，除了id、memory和embedding相关字段
        metadata = {}
        for key, value in item.items():
            if key not in ["id", "memory", "embedding_1024", "embedding_768", "embedding_3072", "embedding"]:
                # 类型转换
                if key == "confidence":
                    # confidence 应该是 float
                    try:
                        metadata[key] = float(value)
                    except (ValueError, TypeError):
                        metadata[key] = value
                elif key == "sources":
                    # sources 应该是 List[str]，每个元素是JSON字符串
                    if isinstance(value, str):
                        try:
                            parsed_sources = json.loads(value)
                            # 将每个对象转换为JSON字符串
                            if isinstance(parsed_sources, list):
                                metadata[key] = [json.dumps(item) for item in parsed_sources]
                            else:
                                metadata[key] = [json.dumps(parsed_sources)]
                        except json.JSONDecodeError:
                            metadata[key] = value
                    else:
                        metadata[key] = value
                elif key == "usage":
                    # usage 应该是 List[str]，每个元素是JSON字符串（和sources格式一样）
                    if isinstance(value, str):
                        try:
                            parsed_usage = json.loads(value)
                            # 将每个对象转换为JSON字符串
                            if isinstance(parsed_usage, list):
                                metadata[key] = [json.dumps(item) for item in parsed_usage]
                            else:
                                metadata[key] = [json.dumps(parsed_usage)]
                        except json.JSONDecodeError:
                            metadata[key] = value
                    else:
                        metadata[key] = value
                elif key == "tags":
                    # tags 应该是 List[str]
                    if isinstance(value, str):
                        # 尝试解析为列表，如果失败则保持原样
                        if value.startswith('[') and value.endswith(']'):
                            try:
                                metadata[key] = json.loads(value)
                            except json.JSONDecodeError:
                                # 如果不是有效的JSON，尝试按逗号分割
                                metadata[key] = [tag.strip() for tag in value[1:-1].split(',')]
                        else:
                            metadata[key] = value
                    else:
                        metadata[key] = value
                else:
                    metadata[key] = value
        
        # 处理embedding字段
        embedding = None
        for embedding_key in ["embedding_1024", "embedding_768", "embedding_3072", "embedding"]:
            if embedding_key in item and item[embedding_key]:
                embedding_value = item[embedding_key]
                # 如果是字符串，尝试解析为列表
                if isinstance(embedding_value, str):
                    try:
                        embedding = json.loads(embedding_value)
                    except json.JSONDecodeError:
                        print(f"⚠️ 无法解析embedding字符串: {embedding_key}")
                        embedding = None
                else:
                    embedding = embedding_value
                break
        
        # 如果有embedding，添加到metadata中
        if embedding:
            metadata["embedding"] = embedding

        try:
            # 直接调用PolarDB的add_node方法
            graph.add_node(id_, memory_, metadata)
            success_count += 1
            print(f"✅ 成功插入/更新: {id_}")
        except Exception as e:
            error_count += 1
            print(f"❌ 插入失败 (ID: {id_}): {e}")
            # PolarDB的add_node方法内部已经处理了事务，不需要外部rollback

    print(f"✅ 插入完成: 成功 {success_count} 条, 失败 {error_count} 条")


def load_data_from_file(filename):
    print("11111")
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"📂 从文件 {filename} 加载了 {len(data)} 条记录")
    return data


def main():
    json_file = r"/Users/zhudayang/python/1011/MemOS/examples/basic_modules/2.json"
    data = load_data_from_file(json_file)
    if not data:
        print("⚠️ 没有数据")
        return

    # conn = psycopg2.connect(**DB_CONFIG)
    print("✅ 数据库连接成功")

    # create_vector_extension(conn)
    # create_table(conn)

    # 使用默认的图名称，或者可以传入自定义的图名称
    # insert_data(conn, data, "custom_graph_name")
    insert_data(None, data)

    # conn.close()
    print("🔒 数据库连接1已关闭")


if __name__ == "__main__":
    main()
