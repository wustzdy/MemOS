import os
import json
import psycopg2

# 数据库连接配置
DB_CONFIG = {
    'host': 'xxx',
    'port': 5432,
    'database': 'xxx',
    'user': 'xxx',
    'password': 'xxx'
}

# 顶层目录
EDGE_ROOT_DIR = r"/Users/ccl/Desktop/file/ccl/export22"

# 合法的关系文件夹（白名单）
VALID_REL_TYPES = {
    "AGGREGATE_TO",
    "FOLLOWS",
    "INFERS",
    "MERGED_TO",
    "RELATE_TO",
    "PARENT"
}

BATCH_SIZE = 1000


def insert_edges(conn, edges, label_name):
    with conn.cursor() as cur:
        for e in edges:
            src_id = e["src_id"]
            dst_id = e["dst_id"]
            user_name = e["user_name"]

            sql = f"""
                INSERT INTO memtensor_memos_graph."{label_name}"(id, start_id, end_id, properties)
                SELECT
                    ag_catalog._next_graph_id('memtensor_memos_graph'::name, '{label_name}'),
                    ag_catalog._make_graph_id('memtensor_memos_graph'::name, 'Memory'::name, '{src_id}'::text::cstring),
                    ag_catalog._make_graph_id('memtensor_memos_graph'::name, 'Memory'::name, '{dst_id}'::text::cstring),
                    jsonb_build_object('user_name', '{user_name}')::text::agtype
                WHERE NOT EXISTS (
                    SELECT 1 FROM memtensor_memos_graph."{label_name}"
                    WHERE start_id = ag_catalog._make_graph_id('memtensor_memos_graph'::name, 'Memory'::name, '{src_id}'::text::cstring)
                      AND end_id   = ag_catalog._make_graph_id('memtensor_memos_graph'::name, 'Memory'::name, '{dst_id}'::text::cstring)
                );
            """
            cur.execute(sql)
        conn.commit()


def process_relation_folder(conn, folder_path, label_name):
    print(f"\n🔗 Processing relation: {label_name}")

    # create_elabel(conn, label_name)
    for root, _, files in os.walk(folder_path):
        for file in files:
            if not (file.endswith(".json") or file.endswith(".txt")):
                continue
            file_path = os.path.join(root, file)
            print(f"📄 Reading file: {file_path}")
            batch = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        batch.append(obj)
                    except json.JSONDecodeError:
                        print(f"⚠️ JSON decode error in {file_path}: {line}")
                        continue

                    if len(batch) >= BATCH_SIZE:
                        insert_edges(conn, batch, label_name)
                        print(f"✅ Inserted (or skipped) {len(batch)} edges.")
                        batch.clear()

            if batch:
                insert_edges(conn, batch, label_name)
                print(f"✅ Inserted (or skipped) {len(batch)} edges.")


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        for folder_name in os.listdir(EDGE_ROOT_DIR):
            folder_path = os.path.join(EDGE_ROOT_DIR, folder_name)
            if not os.path.isdir(folder_path):
                continue


            if folder_name.upper() not in VALID_REL_TYPES:
                print(f"🚫 Skipping non-relation folder: {folder_name}")
                continue


            label_name = folder_name
            process_relation_folder(conn, folder_path, label_name)

        print("\n🎉 All relation folders processed successfully!")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
