import os
import json
import psycopg2
import sys

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polardb_export_insert import insert_data

DB_CONFIG = {
    'host': 'xxxxxxx',
    'port': 5432,
    'database': 'xxxxx',
    'user': 'xxxx',
    'password': 'xxxx'
}
conn = psycopg2.connect(**DB_CONFIG)

def insert(batch):
    """
    模拟插入函数。
    这里你可以替换成实际数据库或API调用逻辑。
    """
    print(f"✅ 调用 insert() 插入 {len(batch)} 条记录")
    insert_data(conn, batch)
    # 示例：你的数据库插入逻辑写在这里
    # db.insert_many(batch)


def process_folder(folder_path, batch_size=1000):
    """
    遍历文件夹，按 batch_size 分批解析 JSON 并调用 insert。
    """
    batch = []
    total_count = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Only process .json files
            if not file.endswith('.json'):
                continue
                
            file_path = os.path.join(root, file)
            print(f"📄 正在读取文件: {file_path}")

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            # 确保解析出的对象是字典类型，并且包含必要的字段
                            if isinstance(obj, dict) and "id" in obj and "memory" in obj:
                                batch.append(obj)
                                total_count += 1

                                # 每满 batch_size 条，调用 insert 并清空缓存
                                if len(batch) >= batch_size:
                                    insert(batch)
                                    batch = []  # 清空
                            else:
                                print(f"⚠️ 跳过无效对象（缺少必要字段）: {line[:80]}...")
                        except json.JSONDecodeError:
                            print(f"⚠️ 跳过无效 JSON: {line[:80]}...")
            except (UnicodeDecodeError, IOError) as e:
                print(f"⚠️ 跳过无法读取的文件 {file_path}: {e}")
                continue

    # 处理最后不足 batch_size 的部分
    if batch:
        insert(batch)

    print(f"\n✅ 全部完成，共处理 {total_count} 条记录。")


if __name__ == "__main__":
    folder_path = r"/Users/zhudayang/python/github/1/MemOS/examples/basic_modules"
    process_folder(folder_path, batch_size=1000)
