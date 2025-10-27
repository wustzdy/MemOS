import os
import json
import psycopg2
import sys

# Add the parent directory to the path to allow imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, src_path)

# from polardb_export_insert_1 import insert_data
from batchImport_polardbFromJson import insert_data, update_graph

DB_CONFIG = {
    'host': 'xxx',
    'port': 5432,
    'database': 'xx',
    'user': 'xx',
    'password': 'xx'
}
conn = psycopg2.connect(**DB_CONFIG)

def insert(batch):

    print(f"✅  insert()  {len(batch)} records")
    insert_data(conn, batch)



def process_folder(folder_path, batch_size=1000):

    batch = []
    total_count = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Only process .json files
            if not file.endswith('.json'):
                continue
                
            file_path = os.path.join(root, file)
            print(f"📄 read file: {file_path}")

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)

                            if isinstance(obj, dict) and "id" in obj and "memory" in obj:
                                batch.append(obj)
                                total_count += 1


                                if len(batch) >= batch_size:
                                    insert(batch)
                                    batch = []
                                if "sources" in obj and isinstance(obj["sources"], str):
                                    if not("user:" in obj["sources"] )and not("assistant:" in obj["sources"]):
                                        continue
                                    try:
                                        import re

                                        cleaned_sources = obj["sources"].replace('\n', '').replace('\\n', '').replace('\ufeff', '')


                                        if cleaned_sources.startswith('[') and cleaned_sources.endswith(']'):
                                            inner_str = cleaned_sources[1:-1].strip()

                                            parts = re.split(r',\s*(?=\w+:)', inner_str)

                                            parts = [part.strip() for part in parts]
                                            obj["sources"] = parts
                                        else:

                                            obj["sources"] = [cleaned_sources]
                                    except Exception as e:
                                        print(f"⚠️ not parse sources: {e}")
                                        print(f"⚠️ source content: {obj['sources'][:100]}...")
                                        obj["sources"] = []
                            else:
                                print(f"⚠️ skip: {line[:80]}...")
                        except json.JSONDecodeError:
                            print(f"⚠️ skil valid JSON: {line[:80]}...")
            except (UnicodeDecodeError, IOError) as e:
                print(f"⚠️ skip file {file_path}: {e}")
                continue

    # 处理最后不足 batch_size 的部分
    if batch:
        insert(batch)
        update_graph()

    print(f"\n✅ end，total {total_count} records。")


if __name__ == "__main__":
    # folder_path = r"/Users/ccl/Desktop/file/export13/ceshi"
    # 10W
    folder_path = r"/Users/ccl/Desktop/file/export15/Memory"
    # 70W
    folder_path = r"/Users/ccl/Desktop/file/ccl/export22/Memory"
    process_folder(folder_path, batch_size=1000)
