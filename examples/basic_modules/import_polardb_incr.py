import json
import os
from collections import Counter
from psycopg2.extras import execute_batch, Json
import psycopg2


class MemoryDataProcessor:
    def __init__(self, db_config):
        """
        Initialize database connection

        Args:
            db_config: Database connection configuration
            graph_name: Graph database name
        """
        self.db_config = db_config
        self.graph_name = db_config.get('graph_name')
        print("fff:",db_config.get('graph_name'))
        self.connection = None

    def connect(self):
        """Connect to database"""
        try:
            self.connection = psycopg2.connect(
                host=self.db_config["host"],
                port=self.db_config["port"],
                user=self.db_config["user"],
                password=self.db_config["password"],
                database=self.db_config["database"]
            )
            print("✅ Database connection successful")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect database connection"""
        if self.connection:
            self.connection.close()
            print("✅ Database connection closed")

    def extract_nodes_simple(self, file_path):
        """Extract simplified id and properties from JSON file"""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"❌ Error: File '{file_path}' does not exist")
                return []

            # First try reading with utf-8-sig (handle BOM)
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as file:
                    data = json.load(file)
                print("✅ Successfully read file with utf-8-sig encoding")
            except json.JSONDecodeError:
                # If utf-8-sig fails, try utf-8
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                    print("✅ Successfully read file with utf-8 encoding")
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parse error: {e}")
                    return []

            result = []
            tables = data.get('tables', [])

            print(f"📊 Found {len(tables)} tables")

            for i, table in enumerate(tables, 1):
                n_data = table.get('n', {})
                value = n_data.get('value', {})

                # Extract id and properties
                # node_id = value.get('id')
                properties = value.get('properties', {})
                node_id = properties.get('id', {})



                if node_id is not None:
                    # Build data in insertion format
                    node_data = {
                        "id": str(node_id),  # 转换为字符串
                        "memory": properties.get("memory", ""),
                        "metadata": properties
                    }
                    result.append(node_data)

            print(f"🎯 Successfully extracted {len(result)} nodes")
            return result

        except Exception as e:
            print(f"❌ Error occurred while reading file: {e}")
            return []

    def clean_properties(self, properties):
        """Clean properties and remove unnecessary fields"""
        # Remove embedding-related fields; these will be handled separately
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
        """Detect embedding dimension and return corresponding field name"""
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
        """Convert embedding to PostgreSQL vector format"""
        if not embedding:
            return None

        try:
            if isinstance(embedding, list):
                # Convert to PostgreSQL vector string format: [1,2,3]
                vector_str = "[" + ",".join(map(str, embedding)) + "]"
                return vector_str
            else:
                return None
        except Exception as e:
            print(f"⚠️ Error converting vector: {e}")
            return None

    def insert_nodes_to_db(self, nodes, batch_size=1000):
        """Insert node data into the database"""
        if not nodes:
            print("❌ No data to insert")
            return 0, []

        if not self.connection:
            print("❌ Database not connected")
            return 0, []

        total_success = 0
        all_errors = []

        # 分批处理
        for i in range(0, len(nodes), batch_size):
            current_batch = nodes[i:i + batch_size]
            batch_success = 0
            batch_errors = []

            print(
                f"🔄 Processing batch {i // batch_size + 1}/{(len(nodes) - 1) // batch_size + 1} ({len(current_batch)} nodes)")

            try:
                with self.connection.cursor() as cursor:
                    # Prepare batch insert data
                    insert_data_1024 = []
                    insert_data_no_embedding = []

                    for node in current_batch:
                        try:
                            id_ = node["id"]
                            memory_ = node["memory"]
                            metadata = node["metadata"]

                            embedding = None
                            for embedding_key in ["embedding_1024", "embedding_768", "embedding_3072", "embedding"]:
                                if embedding_key in metadata and metadata[embedding_key]:
                                    embedding = metadata[embedding_key]
                                    break

                            if isinstance(embedding, str):
                                try:
                                    embedding = json.loads(embedding)
                                except json.JSONDecodeError:
                                    print(f"⚠️ Unable to parse embedding string: {embedding_key}")
                                    embedding = None

                            properties = self.clean_properties(metadata)
                            properties["id"] = id_
                            properties["memory"] = memory_

                            try:
                                get_graph_id_query = f"""
                                                            SELECT ag_catalog._make_graph_id('{self.graph_name}'::name, 'Memory'::name, %s::text::cstring)
                                                        """
                                cursor.execute(get_graph_id_query, (id_,))
                                graph_id = cursor.fetchone()[0]
                                properties['graph_id'] = str(graph_id)
                            except Exception as e:
                                print(f"⚠️ Failed to generate graph_id: {e}")
                                properties['graph_id'] = str(id_)


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
                        print(f"  ✅ Inserted {len(insert_data_1024)} nodes with embedding")

                    if insert_data_no_embedding:
                        insert_sql_no_embedding = f"""
                            INSERT INTO "Memory" (id, properties)
                            VALUES (ag_catalog._make_graph_id('{self.graph_name}'::name, 'Memory'::name, %s::text::cstring), %s)
                        """
                        execute_batch(cursor, insert_sql_no_embedding, insert_data_no_embedding)
                        batch_success += len(insert_data_no_embedding)
                        print(f"  ✅ Inserted {len(insert_data_no_embedding)} nodes without embedding")

                    # 提交当前批次
                    self.connection.commit()
                    total_success += batch_success
                    all_errors.extend(batch_errors)

                    print(f"  ✅ Batch complete: {batch_success} nodes inserted successfully")

            except Exception as e:
                self.connection.rollback()
                batch_errors.append(f"Batch insert failed: {e}")
                all_errors.extend(batch_errors)
                print(f"❌ Batch insertion failed: {e}")

        return total_success, all_errors

    def process_file(self, file_path, batch_size):
        """Complete processing flow: extract data and insert into database"""
        print("🚀 Starting to process data file...")

        # 1. Extract data
        nodes = self.extract_nodes_simple(file_path)
        if not nodes:
            return

        # 3. Connect to database
        if not self.connect():
            return

        try:
            # 4. Insert data into database
            print(f"\n💾 Starting to insert data into database...")
            success_count, errors = self.insert_nodes_to_db(nodes, batch_size)

            # 5. Display results
            print(f"\n🎉 Processing complete!")
            print(f"✅ Successfully inserted: {success_count}/{len(nodes)} nodes")
            print(f"❌ Error count: {len(errors)}")

            if errors:
                print(f"\n📋 Error details (first 10):")
                for error in errors[:10]:
                    print(f"  - {error}")
                if len(errors) > 10:
                    print(f"  ... {len(errors) - 10} more errors")

        finally:
            # 6. Disconnect database connection
            self.disconnect()


if __name__ == "__main__":

    POLARDB_CONFIG = {
        "host": "xxx",
        "port": 5432,
        "user": "xxx",
        "password": "xxx",
        "database": "xxx",
        "graph_name": "xxx"

    }


    file_path = "/Users/ccl/Desktop/file/temp/result.json"


    processor = MemoryDataProcessor(POLARDB_CONFIG)


    processor.process_file(file_path, batch_size=1000)