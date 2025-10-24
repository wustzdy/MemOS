import json
import psycopg2
from psycopg2.extras import Json, execute_batch
import numpy as np
import sys
import os
from datetime import datetime

# PolarDB configuration
POLARDB_CONFIG = {
    "host": "xxx",
    "port": 5432,
    "user": "xxx",
    "password": "xxx",
    # "database": "xxx",
    "database": "xxx",
    # "graph_name": "xxx"
    "graph_name": "xxx"
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
        # Set autocommit to False to manually control transactions
        self.connection.autocommit = False
        print("✅ PolarDB connection successful")

    def update_graph_id_in_properties(self):
        """Update properties field to add graph_id"""
        print("🔄 Starting to update properties field, adding graph_id...")
        start_time = datetime.now()

        try:
            with self.connection.cursor() as cursor:
                # Execute UPDATE to add graph_id into properties
                update_sql = f"""
                    UPDATE {self.graph_name}."Memory"
                    SET properties = agtype_concat(properties, agtype_build_map('graph_id', id::text))
                """
                cursor.execute(update_sql)
                updated_count = cursor.rowcount

                self.connection.commit()

                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"✅ Successfully updated {updated_count} records' properties, elapsed: {elapsed:.2f}s")
                return updated_count

        except Exception as e:
            self.connection.rollback()
            print(f"❌ Failed to update properties field: {e}")
            return 0

    def batch_add_nodes_optimized(self, nodes, batch_size=1000):
        """Optimized batch insertion of nodes"""
        success_count = 0
        error_count = 0
        total_nodes = len(nodes)

        print(f"🚀 Start processing {total_nodes} records, batch size: {batch_size}")
        start_time = datetime.now()

        # Process in batches
        for batch_start in range(0, total_nodes, batch_size):
            batch_end = min(batch_start + batch_size, total_nodes)
            current_batch = nodes[batch_start:batch_end]

            batch_success = 0
            batch_errors = []

            try:
                with self.connection.cursor() as cursor:

                    # Prepare batch insert data
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

                            # Extract embedding
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
                            # Clean properties
                            properties = self.clean_properties(metadata)
                            properties["id"] = id_
                            properties["memory"] = memory_

                            # Classify by embedding dimension
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

                    # Batch insert for different dimensions
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

                # Commit current batch
                self.connection.commit()
                success_count += batch_success
                error_count += len(batch_errors)

                # Progress display
                elapsed = (datetime.now() - start_time).total_seconds()
                progress = (batch_end / total_nodes) * 100
                estimated_total = (elapsed / batch_end) * total_nodes if batch_end > 0 else 0
                remaining = estimated_total - elapsed

                print(f"📊 Progress: {batch_end}/{total_nodes} ({progress:.1f}%) | "
                      f"Success: {success_count} | Failures: {error_count} | "
                      f"Elapsed: {elapsed:.0f}s | Remaining: {remaining:.0f}s")

                # Output batch errors
                if batch_errors:
                    print(f"❌ Errors in this batch: {len(batch_errors)}")
                    for i, error in enumerate(batch_errors[:5]):  # Only show first 5 errors
                        print(f"   {i + 1}. {error}")
                    if len(batch_errors) > 5:
                        print(f"   ... {len(batch_errors) - 5} more errors")

            except Exception as e:
                self.connection.rollback()
                error_count += len(current_batch)
                print(f"❌ Batch {batch_start}-{batch_end} failed: {e}")

        total_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Batch insertion complete: Success {success_count}, Failures {error_count}, Total time: {total_time:.2f}s")

        return success_count, error_count

    def clean_properties(self, props):
        """Remove vector fields"""
        vector_keys = {"embedding", "embedding_1024", "embedding_3072", "embedding_768"}
        if not isinstance(props, dict):
            return {}
        return {k: v for k, v in props.items() if k not in vector_keys}

    def detect_embedding_field(self, embedding_list):
        """Detect embedding dimension and return corresponding field name"""
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
            print(f"⚠️ Unknown embedding dimension {dim}, skipping vector")
            return None

    def convert_to_vector(self, embedding_list):
        """Convert embedding list to vector string"""
        if not embedding_list:
            return None
        if isinstance(embedding_list, np.ndarray):
            embedding_list = embedding_list.tolist()
        return "[" + ",".join(str(float(x)) for x in embedding_list) + "]"

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("🔒 PolarDB connection closed")


def getPolarDb():
    """Create PolarDB graph database instance"""
    return PolarDBGraph(POLARDB_CONFIG)


def process_metadata(item):
    """Process metadata, extract and convert fields"""
    metadata = {}
    for key, value in item.items():
        if key not in ["id", "memory"]:
            # Type conversion
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
    """Extract embedding from data item"""
    embedding = None
    for embedding_key in ["embedding_1024", "embedding_768", "embedding_3072", "embedding"]:
        if embedding_key in item and item[embedding_key]:
            embedding_value = item[embedding_key]
            if isinstance(embedding_value, str):
                try:
                    embedding = json.loads(embedding_value)
                except json.JSONDecodeError:
                    print(f"⚠️ Unable to parse embedding string: {embedding_key}")
                    embedding = None
            else:
                embedding = embedding_value
            break
    return embedding


def prepare_nodes_for_insertion(data_list):
    """Prepare node data for insertion"""
    nodes_to_insert = []
    processed_count = 0
    skipped_count = 0

    for item in data_list:
        id_ = item.get("id")
        memory_ = item.get("memory")

        if not id_ or not memory_:
            print(f"⚠️ Skipping invalid data: ID or memory is empty")
            skipped_count += 1
            continue

        # Process metadata
        metadata = process_metadata(item)

        # Handle embedding field
        embedding = extract_embedding(item)
        if embedding:
            metadata["embedding"] = embedding

        # Build data for insertion
        nodes_to_insert.append({
            "id": id_,
            "memory": memory_,
            "metadata": metadata
        })
        processed_count += 1

        # Show progress
        if processed_count % 10000 == 0:
            print(f"📝 Preprocessed {processed_count} records")

    print(f"✅ Data preprocessing complete: Valid {processed_count}, Skipped {skipped_count}")
    return nodes_to_insert


def insert_data_optimized(data_list, batch_size=1000):
    """Optimized data insertion"""
    graph = getPolarDb()

    # Data preprocessing
    print("🔄 Starting data preprocessing...")
    nodes_to_insert = prepare_nodes_for_insertion(data_list)

    if not nodes_to_insert:
        print("⚠️ No valid data to insert")
        graph.close()
        return 0, 0

    # Use optimized version, set batch size to 1000
    # Adjust batch size based on conditions:
    # - Good network: 1000-2000
    # - Average network: 500-1000
    # - Limited memory: 200-500
    success_count, error_count = graph.batch_add_nodes_optimized(nodes_to_insert, batch_size)

    graph.close()
    return success_count, error_count



def load_data_from_file(filename):
    """Load data from file"""
    print(f"📂 Loading file: {filename}")
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"📂 Loaded {len(data)} records from file {filename}")
        return data
    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        return []

def update_graph():
    print("-----------update_graph[start]")
    graph = getPolarDb()
    graph.update_graph_id_in_properties()
    print("---------update_graph[end]")

def insert_data(conn, data):
    # Record total start time
    total_start_time = datetime.now()


    if not data:
        print("⚠️ No data")
        return

    print(f"🎯 Total records to process: {len(data)}")
    success_count, error_count = insert_data_optimized(data, batch_size=1000)

    # Compute total time
    total_time = (datetime.now() - total_start_time).total_seconds()
    minutes, seconds = divmod(total_time, 60)
    hours, minutes = divmod(minutes, 60)

    print(f"\n🎉 Processing complete!")
    print(f"📊 Final results:")
    print(f"   ✅ Success: {success_count}")
    print(f"   ❌ Failures: {error_count}")
    print(f"   ⏱️  Total time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

def main():
    json_file = r"/Users/ccl/Desktop/file/export13/ceshi/ceshi.json"

    # Record total start time
    total_start_time = datetime.now()

    # Load data
    data = load_data_from_file(json_file)
    if not data:
        print("⚠️ No data")
        return

    print(f"🎯 Total records to process: {len(data)}")

    # Use optimized version, set batch size to 1000
    # Adjust batch size based on conditions:
    # - Good network: 1000-2000
    # - Average network: 500-1000
    # - Limited memory: 200-500
    success_count, error_count = insert_data_optimized(data, batch_size=1000)

    # Compute total time
    total_time = (datetime.now() - total_start_time).total_seconds()
    minutes, seconds = divmod(total_time, 60)
    hours, minutes = divmod(minutes, 60)

    print(f"\n🎉 Processing complete!")
    print(f"📊 Final results:")
    print(f"   ✅ Success: {success_count}")
    print(f"   ❌ Failures: {error_count}")
    print(f"   ⏱️  Total time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    if success_count > 0:
        records_per_second = success_count / total_time
        print(f"   🚀 Processing speed: {records_per_second:.2f} records/sec")


if __name__ == "__main__":
    main()