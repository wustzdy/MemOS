import json
import time
import random
from datetime import datetime
from typing import Any, Literal

import numpy as np
import psycopg2
from psycopg2.extras import Json

from memos.configs.graph_db import PolarDBGraphDBConfig
from memos.dependency import require_python_package
from memos.graph_dbs.base import BaseGraphDB
from memos.log import get_logger

logger = get_logger(__name__)

# 图数据库配置
GRAPH_NAME = 'test_memos_graph'


def _compose_node(item: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    node_id = item["id"]
    memory = item["memory"]
    metadata = item.get("metadata", {})
    return node_id, memory, metadata


def _prepare_node_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure metadata has proper datetime fields and normalized types.

    - Fill `created_at` and `updated_at` if missing (in ISO 8601 format).
    - Convert embedding to list of float if present.
    """
    now = datetime.utcnow().isoformat()

    # Fill timestamps if missing
    metadata.setdefault("created_at", now)
    metadata.setdefault("updated_at", now)

    # Normalize embedding type
    embedding = metadata.get("embedding")
    if embedding and isinstance(embedding, list):
        metadata["embedding"] = [float(x) for x in embedding]

    return metadata


def generate_vector(dim=1024, low=-0.2, high=0.2):
    """Generate a random vector for testing purposes."""
    return [round(random.uniform(low, high), 6) for _ in range(dim)]


def find_embedding(metadata):
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


def detect_embedding_field(embedding_list):
    if not embedding_list:
        return None
    dim = len(embedding_list)
    if dim == 1024:
        return "embedding"
    else:
        print(f"⚠️ 未知 embedding 维度 {dim}，跳过该向量")
        return None
def convert_to_vector(embedding_list):
    if not embedding_list:
        return None
    if isinstance(embedding_list, np.ndarray):
        embedding_list = embedding_list.tolist()
    return "[" + ",".join(str(float(x)) for x in embedding_list) + "]"

def clean_properties(props):
    """移除向量字段"""
    vector_keys = {"embedding", "embedding_1024", "embedding_3072", "embedding_768"}
    if not isinstance(props, dict):
        return {}
    return {k: v for k, v in props.items() if k not in vector_keys}


class PolarDBGraphDB(BaseGraphDB):
    """PolarDB-based implementation using Apache AGE graph database extension."""

    @require_python_package(
        import_name="psycopg2",
        install_command="pip install psycopg2-binary",
        install_link="https://pypi.org/project/psycopg2-binary/",
    )
    def __init__(self, config: PolarDBGraphDBConfig):
        """PolarDB-based implementation using Apache AGE.

        Tenant Modes:
        - use_multi_db = True:
            Dedicated Database Mode (Multi-Database Multi-Tenant).
            Each tenant or logical scope uses a separate PolarDB database.
            `db_name` is the specific tenant database.
            `user_name` can be None (optional).

        - use_multi_db = False:
            Shared Database Multi-Tenant Mode.
            All tenants share a single PolarDB database.
            `db_name` is the shared database.
            `user_name` is required to isolate each tenant's data at the node level.
            All node queries will enforce `user_name` in WHERE conditions and store it in metadata,
            but it will be removed automatically before returning to external consumers.
        """
        import psycopg2

        self.config = config

        # Handle both dict and object config
        if isinstance(config, dict):
            self.db_name = config.get("db_name")
            self.user_name = config.get("user_name")
            host = config.get("host")
            port = config.get("port")
            user = config.get("user")
            password = config.get("password")
        else:
            self.db_name = config.db_name
            self.user_name = config.user_name
            host = config.host
            port = config.port
            user = config.user
            password = config.password

        # Create connection
        self.connection = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=self.db_name
        )
        self.connection.autocommit = True

        # Handle auto_create
        auto_create = config.get("auto_create", False) if isinstance(config, dict) else config.auto_create
        if auto_create:
            self._ensure_database_exists()

        # Create graph and tables
        # self._create_graph()

        # Handle embedding_dimension
        embedding_dim = config.get("embedding_dimension", 1024) if isinstance(config,dict) else config.embedding_dimension
        self.create_index(dimensions=embedding_dim)

    def _get_config_value(self, key: str, default=None):
        """Safely get config value from either dict or object."""
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        else:
            return getattr(self.config, key, default)

    def _ensure_database_exists(self):
        """Create database if it doesn't exist."""
        try:
            # For PostgreSQL/PolarDB, we need to connect to a default database first
            # This is a simplified implementation - in production you might want to handle this differently
            logger.info(f"Using database '{self.db_name}'")
        except Exception as e:
            logger.error(f"Failed to access database '{self.db_name}': {e}")
            raise

    def _create_graph(self):
        """Create PostgreSQL schema and table for graph storage."""
        try:
            with self.connection.cursor() as cursor:
                # Create schema if it doesn't exist
                cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.db_name}_graph";')
                logger.info(f"Schema '{self.db_name}_graph' ensured.")

                # Create Memory table if it doesn't exist
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS "{self.db_name}_graph"."Memory" (
                        id TEXT PRIMARY KEY,
                        properties JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                logger.info(f"Memory table created in schema '{self.db_name}_graph'.")

                # Add embedding column if it doesn't exist (using JSONB for compatibility)
                try:
                    cursor.execute(f"""
                        ALTER TABLE "{self.db_name}_graph"."Memory" 
                        ADD COLUMN IF NOT EXISTS embedding JSONB;
                    """)
                    logger.info(f"Embedding column added to Memory table.")
                except Exception as e:
                    logger.warning(f"Failed to add embedding column: {e}")

                # Create indexes
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_memory_properties 
                    ON "{self.db_name}_graph"."Memory" USING GIN (properties);
                """)

                # Create vector index for embedding field
                try:
                    cursor.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_memory_embedding 
                        ON "{self.db_name}_graph"."Memory" USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    """)
                    logger.info(f"Vector index created for Memory table.")
                except Exception as e:
                    logger.warning(f"Vector index creation failed (might not be supported): {e}")

                logger.info(f"Indexes created for Memory table.")

        except Exception as e:
            logger.error(f"Failed to create graph schema: {e}")
            raise e

    def create_index(
            self,
            label: str = "Memory",
            vector_property: str = "embedding",
            dimensions: int = 1024,
            index_name: str = "memory_vector_index",
    ) -> None:
        """
        Create indexes for embedding and other fields.
        Note: This creates PostgreSQL indexes on the underlying tables.
        """
        try:
            with self.connection.cursor() as cursor:
                # Create indexes on the underlying PostgreSQL tables
                # Apache AGE stores data in regular PostgreSQL tables
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_memory_properties 
                    ON "{self.db_name}_graph"."Memory" USING GIN (properties);
                """)

                # Try to create vector index, but don't fail if it doesn't work
                try:
                    cursor.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_memory_embedding 
                        ON "{self.db_name}_graph"."Memory" USING ivfflat (embedding vector_cosine_ops);
                    """)
                except Exception as ve:
                    logger.warning(f"Vector index creation failed (might not be supported): {ve}")

                logger.debug(f"Indexes created successfully.")
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")

    def get_memory_count(self, memory_type: str, user_name: str | None = None) -> int:
        """Get count of memory nodes by type."""
        user_name = user_name if user_name else self._get_config_value("user_name")
        query = f"""
            SELECT COUNT(*) 
            FROM "{self.db_name}_graph"."Memory" 
            WHERE ag_catalog.agtype_access_operator(properties, '"memory_type"'::agtype) = %s::agtype
        """
        query += "\nAND ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = %s::agtype"
        params = [f'"{memory_type}"', f'"{user_name}"']

        print(f"[get_memory_count] Query: {query}, Params: {params}")

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.error(f"[get_memory_count] Failed: {e}")
            return -1

    def node_not_exist(self, scope: str, user_name: str | None = None) -> int:
        """Check if a node with given scope exists."""
        user_name = user_name if user_name else self._get_config_value("user_name")
        query = f"""
            SELECT id 
            FROM "{self.db_name}_graph"."Memory" 
            WHERE ag_catalog.agtype_access_operator(properties, '"memory_type"'::agtype) = %s::agtype
        """
        query += "\nAND ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = %s::agtype"
        query += "\nLIMIT 1"
        params = [f'"{scope}"', f'"{user_name}"']

        print(f"[node_not_exist] Query: {query}, Params: {params}")

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                result = cursor.fetchone()
                print(f"[node_not_exist] Query result: {result}")
                return len(result)
        except Exception as e:
            logger.error(f"[node_not_exist] Query failed: {e}", exc_info=True)
            raise

    def remove_oldest_memory(self, memory_type: str, keep_latest: int, user_name: str | None = None) -> None:
        """
        Remove all WorkingMemory nodes except the latest `keep_latest` entries.

        Args:
            memory_type (str): Memory type (e.g., 'WorkingMemory', 'LongTermMemory').
            keep_latest (int): Number of latest WorkingMemory entries to keep.
            user_name (str, optional): User name for filtering in non-multi-db mode
        """
        user_name = user_name if user_name else self._get_config_value("user_name")
        
        # 使用真正的 OFFSET 逻辑，与 nebular.py 保持一致
        # 先找到要删除的节点ID，然后删除它们
        select_query = f"""
            SELECT id FROM "{self.db_name}_graph"."Memory" 
            WHERE ag_catalog.agtype_access_operator(properties, '"memory_type"'::agtype) = %s::agtype
            AND ag_catalog.agtype_access_operator(properties, '"user_name"'::agtype) = %s::agtype
            ORDER BY ag_catalog.agtype_access_operator(properties, '"updated_at"'::agtype) DESC 
            OFFSET %s
        """
        select_params = [f'"{memory_type}"', f'"{user_name}"', keep_latest]
        print(f"[remove_oldest_memory] Select query: {select_query}")
        print(f"[remove_oldest_memory] Select params: {select_params}")
        
        try:
            with self.connection.cursor() as cursor:
                # 执行查询获取要删除的ID列表
                cursor.execute(select_query, select_params)
                ids_to_delete = [row[0] for row in cursor.fetchall()]

                if not ids_to_delete:
                    logger.info(f"No {memory_type} memories to remove for user {user_name}")
                    return

                # 构建删除查询
                placeholders = ','.join(['%s'] * len(ids_to_delete))
                delete_query = f"""
                    DELETE FROM "{self.db_name}_graph"."Memory"
                    WHERE id IN ({placeholders})
                """
                delete_params = ids_to_delete

                # 执行删除
                cursor.execute(delete_query, delete_params)
                deleted_count = cursor.rowcount
                logger.info(f"Removed {deleted_count} oldest {memory_type} memories, keeping {keep_latest} latest for user {user_name}")
        except Exception as e:
            logger.error(f"[remove_oldest_memory] Failed: {e}", exc_info=True)
            raise

    def update_node(self, id: str, fields: dict[str, Any], user_name: str | None = None) -> None:
        """
        Update node fields in PolarDB, auto-converting `created_at` and `updated_at` to datetime type if present.
        """
        if not fields:
            return

        # 获取当前节点
        current_node = self.get_node(id, user_name=user_name)
        if not current_node:
            return

        # 更新属性，但保留原始的id字段和memory字段
        properties = current_node["metadata"].copy()
        original_id = properties.get("id", id)  # 保留原始ID
        original_memory = current_node.get("memory", "")  # 保留原始memory
        
        # 如果fields中有memory字段，使用它；否则保留原始的memory
        if "memory" in fields:
            original_memory = fields.pop("memory")
        
        properties.update(fields)
        properties["id"] = original_id  # 确保ID不被覆盖
        properties["memory"] = original_memory  # 确保memory不被覆盖

        # 处理 embedding 字段
        embedding_vector = None
        if "embedding" in fields:
            embedding_vector = fields.pop("embedding")
            if not isinstance(embedding_vector, list):
                embedding_vector = None

        # 构建更新查询
        if embedding_vector is not None:
            query = f"""
                UPDATE "{self.db_name}_graph"."Memory" 
                SET properties = %s, embedding = %s
                WHERE ag_catalog.agtype_access_operator(properties, '"id"'::agtype) = %s::agtype
            """
            params = [json.dumps(properties), json.dumps(embedding_vector), f'"{id}"']
        else:
            query = f"""
                UPDATE "{self.db_name}_graph"."Memory" 
                SET properties = %s
                WHERE ag_catalog.agtype_access_operator(properties, '"id"'::agtype) = %s::agtype
            """
            params = [json.dumps(properties), f'"{id}"']

        # 只有在提供了 user_name 参数时才添加用户过滤
        if user_name is not None:
            query += "\nAND ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = %s::agtype"
            params.append(f'"{user_name}"')

        print(f"[update_node] query: {query}, params: {params}")
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
        except Exception as e:
            logger.error(f"[update_node] Failed to update node '{id}': {e}", exc_info=True)
            raise

    def delete_node(self, id: str, user_name: str | None = None) -> None:
        """
        Delete a node from the graph.
        Args:
            id: Node identifier to delete.
            user_name (str, optional): User name for filtering in non-multi-db mode
        """
        query = f"""
            DELETE FROM "{self.db_name}_graph"."Memory" 
            WHERE ag_catalog.agtype_access_operator(properties, '"id"'::agtype) = %s::agtype
        """
        params = [f'"{id}"']

        # 只有在提供了 user_name 参数时才添加用户过滤
        if user_name is not None:
            query += "\nAND ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = %s::agtype"
            params.append(f'"{user_name}"')

        print(f"[delete_node] query: {query}, params: {params}")
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
        except Exception as e:
            logger.error(f"[delete_node] Failed to delete node '{id}': {e}", exc_info=True)
            raise

    def create_extension(self):
        extensions = [
            ("polar_age", "图引擎"),
            ("vector", "向量引擎")
        ]
        try:
            with self.connection.cursor() as cursor:
                # 确保在正确的数据库上下文中
                cursor.execute(f"SELECT current_database();")
                current_db = cursor.fetchone()[0]
                print(f"当前数据库上下文: {current_db}")
                
                for ext_name, ext_desc in extensions:
                    try:
                        cursor.execute(f"create extension if not exists {ext_name};")
                        print(f"✅ Extension '{ext_name}' ({ext_desc}) ensured.")
                    except Exception as e:
                        if "already exists" in str(e):
                            print(f"ℹ️ Extension '{ext_name}' ({ext_desc}) already exists.")
                        else:
                            print(f"⚠️ Failed to create extension '{ext_name}' ({ext_desc}): {e}")
                            logger.error(f"Failed to create extension '{ext_name}': {e}", exc_info=True)
        except Exception as e:
            print(f"⚠️ Failed to access database context: {e}")
            logger.error(f"Failed to access database context: {e}", exc_info=True)

    def create_graph(self):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"""
                    SELECT COUNT(*) FROM ag_catalog.ag_graph 
                    WHERE name = '{self.db_name}';
                """)
                graph_exists = cursor.fetchone()[0] > 0
                
                if graph_exists:
                    print(f"ℹ️ Graph '{self.db_name}' already exists.")
                else:
                    cursor.execute(f"select create_graph('{self.db_name}');")
                    print(f"✅ Graph database '{self.db_name}' created.")
        except Exception as e:
            print(f"⚠️ Failed to create graph '{self.db_name}': {e}")
            logger.error(f"Failed to create graph '{self.db_name}': {e}", exc_info=True)

    def create_edge(self):
        """创建所有有效的边类型，如果不存在的话"""
        VALID_REL_TYPES = {
            "AGGREGATE_TO",
            "FOLLOWS",
            "INFERS",
            "MERGED_TO",
            "RELATE_TO",
            "PARENT"
        }
        
        for label_name in VALID_REL_TYPES:
            print(f"🪶 Creating elabel: {label_name}")
            try:
                with self.connection.cursor() as cursor:
                    cursor.execute(f"select create_elabel('{self.db_name}', '{label_name}');")
                    print(f"✅ Successfully created elabel: {label_name}")
            except Exception as e:
                if "already exists" in str(e):
                    print(f"ℹ️ Label '{label_name}' already exists, skipping.")
                else:
                    print(f"⚠️ Failed to create label {label_name}: {e}")
                    logger.error(f"Failed to create elabel '{label_name}': {e}", exc_info=True)

    def add_edge(self, source_id: str, target_id: str, type: str, user_name: str | None = None) -> None:
        if not source_id or not target_id:
            raise ValueError("[add_edge] source_id and target_id must be provided")

        source_exists = self.get_node(source_id) is not None
        target_exists = self.get_node(target_id) is not None

        if not source_exists or not target_exists:
            raise ValueError("[add_edge] source_id and target_id must be provided")

        properties = {}
        if user_name is not None:
            properties["user_name"] = user_name
        query = f"""
            INSERT INTO "{self.db_name}_graph"."{type}"(id, start_id, end_id, properties)
            SELECT
                ag_catalog._next_graph_id('"{self.db_name}_graph"'::name, '{type}'),
                ag_catalog._make_graph_id('"{self.db_name}_graph"'::name, 'Memory'::name, '{source_id}'::text::cstring),
                ag_catalog._make_graph_id('"{self.db_name}_graph"'::name, 'Memory'::name, '{target_id}'::text::cstring),
                jsonb_build_object('user_name', '{user_name}')::text::agtype
            WHERE NOT EXISTS (
                SELECT 1 FROM "{self.db_name}_graph"."{type}"
                WHERE start_id = ag_catalog._make_graph_id('"{self.db_name}_graph"'::name, 'Memory'::name, '{source_id}'::text::cstring)
                  AND end_id   = ag_catalog._make_graph_id('"{self.db_name}_graph"'::name, 'Memory'::name, '{target_id}'::text::cstring)
            );
        """
        print(f"Executing add_edge: {query}")

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, (source_id, target_id, type, json.dumps(properties)))
                logger.info(f"Edge created: {source_id} -[{type}]-> {target_id}")
        except Exception as e:
            logger.error(f"Failed to insert edge: {e}", exc_info=True)
            raise

    def delete_edge(self, source_id: str, target_id: str, type: str) -> None:
        """
        Delete a specific edge between two nodes.
        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            type: Relationship type to remove.
        """
        query = f"""
            DELETE FROM "{self.db_name}_graph"."Edges"
            WHERE source_id = %s AND target_id = %s AND edge_type = %s
        """

        with self.connection.cursor() as cursor:
            cursor.execute(query, (source_id, target_id, type))
            logger.info(f"Edge deleted: {source_id} -[{type}]-> {target_id}")

    def edge_exists_old(
            self, source_id: str, target_id: str, type: str = "ANY", direction: str = "OUTGOING"
    ) -> bool:
        """
        Check if an edge exists between two nodes.
        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            type: Relationship type. Use "ANY" to match any relationship type.
            direction: Direction of the edge.
                       Use "OUTGOING" (default), "INCOMING", or "ANY".
        Returns:
            True if the edge exists, otherwise False.
        """
        where_clauses = []
        params = []
        # SELECT * FROM
        # cypher('memtensor_memos_graph', $$
        # MATCH(a: Memory
        # {id: "13bb9df6-0609-4442-8bed-bba77dadac92"})-[r] - (b:Memory {id: "2dd03a5b-5d5f-49c9-9e0a-9a2a2899b98d"})
        # RETURN
        # r
        # $$) AS(r
        # agtype);

        if direction == "OUTGOING":
            where_clauses.append("source_id = %s AND target_id = %s")
            params.extend([source_id, target_id])
        elif direction == "INCOMING":
            where_clauses.append("source_id = %s AND target_id = %s")
            params.extend([target_id, source_id])
        elif direction == "ANY":
            where_clauses.append("((source_id = %s AND target_id = %s) OR (source_id = %s AND target_id = %s))")
            params.extend([source_id, target_id, target_id, source_id])
        else:
            raise ValueError(
                f"Invalid direction: {direction}. Must be 'OUTGOING', 'INCOMING', or 'ANY'."
            )

        if type != "ANY":
            where_clauses.append("edge_type = %s")
            params.append(type)

        where_clause = " AND ".join(where_clauses)

        query = f"""
            SELECT 1 FROM "{self.db_name}_graph"."Edges"
            WHERE {where_clause}
            LIMIT 1
        """

        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
            result = cursor.fetchone()
            return result is not None

    def edge_exists(
            self,
            source_id: str,
            target_id: str,
            type: str = "ANY",
            direction: str = "OUTGOING",
            user_name: str | None = None,
    ) -> bool:
        """
        Check if an edge exists between two nodes.
        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            type: Relationship type. Use "ANY" to match any relationship type.
            direction: Direction of the edge.
                       Use "OUTGOING" (default), "INCOMING", or "ANY".
            user_name (str, optional): User name for filtering in non-multi-db mode
        Returns:
            True if the edge exists, otherwise False.
        """

        # Prepare the relationship pattern
        user_name = user_name if user_name else self.config.user_name
        print(f"edge_exists direction: {direction}")

        # Prepare the match pattern with direction
        if direction == "OUTGOING":
            pattern = f"(a:Memory)-[r]->(b:Memory)"
        elif direction == "INCOMING":
            pattern = f"(a:Memory)<-[r]-(b:Memory)"
        elif direction == "ANY":
            pattern = f"(a:Memory)-[r]-(b:Memory)"
        else:
            raise ValueError(
                f"Invalid direction: {direction}. Must be 'OUTGOING', 'INCOMING', or 'ANY'."
            )
        query = f"SELECT * FROM cypher('{self.db_name}_graph', $$"
        query += f"\nMATCH {pattern}"
        query += f"\nWHERE a.user_name = '{user_name}' AND b.user_name = '{user_name}'"
        query += f"\nAND a.id = '{source_id}' AND b.id = '{target_id}'"
        if type != "ANY":
            query += f"\n AND type(r) = '{type}'"

        query += "\nRETURN r"
        query += "\n$$) AS (r agtype)"

        print(f"edge_exists query: {query}")
        with self.connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchone()
            return result is not None and result[0] is not None

    def get_node(self, id: str, include_embedding: bool = False, user_name: str | None = None) -> dict[str, Any] | None:
        """
        Retrieve a Memory node by its unique ID.

        Args:
            id (str): Node ID (Memory.id)
            include_embedding: with/without embedding
            user_name (str, optional): User name for filtering in non-multi-db mode

        Returns:
            dict: Node properties as key-value pairs, or None if not found.
        """
        # 构建查询字段
        if include_embedding:
            select_fields = "id, properties, embedding"
        else:
            select_fields = "id, properties"
            
        query = f"""
            SELECT {select_fields}
            FROM "{self.db_name}_graph"."Memory" 
            WHERE ag_catalog.agtype_access_operator(properties, '"id"'::agtype) = %s::agtype
        """
        params = [f'"{id}"']
        
        # 只有在提供了 user_name 参数时才添加用户过滤
        if user_name is not None:
            query += "\nAND ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = %s::agtype"
            params.append(f'"{user_name}"')

        print(f"[get_node] query: {query}, params: {params}")
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                result = cursor.fetchone()

                if result:
                    if include_embedding:
                        node_id, properties_json, embedding_json = result
                    else:
                        node_id, properties_json = result
                        embedding_json = None
                    
                    # Parse properties from JSONB if it's a string
                    if isinstance(properties_json, str):
                        try:
                            properties = json.loads(properties_json)
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(f"Failed to parse properties for node {id}")
                            properties = {}
                    else:
                        properties = properties_json if properties_json else {}
                    
                    # Parse embedding from JSONB if it exists and include_embedding is True
                    if include_embedding and embedding_json is not None:
                        try:
                            embedding = json.loads(embedding_json) if isinstance(embedding_json, str) else embedding_json
                            properties["embedding"] = embedding
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(f"Failed to parse embedding for node {id}")
                    
                    return self._parse_node({"id": id, "memory": properties.get("memory", ""), **properties})
                return None

        except Exception as e:
            logger.error(f"[get_node] Failed to retrieve node '{id}': {e}", exc_info=True)
            return None

    def get_nodes(self, ids: list[str], **kwargs) -> list[dict[str, Any]]:
        """
        Retrieve the metadata and memory of a list of nodes.
        Args:
            ids: List of Node identifier.
        Returns:
        list[dict]: Parsed node records containing 'id', 'memory', and 'metadata'.

        Notes:
            - Assumes all provided IDs are valid and exist.
            - Returns empty list if input is empty.
        """
        if not ids:
            return []

        # Build WHERE clause using agtype_access_operator like get_node method
        where_conditions = []
        params = []
        
        for id_val in ids:
            where_conditions.append("ag_catalog.agtype_access_operator(properties, '\"id\"'::agtype) = %s::agtype")
            params.append(f'"{id_val}"')
        
        where_clause = " OR ".join(where_conditions)
        
        query = f"""
            SELECT id, properties, embedding
            FROM "{self.db_name}_graph"."Memory" 
            WHERE ({where_clause})
        """

        if not self._get_config_value("use_multi_db", True) and self._get_config_value("user_name"):
            user_name = kwargs.get("cube_name", self._get_config_value("user_name"))
            query += " AND properties::text LIKE %s"
            params.append(f"%{user_name}%")

        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()

            nodes = []
            for row in results:
                node_id, properties_json, embedding_json = row
                # Parse properties from JSONB if it's a string
                if isinstance(properties_json, str):
                    try:
                        properties = json.loads(properties_json)
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(f"Failed to parse properties for node {node_id}")
                        properties = {}
                else:
                    properties = properties_json if properties_json else {}
                
                # Parse embedding from JSONB if it exists
                if embedding_json is not None:
                    try:
                        embedding = json.loads(embedding_json) if isinstance(embedding_json, str) else embedding_json
                        properties["embedding"] = embedding
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(f"Failed to parse embedding for node {node_id}")
                nodes.append(self._parse_node(
                    {"id": properties.get("id", node_id), "memory": properties.get("memory", ""), "metadata": properties}))
            return nodes

    def get_edges(self, id: str, type: str = "ANY", direction: str = "ANY") -> list[dict[str, str]]:
        """
        Get edges connected to a node, with optional type and direction filter.

        Args:
            id: Node ID to retrieve edges for.
            type: Relationship type to match, or 'ANY' to match all.
            direction: 'OUTGOING', 'INCOMING', or 'ANY'.

        Returns:
            List of edges:
            [
              {"from": "source_id", "to": "target_id", "type": "RELATE"},
              ...
            ]
        """
        # 由于PolarDB没有完整的图数据库功能，这里使用简化的实现
        # 在实际应用中，你可能需要创建专门的边表来存储关系

        # 创建一个简单的边表来存储关系（如果不存在的话）
        try:
            with self.connection.cursor() as cursor:
                # 创建边表
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS "{self.db_name}_graph"."Edges" (
                        id SERIAL PRIMARY KEY,
                        source_id TEXT NOT NULL,
                        target_id TEXT NOT NULL,
                        edge_type TEXT NOT NULL,
                        properties JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (source_id) REFERENCES "{self.db_name}_graph"."Memory"(id),
                        FOREIGN KEY (target_id) REFERENCES "{self.db_name}_graph"."Memory"(id)
                    );
                """)

                # 创建索引
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_edges_source 
                    ON "{self.db_name}_graph"."Edges" (source_id);
                """)
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_edges_target 
                    ON "{self.db_name}_graph"."Edges" (target_id);
                """)
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_edges_type 
                    ON "{self.db_name}_graph"."Edges" (edge_type);
                """)
        except Exception as e:
            logger.warning(f"Failed to create edges table: {e}")

        # 查询边
        where_clauses = []
        params = [id]

        if type != "ANY":
            where_clauses.append("edge_type = %s")
            params.append(type)

        if direction == "OUTGOING":
            where_clauses.append("source_id = %s")
        elif direction == "INCOMING":
            where_clauses.append("target_id = %s")
        else:  # ANY
            where_clauses.append("(source_id = %s OR target_id = %s)")
            params.append(id)  # 添加第二个参数用于ANY方向

        where_clause = " AND ".join(where_clauses)

        query = f"""
            SELECT source_id, target_id, edge_type
            FROM "{self.db_name}_graph"."Edges"
            WHERE {where_clause}
        """

        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()

            edges = []
            for row in results:
                source_id, target_id, edge_type = row
                edges.append({
                    "from": source_id,
                    "to": target_id,
                    "type": edge_type
                })
            return edges

    def get_neighbors(
            self, id: str, type: str, direction: Literal["in", "out", "both"] = "out"
    ) -> list[str]:
        """Get connected node IDs in a specific direction and relationship type."""
        raise NotImplementedError

    def get_neighbors_by_tag(
            self,
            tags: list[str],
            exclude_ids: list[str],
            top_k: int = 5,
            min_overlap: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Find top-K neighbor nodes with maximum tag overlap.

        Args:
            tags: The list of tags to match.
            exclude_ids: Node IDs to exclude (e.g., local cluster).
            top_k: Max number of neighbors to return.
            min_overlap: Minimum number of overlapping tags required.

        Returns:
            List of dicts with node details and overlap count.
        """
        # 构建查询条件
        where_clauses = []
        params = []

        # 排除指定的ID
        if exclude_ids:
            placeholders = ','.join(['%s'] * len(exclude_ids))
            where_clauses.append(f"id NOT IN ({placeholders})")
            params.extend(exclude_ids)

        # 状态过滤
        where_clauses.append("properties->>'status' = %s")
        params.append('activated')

        # 类型过滤
        where_clauses.append("properties->>'type' != %s")
        params.append('reasoning')

        where_clauses.append("properties->>'memory_type' != %s")
        params.append('WorkingMemory')

        # 用户过滤
        if not self._get_config_value("use_multi_db", True) and self._get_config_value("user_name"):
            where_clauses.append("properties->>'user_name' = %s")
            params.append(self._get_config_value("user_name"))

        where_clause = " AND ".join(where_clauses)

        # 获取所有候选节点
        query = f"""
            SELECT id, properties, embedding
            FROM "{self.db_name}_graph"."Memory" 
            WHERE {where_clause}
        """

        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()

            nodes_with_overlap = []
            for row in results:
                node_id, properties_json, embedding_json = row
                properties = properties_json if properties_json else {}

                # 解析embedding
                if embedding_json is not None:
                    try:
                        embedding = json.loads(embedding_json) if isinstance(embedding_json, str) else embedding_json
                        properties["embedding"] = embedding
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(f"Failed to parse embedding for node {node_id}")

                # 计算标签重叠
                node_tags = properties.get("tags", [])
                if isinstance(node_tags, str):
                    try:
                        node_tags = json.loads(node_tags)
                    except (json.JSONDecodeError, TypeError):
                        node_tags = []

                overlap_tags = [tag for tag in tags if tag in node_tags]
                overlap_count = len(overlap_tags)

                if overlap_count >= min_overlap:
                    node_data = self._parse_node({
                        "id": properties.get("id", node_id),
                        "memory": properties.get("memory", ""),
                        "metadata": properties
                    })
                    nodes_with_overlap.append((node_data, overlap_count))

            # 按重叠数量排序并返回前top_k个
            nodes_with_overlap.sort(key=lambda x: x[1], reverse=True)
            return [node for node, _ in nodes_with_overlap[:top_k]]

    def get_children_with_embeddings(self, id: str, user_name: str | None = None) -> list[dict[str, Any]]:
        """Get children nodes with their embeddings."""
        user_name = user_name if user_name else self._get_config_value("user_name")
        where_user = f"AND p.user_name = '{user_name}' AND c.user_name = '{user_name}'"

        query = f"""
            WITH t as (
                SELECT *
                FROM cypher('{self.db_name}_graph', $$
                MATCH (p:Memory)-[r:PARENT]->(c:Memory)
                WHERE p.id = '{id}' {where_user} 
                RETURN id(c) as cid, c.id AS id, c.memory AS memory
                $$) as (cid agtype, id agtype, memory agtype)
                )
                SELECT t.id, m.embedding, t.memory FROM t,
                "{self.db_name}_graph"."Memory" m
            WHERE t.cid::graphid = m.id;
        """


        print("[get_children_with_embeddings] query:", query)

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()

                children = []
                for row in results:
                    # 处理 child_id - 移除可能的引号
                    child_id_raw = row[0].value if hasattr(row[0], 'value') else str(row[0])
                    if isinstance(child_id_raw, str):
                        # 如果字符串以引号开始和结束，去掉引号
                        if child_id_raw.startswith('"') and child_id_raw.endswith('"'):
                            child_id = child_id_raw[1:-1]
                        else:
                            child_id = child_id_raw
                    else:
                        child_id = str(child_id_raw)

                    # 处理 embedding - 从数据库的embedding列获取
                    embedding_raw = row[1]
                    embedding = []
                    if embedding_raw is not None:
                        try:
                            if isinstance(embedding_raw, str):
                                # 如果是JSON字符串，解析它
                                embedding = json.loads(embedding_raw)
                            elif isinstance(embedding_raw, list):
                                # 如果已经是列表，直接使用
                                embedding = embedding_raw
                            else:
                                # 尝试转换为列表
                                embedding = list(embedding_raw)
                        except (json.JSONDecodeError, TypeError, ValueError) as e:
                            logger.warning(f"Failed to parse embedding for child node {child_id}: {e}")
                            embedding = []

                    # 处理 memory - 移除可能的引号
                    memory_raw = row[2].value if hasattr(row[2], 'value') else str(row[2])
                    if isinstance(memory_raw, str):
                        # 如果字符串以引号开始和结束，去掉引号
                        if memory_raw.startswith('"') and memory_raw.endswith('"'):
                            memory = memory_raw[1:-1]
                        else:
                            memory = memory_raw
                    else:
                        memory = str(memory_raw)

                    children.append({
                        "id": child_id,
                        "embedding": embedding,
                        "memory": memory
                    })

                return children

        except Exception as e:
            logger.error(f"[get_children_with_embeddings] Failed: {e}", exc_info=True)
            return []

    def get_path(self, source_id: str, target_id: str, max_depth: int = 3) -> list[str]:
        """Get the path of nodes from source to target within a limited depth."""
        raise NotImplementedError

    def get_subgraph(
        self,
        center_id: str,
        depth: int = 2,
        center_status: str = "activated",
        user_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve a local subgraph centered at a given node.
        Args:
            center_id: The ID of the center node.
            depth: The hop distance for neighbors.
            center_status: Required status for center node.
            user_name (str, optional): User name for filtering in non-multi-db mode
        Returns:
            {
                "core_node": {...},
                "neighbors": [...],
                "edges": [...]
            }
        """
        if not 1 <= depth <= 5:
            raise ValueError("depth must be 1-5")

        user_name = user_name if user_name else self._get_config_value("user_name")

        # 使用简化的查询获取子图（暂时只获取直接邻居）
        query = f"""
            WITH center AS (
                SELECT * FROM cypher('{self.db_name}_graph', $$
                MATCH (center:Memory)
                WHERE center.id = '{center_id}'
                  AND center.status = '{center_status}'
                  AND center.user_name = '{user_name}'
                RETURN center
                $$) AS (center agtype)
            ),
            neighbors AS (
                SELECT * FROM cypher('{self.db_name}_graph', $$
                MATCH (center:Memory)-[e]->(neighbor:Memory)
                WHERE center.id = '{center_id}'
                  AND center.status = '{center_status}'
                  AND center.user_name = '{user_name}'
                  AND neighbor.user_name = '{user_name}'
                RETURN neighbor, e
                $$) AS (neighbor agtype, e agtype)
            )
            SELECT 
                (SELECT center FROM center) as center,
                ARRAY_AGG(neighbor) as neighbors,
                ARRAY_AGG(e) as edges
            FROM neighbors
        """

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchone()
                
                if not result or not result[0]:
                    return {"core_node": None, "neighbors": [], "edges": []}

                # 解析中心节点
                center_agtype = result[0]
                if center_agtype and hasattr(center_agtype, 'value'):
                    center_props = center_agtype.value
                    core_node = self._parse_node(center_props)
                else:
                    return {"core_node": None, "neighbors": [], "edges": []}

                # 解析邻居节点
                neighbors = []
                neighbors_agtype = result[1] if result[1] else []
                for neighbor_agtype in neighbors_agtype:
                    if neighbor_agtype and hasattr(neighbor_agtype, 'value'):
                        neighbor_props = neighbor_agtype.value
                        neighbor_parsed = self._parse_node(neighbor_props)
                        neighbors.append(neighbor_parsed)

                # 解析边
                edges = []
                edges_agtype = result[2] if result[2] else []
                for edge_agtype in edges_agtype:
                    if edge_agtype and hasattr(edge_agtype, 'value'):
                        edge_data = edge_agtype.value
                        if isinstance(edge_data, dict):
                            edges.append({
                                "type": edge_data.get("type", ""),
                                "source": edge_data.get("source", ""),
                                "target": edge_data.get("target", "")
                            })

                return {"core_node": core_node, "neighbors": neighbors, "edges": edges}

        except Exception as e:
            logger.error(f"Failed to get subgraph: {e}", exc_info=True)
            return {"core_node": None, "neighbors": [], "edges": []}

    def get_context_chain(self, id: str, type: str = "FOLLOWS") -> list[str]:
        """Get the ordered context chain starting from a node."""
        raise NotImplementedError

    def search_by_embedding(
            self,
            vector: list[float],
            top_k: int = 5,
            scope: str | None = None,
            status: str | None = None,
            threshold: float | None = None,
            search_filter: dict | None = None,
            **kwargs,
    ) -> list[dict]:
        """
        Retrieve node IDs based on vector similarity using PostgreSQL vector operations.
        """
        # Build WHERE clause dynamically like nebular.py
        where_clauses = []
        if scope:
            where_clauses.append(f"ag_catalog.agtype_access_operator(properties, '\"memory_type\"'::agtype) = '\"{scope}\"'::agtype")
        if status:
            where_clauses.append(f"ag_catalog.agtype_access_operator(properties, '\"status\"'::agtype) = '\"{status}\"'::agtype")
        else:
            where_clauses.append("ag_catalog.agtype_access_operator(properties, '\"status\"'::agtype) = '\"activated\"'::agtype")
        where_clauses.append("embedding is not null")
        # Add user_name filter like nebular.py
        user_name = self._get_config_value("user_name")
        if not self.config.use_multi_db and user_name:
            if kwargs.get("cube_name"):
                where_clauses.append(f"ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = '\"{kwargs['cube_name']}\"'::agtype")
            else:
                where_clauses.append(f"ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = '\"{user_name}\"'::agtype")
        
        # Add search_filter conditions like nebular.py
        if search_filter:
            for key, value in search_filter.items():
                if isinstance(value, str):
                    where_clauses.append(f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) = '\"{value}\"'::agtype")
                else:
                    where_clauses.append(f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) = {value}::agtype")
        
        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        
        # Keep original simple query structure but add dynamic WHERE clause
        query = f"""
                    WITH t AS (
                        SELECT id,
                               properties,
                               timeline,
                               embedding,
                               ag_catalog.agtype_access_operator(properties, '"id"'::agtype) AS old_id,
                               (1 - (embedding <=> %s::vector(1024))) AS scope
                        FROM "{self.db_name}_graph"."Memory"
                        {where_clause}
                        ORDER BY scope DESC
                        LIMIT {top_k}
                    )
                    SELECT *
                    FROM t
                    WHERE scope > 0.1;
                """
        params = [vector]
        print(where_clause)
        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()
            output = []
            for row in results:
                polarId = row[0]  # id
                properties = row[1]  # properties
                embedding = row[3]  # embedding
                oldId = row[4]  # old_id
                score = row[5]  # scope
                id_val = str(oldId)
                score_val = float(score)
                score_val = (score_val + 1) / 2  # align to neo4j, Normalized Cosine Score
                if threshold is None or score_val >= threshold:
                    output.append({"id": id_val, "score": score_val})
            return output[:top_k]

    def get_by_metadata(self, filters: list[dict[str, Any]]) -> list[str]:
        """Retrieve node IDs that match given metadata filters."""
        where_clauses = []
        params = []

        for i, f in enumerate(filters):
            field = f["field"]
            op = f.get("op", "=")
            value = f["value"]

            if op == "=":
                where_clauses.append(f"properties->>'{field}' = %s")
                params.append(value)
            elif op == "in":
                placeholders = ','.join(['%s'] * len(value))
                where_clauses.append(f"properties->>'{field}' IN ({placeholders})")
                params.extend(value)
            elif op == "contains":
                where_clauses.append(f"properties->'{field}' ? %s")
                params.append(value)
            elif op == "starts_with":
                where_clauses.append(f"properties->>'{field}' LIKE %s")
                params.append(f"{value}%")
            elif op == "ends_with":
                where_clauses.append(f"properties->>'{field}' LIKE %s")
                params.append(f"%{value}")
            elif op in [">", ">=", "<", "<="]:
                where_clauses.append(f"(properties->>'{field}')::numeric {op} %s")
                params.append(value)
            else:
                raise ValueError(f"Unsupported operator: {op}")

        if not self._get_config_value("use_multi_db", True) and self._get_config_value("user_name"):
            where_clauses.append("properties->>'user_name' = %s")
            params.append(self._get_config_value("user_name"))

        where_str = " AND ".join(where_clauses)
        query = f"SELECT properties->>'id' as id FROM \"{self.db_name}_graph\".\"Memory\" WHERE {where_str}"

        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()
            return [row[0] for row in results if row[0]]

    def get_grouped_counts(
        self,
        group_fields: list[str],
        where_clause: str = "",
        params: dict[str, Any] | None = None,
        user_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Count nodes grouped by any fields.

        Args:
            group_fields (list[str]): Fields to group by, e.g., ["memory_type", "status"]
            where_clause (str, optional): Extra WHERE condition. E.g.,
            "WHERE n.status = 'activated'"
            params (dict, optional): Parameters for WHERE clause.
            user_name (str, optional): User name for filtering in non-multi-db mode

        Returns:
            list[dict]: e.g., [{ 'memory_type': 'WorkingMemory', 'status': 'active', 'count': 10 }, ...]
        """
        if not group_fields:
            raise ValueError("group_fields cannot be empty")
        
        user_name = user_name if user_name else self._get_config_value("user_name")
        
        # Build user clause
        user_clause = f"ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = '\"{user_name}\"'::agtype"
        if where_clause:
            where_clause = where_clause.strip()
            if where_clause.upper().startswith("WHERE"):
                where_clause += f" AND {user_clause}"
            else:
                where_clause = f"WHERE {where_clause} AND {user_clause}"
        else:
            where_clause = f"WHERE {user_clause}"

        # Inline parameters if provided
        if params and isinstance(params, dict):
            for key, value in params.items():
                # Handle different value types appropriately
                if isinstance(value, str):
                    value = f"'{value}'"
                where_clause = where_clause.replace(f"${key}", str(value))

        # Build return fields and group by fields
        return_fields = []
        group_by_fields = []

        for field in group_fields:
            alias = field.replace(".", "_")
            return_fields.append(f"ag_catalog.agtype_access_operator(properties, '\"{field}\"'::agtype) AS {alias}")
            group_by_fields.append(alias)

        # Full SQL query construction
        query = f"""
            SELECT {", ".join(return_fields)}, COUNT(*) AS count
            FROM "{self.db_name}_graph"."Memory"
            {where_clause}
            GROUP BY {", ".join(group_by_fields)}
        """

        try:
            with self.connection.cursor() as cursor:
                # 处理参数化查询
                if params and isinstance(params, list):
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                results = cursor.fetchall()

                output = []
                for row in results:
                    group_values = {}
                    for i, field in enumerate(group_fields):
                        value = row[i]
                        if hasattr(value, 'value'):
                            group_values[field] = value.value
                        else:
                            group_values[field] = str(value)
                    count_value = row[-1]  # Last column is count
                    output.append({**group_values, "count": count_value})

                return output

        except Exception as e:
            logger.error(f"Failed to get grouped counts: {e}", exc_info=True)
            return []

    def deduplicate_nodes(self) -> None:
        """Deduplicate redundant or semantically similar nodes."""
        raise NotImplementedError

    def detect_conflicts(self) -> list[tuple[str, str]]:
        """Detect conflicting nodes based on logical or semantic inconsistency."""
        raise NotImplementedError

    def merge_nodes(self, id1: str, id2: str) -> str:
        """Merge two similar or duplicate nodes into one."""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear the entire graph."""
        try:
            with self.connection.cursor() as cursor:
                # First check if the graph exists
                cursor.execute(f"""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_schema = '"{self.db_name}_graph"' 
                        AND table_name = 'Memory'
                    )
                """)
                graph_exists = cursor.fetchone()[0]

                if not graph_exists:
                    logger.info(f"Graph '{self.db_name}_graph' does not exist, nothing to clear.")
                    return

                if not self._get_config_value("use_multi_db", True) and self._get_config_value("user_name"):
                    cursor.execute(f"""
                        DELETE FROM "{self.db_name}_graph"."Memory" 
                        WHERE properties::text LIKE %s
                    """, (f"%{self._get_config_value('user_name')}%",))
                else:
                    cursor.execute(f'DELETE FROM "{self.db_name}_graph"."Memory"')

                logger.info(f"Cleared all nodes from graph '{self.db_name}_graph'.")
        except Exception as e:
            logger.warning(f"Failed to clear graph '{self.db_name}_graph': {e}")
            # Don't raise the exception, just log it as a warning

    def export_graph(self, **kwargs) -> dict[str, Any]:
        """Export all graph nodes and edges in a structured form."""
        with self.connection.cursor() as cursor:
            # Export nodes
            node_query = f'SELECT id, properties FROM "{self.db_name}_graph"."Memory"'
            params = []

            if not self._get_config_value("use_multi_db", True) and self._get_config_value("user_name"):
                user_name = self._get_config_value("user_name")
                node_query += f" WHERE properties::text LIKE '%{user_name}%'"

            cursor.execute(node_query)
            node_results = cursor.fetchall()
            nodes = []
            for row in node_results:
                node_id, properties_json = row
                # properties_json is already a dict from psycopg2
                properties = properties_json if properties_json else {}
                nodes.append(self._parse_node(
                    {"id": properties.get("id", ""), "memory": properties.get("memory", ""), "metadata": properties}))

            # Export edges (simplified - would need more complex Cypher query for full edge export)
            edges = []

            return {"nodes": nodes, "edges": edges}

    def import_graph(self, data: dict[str, Any]) -> None:
        """Import the entire graph from a serialized dictionary."""
        with self.connection.cursor() as cursor:
            for node in data.get("nodes", []):
                id, memory, metadata = _compose_node(node)

                if not self._get_config_value("use_multi_db", True) and self._get_config_value("user_name"):
                    metadata["user_name"] = self._get_config_value("user_name")

                metadata = _prepare_node_metadata(metadata)

                # Generate embedding if not provided
                if "embedding" not in metadata or not metadata["embedding"]:
                    metadata["embedding"] = generate_vector(self._get_config_value("embedding_dimension", 1024))

                self.add_node(id, memory, metadata)

            # Import edges
            for edge in data.get("edges", []):
                self.add_edge(edge["source"], edge["target"], edge["type"])

    def get_all_memory_items(
        self, scope: str, include_embedding: bool = False, user_name: str | None = None
    ) -> list[dict]:
        """
        Retrieve all memory items of a specific memory_type.

        Args:
            scope (str): Must be one of 'WorkingMemory', 'LongTermMemory', or 'UserMemory'.
            include_embedding: with/without embedding
            user_name (str, optional): User name for filtering in non-multi-db mode

        Returns:
            list[dict]: Full list of memory items under this scope.
        """
        user_name = user_name if user_name else self._get_config_value("user_name")
        if scope not in {"WorkingMemory", "LongTermMemory", "UserMemory", "OuterMemory"}:
            raise ValueError(f"Unsupported memory type scope: {scope}")

        where_clause = f"ag_catalog.agtype_access_operator(properties, '\"memory_type\"'::agtype) = '\"{scope}\"'::agtype"
        where_clause += f" AND ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = '\"{user_name}\"'::agtype"

        # Build return fields
        if include_embedding:
            return_fields = "id, properties, embedding"
        else:
            return_fields = "id, properties"

        query = f"""
            SELECT {return_fields}
            FROM "{self.db_name}_graph"."Memory"
            WHERE {where_clause}
            LIMIT 100
        """

        nodes = []
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
                
                for row in results:
                    if include_embedding:
                        node_id, properties_json, embedding_json = row
                    else:
                        node_id, properties_json = row
                        embedding_json = None
                    
                    # Parse properties from JSONB if it's a string
                    if isinstance(properties_json, str):
                        try:
                            properties = json.loads(properties_json)
                        except json.JSONDecodeError:
                            properties = {}
                    else:
                        properties = properties_json if properties_json else {}
                    
                    # Build node data
                    node_data = {
                        "id": properties.get("id", node_id),
                        "memory": properties.get("memory", ""),
                        "metadata": properties
                    }
                    
                    if include_embedding and embedding_json is not None:
                        node_data["embedding"] = embedding_json
                    
                    nodes.append(self._parse_node(node_data))
                    
        except Exception as e:
            logger.error(f"Failed to get memories: {e}", exc_info=True)
            
        return nodes

    def get_structure_optimization_candidates(self, scope: str, **kwargs) -> list[dict]:
        """Find nodes that are likely candidates for structure optimization."""
        # This would require more complex graph traversal queries
        # For now, return nodes without parent relationships
        query = f"""
            SELECT id, properties 
            FROM "{self.db_name}_graph"."Memory" 
            WHERE properties->>'memory_type' = %s 
              AND properties->>'status' = 'activated'
        """
        params = [scope]

        if not self._get_config_value("use_multi_db", True) and self._get_config_value("user_name"):
            query += " AND properties->>'user_name' = %s"
            params.append(self._get_config_value("user_name"))

        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()

            nodes = []
            for row in results:
                node_id, properties_json = row
                # properties_json is already a dict from psycopg2
                properties = properties_json if properties_json else {}
                nodes.append(self._parse_node(
                    {"id": properties.get("id", ""), "memory": properties.get("memory", ""), "metadata": properties}))
            return nodes

    def drop_database(self) -> None:
        """Permanently delete the entire graph this instance is using."""
        if self._get_config_value("use_multi_db", True):
            with self.connection.cursor() as cursor:
                cursor.execute(f"SELECT drop_graph('{self.db_name}_graph', true)")
                print(f"Graph '{self.db_name}_graph' has been dropped.")
        else:
            raise ValueError(
                f"Refusing to drop graph '{self.db_name}_graph' in "
                f"Shared Database Multi-Tenant mode"
            )

    def _parse_node(self, node_data: dict[str, Any]) -> dict[str, Any]:
        """Parse node data from database format to standard format."""
        node = node_data.copy()

        # Convert datetime to string
        for time_field in ("created_at", "updated_at"):
            if time_field in node and hasattr(node[time_field], "isoformat"):
                node[time_field] = node[time_field].isoformat()

        # 不再对sources和usage字段进行反序列化，保持List[str]格式
        # 不再移除user_name字段，保持所有字段

        return {"id": node.pop("id"), "memory": node.pop("memory", ""), "metadata": node}

    def __del__(self):
        """Close database connection when object is destroyed."""
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()

    #deprecated
    def add_node_old(conn, id: str, memory: str, metadata: dict, graph_name=None):
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

    def add_node(self, id: str, memory: str, metadata: dict[str, Any]) -> None:
        """Add a memory node to the graph."""
        # user_name 从 metadata 中获取，如果不存在则从配置中获取
        if "user_name" not in metadata:
            if not self._get_config_value("use_multi_db", True) and self._get_config_value("user_name"):
                metadata["user_name"] = self._get_config_value("user_name")

        # Safely process metadata
        metadata = _prepare_node_metadata(metadata)

        # Merge node and set metadata
        created_at = metadata.pop("created_at", datetime.utcnow().isoformat())
        updated_at = metadata.pop("updated_at", datetime.utcnow().isoformat())

        # Prepare properties
        properties = {
            "id": id,
            "memory": memory,
            "created_at": created_at,
            "updated_at": updated_at,
            **metadata
        }

        # Generate embedding if not provided
        if "embedding" not in properties or not properties["embedding"]:
            properties["embedding"] = generate_vector(self._get_config_value("embedding_dimension", 1024))

        # serialization - 处理sources和usage字段的JSON序列化
        for field_name in ["sources", "usage"]:
            if field_name in properties and properties[field_name]:
                if isinstance(properties[field_name], list):
                    for idx in range(len(properties[field_name])):
                        # 只有当元素不是字符串时才进行序列化
                        if not isinstance(properties[field_name][idx], str):
                            properties[field_name][idx] = json.dumps(properties[field_name][idx])
                elif isinstance(properties[field_name], str):
                    # 如果已经是字符串，保持不变
                    pass

        # Extract embedding for separate column
        embedding_vector = properties.pop("embedding", [])
        if not isinstance(embedding_vector, list):
            embedding_vector = []

        # 根据embedding维度选择正确的列名
        embedding_column = "embedding"  # 默认列
        if len(embedding_vector) == 3072:
            embedding_column = "embedding_3072"
        elif len(embedding_vector) == 1024:
            embedding_column = "embedding"
        elif len(embedding_vector) == 768:
            embedding_column = "embedding_768"

        with self.connection.cursor() as cursor:
            # 先删除现有记录（如果存在）
            delete_query = f"""
                DELETE FROM "{self.db_name}_graph"."Memory" 
                WHERE id = ag_catalog._make_graph_id('"{self.db_name}_graph"'::name, 'Memory'::name, %s::text::cstring)
            """
            cursor.execute(delete_query, (id,))

            # 然后插入新记录
            if embedding_vector:
                insert_query = f"""
                    INSERT INTO "{self.db_name}_graph"."Memory"(id, properties, {embedding_column})
                    VALUES (
                        ag_catalog._make_graph_id('"{self.db_name}_graph"'::name, 'Memory'::name, %s::text::cstring),
                        %s,
                        %s
                    )
                """
                cursor.execute(insert_query, (id, json.dumps(properties), json.dumps(embedding_vector)))
            else:
                insert_query = f"""
                    INSERT INTO "{self.db_name}_graph"."Memory"(id, properties)
                    VALUES (
                        ag_catalog._make_graph_id('"{self.db_name}_graph"'::name, 'Memory'::name, %s::text::cstring),
                        %s
                    )
                """
                cursor.execute(insert_query, (id, json.dumps(properties)))
                logger.info(f"Added node {id} to graph '{self.db_name}_graph'.")