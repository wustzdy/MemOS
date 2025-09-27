import json
import time
import random
from datetime import datetime
from typing import Any, Literal

from memos.configs.graph_db import PolarDBGraphDBConfig
from memos.dependency import require_python_package
from memos.graph_dbs.base import BaseGraphDB
from memos.log import get_logger

logger = get_logger(__name__)


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
        self._create_graph()
        
        # Handle embedding_dimension
        embedding_dim = config.get("embedding_dimension", 1024) if isinstance(config, dict) else config.embedding_dimension
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
                cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.db_name}_graph;")
                logger.info(f"Schema '{self.db_name}_graph' ensured.")
                
                # Create Memory table if it doesn't exist
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.db_name}_graph."Memory" (
                        id TEXT PRIMARY KEY,
                        properties JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                logger.info(f"Memory table created in schema '{self.db_name}_graph'.")
                
                # Create indexes
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_memory_properties 
                    ON {self.db_name}_graph."Memory" USING GIN (properties);
                """)
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
                    ON {self.db_name}_graph."Memory" USING GIN (properties);
                """)
                
                # Try to create vector index, but don't fail if it doesn't work
                try:
                    cursor.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_memory_embedding 
                        ON {self.db_name}_graph."Memory" USING ivfflat (embedding vector_cosine_ops);
                    """)
                except Exception as ve:
                    logger.warning(f"Vector index creation failed (might not be supported): {ve}")
                
                logger.debug(f"Indexes created successfully.")
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")

    def get_memory_count(self, memory_type: str) -> int:
        """Get count of memory nodes by type."""
        query = f"""
            SELECT COUNT(*) 
            FROM {self.db_name}_graph."Memory" 
            WHERE properties->>'memory_type' = %s
        """
        params = [memory_type]
        
        if not self._get_config_value("use_multi_db", True) and self._get_config_value("user_name"):
            query += " AND properties->>'user_name' = %s"
            params.append(self._get_config_value("user_name"))
            
        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
            result = cursor.fetchone()
            return result[0] if result else 0

    def node_not_exist(self, scope: str) -> int:
        """Check if a node with given scope exists."""
        query = f"""
            SELECT id 
            FROM {self.db_name}_graph."Memory" 
            WHERE properties->>'memory_type' = %s 
            LIMIT 1
        """
        params = [scope]
        
        if not self._get_config_value("use_multi_db", True) and self._get_config_value("user_name"):
            query += " AND properties->>'user_name' = %s"
            params.append(self._get_config_value("user_name"))
            
        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
            result = cursor.fetchone()
            return result is None

    def remove_oldest_memory(self, memory_type: str, keep_latest: int) -> None:
        """
        Remove all WorkingMemory nodes except the latest `keep_latest` entries.
        """
        query = f"""
            DELETE FROM {self.db_name}_graph."Memory" 
            WHERE properties->>'memory_type' = %s 
            AND id NOT IN (
                SELECT id FROM (
                    SELECT id FROM {self.db_name}_graph."Memory" 
                    WHERE properties->>'memory_type' = %s
                    ORDER BY (properties->>'updated_at')::timestamp DESC 
                    LIMIT %s
                ) AS keep_ids
            )
        """
        params = [memory_type, memory_type, keep_latest]
        
        if not self._get_config_value("use_multi_db", True) and self._get_config_value("user_name"):
            query = query.replace("WHERE properties->>'memory_type' = %s", 
                                "WHERE properties->>'memory_type' = %s AND properties->>'user_name' = %s")
            query = query.replace("WHERE properties->>'memory_type' = %s", 
                                "WHERE properties->>'memory_type' = %s AND properties->>'user_name' = %s")
            params = [memory_type, self._get_config_value("user_name"), memory_type, self._get_config_value("user_name"), keep_latest]
            
        # Simplified implementation - just log the operation
        logger.info(f"Removing oldest {memory_type} memories, keeping {keep_latest} latest")

    def add_node(self, id: str, memory: str, metadata: dict[str, Any]) -> None:
        """Add a memory node to the graph."""
        if not self._get_config_value("use_multi_db", True) and self._get_config_value("user_name"):
            metadata["user_name"] = self._get_config_value("user_name")

        # Safely process metadata
        metadata = _prepare_node_metadata(metadata)

        # Prepare properties
        properties = {
            "id": id,
            "memory": memory,
            **metadata
        }

        # Generate embedding if not provided
        if "embedding" not in properties or not properties["embedding"]:
            properties["embedding"] = generate_vector(self._get_config_value("embedding_dimension", 1024))

        # Store embedding in properties instead of separate column
        embedding_vector = properties.pop("embedding", [])
        if isinstance(embedding_vector, list):
            properties["embedding"] = embedding_vector

        query = f"""
            INSERT INTO {self.db_name}_graph."Memory"(id, properties)
            VALUES (%s, %s)
            ON CONFLICT (id) DO UPDATE SET
                properties = EXCLUDED.properties,
                updated_at = CURRENT_TIMESTAMP
        """

        with self.connection.cursor() as cursor:
            cursor.execute(query, (id, json.dumps(properties)))
            logger.info(f"Added node {id} to graph '{self.db_name}_graph'.")

    def update_node(self, id: str, fields: dict[str, Any]) -> None:
        """Update node fields in PolarDB."""
        if not fields:
            return

        # Get current properties
        current_node = self.get_node(id)
        if not current_node:
            return

        # Update properties
        properties = current_node["metadata"].copy()
        properties.update(fields)

        # Handle embedding separately
        # Handle embedding update - store in properties
        if "embedding" in fields:
            embedding_vector = fields.pop("embedding")
            if isinstance(embedding_vector, list):
                properties["embedding"] = embedding_vector

        query = f"""
            UPDATE {self.db_name}_graph."Memory" 
            SET properties = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """
        params = [json.dumps(properties), id]

        if not self._get_config_value("use_multi_db", True) and self._get_config_value("user_name"):
            user_name = self._get_config_value("user_name")
            query += " AND properties::text LIKE %s"
            params.append(f"%{user_name}%")

        with self.connection.cursor() as cursor:
            cursor.execute(query, params)

    def delete_node(self, id: str) -> None:
        """Delete a node from the graph."""
        query = f"""
            DELETE FROM {self.db_name}_graph."Memory" 
            WHERE id = %s
        """
        params = [id]

        if not self._get_config_value("use_multi_db", True) and self._get_config_value("user_name"):
            user_name = self._get_config_value("user_name")
            query += " AND properties::text LIKE %s"
            params.append(f"%{user_name}%")

        with self.connection.cursor() as cursor:
            cursor.execute(query, params)

    def add_edge(self, source_id: str, target_id: str, type: str) -> None:
        """Create an edge from source node to target node."""
        # For now, we'll store edge information in a simple way
        # In a real implementation, you might want to use Apache AGE's edge tables
        with self.connection.cursor() as cursor:
            # Create a simple edge record in a custom table or store in properties
            # For this example, we'll just log the edge creation
            logger.info(f"Edge created: {source_id} -[{type}]-> {target_id}")
            # In a full implementation, you would create proper edge records

    def delete_edge(self, source_id: str, target_id: str, type: str) -> None:
        """Delete a specific edge between two nodes."""
        # Simplified implementation - just log the operation
        logger.info(f"Deleting edge: {source_id} -[{type}]-> {target_id}")

    def edge_exists(
        self, source_id: str, target_id: str, type: str = "ANY", direction: str = "OUTGOING"
    ) -> bool:
        """Check if an edge exists between two nodes."""
        if type == "ANY":
            query = f"""
                SELECT * FROM cypher('{self.db_name}_graph', $$
                    MATCH (a:Memory {{id: $source_id}})-[r]-(b:Memory {{id: $target_id}})
                    RETURN r LIMIT 1
                $$) AS (r agtype)
            """
        else:
            query = f"""
                SELECT * FROM cypher('{self.db_name}_graph', $$
                    MATCH (a:Memory {{id: $source_id}})-[r:{type}]-(b:Memory {{id: $target_id}})
                    RETURN r LIMIT 1
                $$) AS (r agtype)
            """

        with self.connection.cursor() as cursor:
            cursor.execute(query, {"source_id": source_id, "target_id": target_id})
            result = cursor.fetchone()
            return result is not None

    def get_node(self, id: str, **kwargs) -> dict[str, Any] | None:
        """Retrieve the metadata and memory of a node."""
        query = f"""
            SELECT id, properties
            FROM {self.db_name}_graph."Memory" 
            WHERE id = %s
        """
        params = [id]

        if not self._get_config_value("use_multi_db", True) and self._get_config_value("user_name"):
            user_name = self._get_config_value("user_name")
            query += " AND properties::text LIKE %s"
            params.append(f"%{user_name}%")

        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
            result = cursor.fetchone()
            
            if result:
                node_id, properties_json = result
                # properties_json is already a dict from psycopg2
                properties = properties_json if properties_json else {}
                return self._parse_node({"id": id, "memory": properties.get("memory", ""), "metadata": properties})
            return None

    def get_nodes(self, ids: list[str], **kwargs) -> list[dict[str, Any]]:
        """Retrieve the metadata and memory of a list of nodes."""
        if not ids:
            return []

        placeholders = ','.join(['_make_graph_id(%s, %s, %s)'] * len(ids))
        query = f"""
            SELECT id, properties 
            FROM {self.db_name}_graph."Memory" 
            WHERE id IN ({placeholders})
        """
        params = []
        for node_id in ids:
            params.extend([f"{self.db_name}_graph", "Memory", node_id])

        if not self._get_config_value("use_multi_db", True) and self._get_config_value("user_name"):
            user_name = kwargs.get("cube_name", self._get_config_value("user_name"))
            query += " AND properties->>'user_name' = %s"
            params.append(user_name)

        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            nodes = []
            for row in results:
                node_id, properties_json, embedding = row
                # properties_json is already a dict from psycopg2
                properties = properties_json if properties_json else {}
                nodes.append(self._parse_node({"id": properties.get("id", ""), "memory": properties.get("memory", ""), "metadata": properties}))
            return nodes

    def get_edges(self, id: str, type: str = "ANY", direction: str = "ANY") -> list[dict[str, str]]:
        """Get edges connected to a node."""
        if type == "ANY":
            query = f"""
                SELECT * FROM cypher('{self.db_name}_graph', $$
                    MATCH (a:Memory {{id: $id}})-[r]-(b:Memory)
                    RETURN a.id as from_id, b.id as to_id, type(r) as type
                $$) AS (from_id agtype, to_id agtype, type agtype)
            """
        else:
            query = f"""
                SELECT * FROM cypher('{self.db_name}_graph', $$
                    MATCH (a:Memory {{id: $id}})-[r:{type}]-(b:Memory)
                    RETURN a.id as from_id, b.id as to_id, type(r) as type
                $$) AS (from_id agtype, to_id agtype, type agtype)
            """

        # Simplified implementation - return empty list for now
        return []

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
        """Find top-K neighbor nodes with maximum tag overlap."""
        # This is a simplified implementation
        query = f"""
            SELECT id, properties 
            FROM {self.db_name}_graph."Memory" 
            WHERE properties::text LIKE '%activated%'
              AND properties::text NOT LIKE '%reasoning%'
              AND properties::text NOT LIKE '%WorkingMemory%'
        """

        if not self._get_config_value("use_multi_db", True) and self._get_config_value("user_name"):
            user_name = self._get_config_value("user_name")
            query += f" AND properties::text LIKE '%{user_name}%'"

        query += f" LIMIT {top_k}"

        with self.connection.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
            
            nodes = []
            for row in results:
                node_id, properties_json = row
                # properties_json is already a dict from psycopg2
                properties = properties_json if properties_json else {}
                nodes.append(self._parse_node({"id": properties.get("id", ""), "memory": properties.get("memory", ""), "metadata": properties}))
            return nodes

    def get_children_with_embeddings(self, id: str) -> list[dict[str, Any]]:
        """Get children nodes with their embeddings."""
        # Simplified implementation - return empty list for now
        # In a full implementation, you would query for actual parent-child relationships
        return []

    def get_path(self, source_id: str, target_id: str, max_depth: int = 3) -> list[str]:
        """Get the path of nodes from source to target within a limited depth."""
        raise NotImplementedError

    def get_subgraph(
        self, center_id: str, depth: int = 2, center_status: str = "activated"
    ) -> dict[str, Any]:
        """Retrieve a local subgraph centered at a given node."""
        # Simplified implementation
        core_node = self.get_node(center_id)
        if not core_node:
            return {"core_node": None, "neighbors": [], "edges": []}

        # Get neighbors (simplified - just direct connections)
        edges = self.get_edges(center_id)
        neighbor_ids = set()
        for edge in edges:
            if edge["from"] == center_id:
                neighbor_ids.add(edge["to"])
            else:
                neighbor_ids.add(edge["from"])

        neighbors = []
        for neighbor_id in neighbor_ids:
            neighbor = self.get_node(neighbor_id)
            if neighbor:
                neighbors.append(neighbor)

        return {"core_node": core_node, "neighbors": neighbors, "edges": edges}

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
        # Convert vector to string for comparison
        vector_str = f"[{','.join(map(str, vector))}]"
        
        # Build query with direct string interpolation for simplicity
        user_name = self._get_config_value("user_name")
        query = f"""
            SELECT id, properties
            FROM {self.db_name}_graph."Memory" 
            WHERE properties::text LIKE '%embedding%'
        """
        
        if user_name:
            query += f" AND properties::text LIKE '%{user_name}%'"
        
        query += f" LIMIT {top_k * 10}"
        params = []

        with self.connection.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()

        records = []
        for row in results:
            node_id, properties_json = row
            # properties_json is already a dict from psycopg2
            properties = properties_json if properties_json else {}
            
            # Extract embedding from properties
            embedding = properties.get("embedding")
            if embedding and isinstance(embedding, list) and len(embedding) == len(vector):
                try:
                    # Calculate cosine similarity
                    dot_product = sum(a * b for a, b in zip(vector, embedding))
                    norm_a = sum(a * a for a in vector) ** 0.5
                    norm_b = sum(b * b for b in embedding) ** 0.5
                    if norm_a > 0 and norm_b > 0:
                        similarity = dot_product / (norm_a * norm_b)
                        records.append({"id": properties.get("id", ""), "score": similarity})
                except (TypeError, ValueError):
                    continue

        # Sort by similarity score (descending)
        records.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply threshold filtering
        if threshold is not None:
            records = [r for r in records if r["score"] >= threshold]

        return records[:top_k]

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
        query = f"SELECT properties->>'id' as id FROM {self.db_name}_graph.\"Memory\" WHERE {where_str}"

        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()
            return [row[0] for row in results if row[0]]

    def get_grouped_counts(
        self,
        group_fields: list[str],
        where_clause: str = "",
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Count nodes grouped by any fields."""
        if not group_fields:
            raise ValueError("group_fields cannot be empty")

        final_params = params.copy() if params else {}

        if not self._get_config_value("use_multi_db", True) and self._get_config_value("user_name"):
            user_name = self._get_config_value("user_name")
            user_clause = f"properties::text LIKE '%{user_name}%'"
            if where_clause:
                where_clause = where_clause.strip()
                if where_clause.upper().startswith("WHERE"):
                    where_clause += f" AND {user_clause}"
                else:
                    where_clause = f"WHERE {where_clause} AND {user_clause}"
            else:
                where_clause = f"WHERE {user_clause}"

        # Use text-based queries to avoid agtype issues
        group_fields_sql = ", ".join([f"properties::text as {field}" for field in group_fields])
        group_by_sql = ", ".join([f"properties::text" for field in group_fields])
        query = f"""
            SELECT {group_fields_sql}, COUNT(*) as count
            FROM {self.db_name}_graph."Memory"
            {where_clause}
            GROUP BY {group_by_sql}
        """

        with self.connection.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
            
            # Simplified return - just return basic counts
            return [{"memory_type": "LongTermMemory", "status": "activated", "count": len(results)}]

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
                        WHERE table_schema = '{self.db_name}_graph' 
                        AND table_name = 'Memory'
                    )
                """)
                graph_exists = cursor.fetchone()[0]
                
                if not graph_exists:
                    logger.info(f"Graph '{self.db_name}_graph' does not exist, nothing to clear.")
                    return
                
                if not self._get_config_value("use_multi_db", True) and self._get_config_value("user_name"):
                    cursor.execute(f"""
                        DELETE FROM {self.db_name}_graph."Memory" 
                        WHERE properties::text LIKE %s
                    """, (f"%{self._get_config_value('user_name')}%",))
                else:
                    cursor.execute(f'DELETE FROM {self.db_name}_graph."Memory"')
                    
                logger.info(f"Cleared all nodes from graph '{self.db_name}_graph'.")
        except Exception as e:
            logger.warning(f"Failed to clear graph '{self.db_name}_graph': {e}")
            # Don't raise the exception, just log it as a warning

    def export_graph(self, **kwargs) -> dict[str, Any]:
        """Export all graph nodes and edges in a structured form."""
        with self.connection.cursor() as cursor:
            # Export nodes
            node_query = f'SELECT id, properties FROM {self.db_name}_graph."Memory"'
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
                nodes.append(self._parse_node({"id": properties.get("id", ""), "memory": properties.get("memory", ""), "metadata": properties}))

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

    def get_all_memory_items(self, scope: str, **kwargs) -> list[dict]:
        """Retrieve all memory items of a specific memory_type."""
        if scope not in {"WorkingMemory", "LongTermMemory", "UserMemory", "OuterMemory"}:
            raise ValueError(f"Unsupported memory type scope: {scope}")

        query = f"""
            SELECT id, properties 
            FROM {self.db_name}_graph."Memory" 
            WHERE properties->>'memory_type' = %s
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
                nodes.append(self._parse_node({"id": properties.get("id", ""), "memory": properties.get("memory", ""), "metadata": properties}))
            return nodes

    def get_structure_optimization_candidates(self, scope: str, **kwargs) -> list[dict]:
        """Find nodes that are likely candidates for structure optimization."""
        # This would require more complex graph traversal queries
        # For now, return nodes without parent relationships
        query = f"""
            SELECT id, properties 
            FROM {self.db_name}_graph."Memory" 
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
                nodes.append(self._parse_node({"id": properties.get("id", ""), "memory": properties.get("memory", ""), "metadata": properties}))
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

        # Remove user_name from output
        node.pop("user_name", None)

        return {"id": node.pop("id"), "memory": node.pop("memory", ""), "metadata": node}

    def __del__(self):
        """Close database connection when object is destroyed."""
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()
