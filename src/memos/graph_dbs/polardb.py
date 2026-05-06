import hashlib
import json
import random
import textwrap
import threading
import time

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Literal

import numpy as np

from memos.configs.graph_db import PolarDBGraphDBConfig
from memos.dependency import require_python_package
from memos.graph_dbs.base import BaseGraphDB
from memos.log import get_logger
from memos.utils import timed


logger = get_logger(__name__)


def _compose_node(item: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    node_id = item["id"]
    memory = item["memory"]
    metadata = item.get("metadata", {})
    return node_id, memory, metadata


def _prepare_node_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    now = datetime.utcnow().isoformat()

    metadata.setdefault("created_at", now)
    metadata.setdefault("updated_at", now)

    embedding = metadata.get("embedding")
    if embedding and isinstance(embedding, list):
        metadata["embedding"] = [float(x) for x in embedding]

    return metadata


def generate_vector(dim=1024, low=-0.2, high=0.2):
    return [round(random.uniform(low, high), 6) for _ in range(dim)]


def detect_embedding_field(embedding_list):
    if not embedding_list:
        return None
    dim = len(embedding_list)
    if dim == 1024:
        return "embedding"
    else:
        logger.warning(f"Unknown embedding dimension {dim}, skipping this vector")
        return None


def convert_to_vector(embedding_list):
    if not embedding_list:
        return None
    if isinstance(embedding_list, np.ndarray):
        embedding_list = embedding_list.tolist()
    return "[" + ",".join(str(float(x)) for x in embedding_list) + "]"


def clean_properties(props):
    vector_keys = {"embedding", "embedding_1024", "embedding_3072", "embedding_768"}
    if not isinstance(props, dict):
        return {}
    return {k: v for k, v in props.items() if k not in vector_keys}


def escape_sql_string(value: str) -> str:
    return value.replace("'", "''")


class PolarDBGraphDB(BaseGraphDB):
    @require_python_package(
        import_name="psycopg2",
        install_command="pip install psycopg2-binary",
        install_link="https://pypi.org/project/psycopg2-binary/",
    )
    def __init__(self, config: PolarDBGraphDBConfig):
        import psycopg2.pool

        self.config = config

        if isinstance(config, dict):
            self.db_name = config.get("db_name")
            self.user_name = config.get("user_name")
            host = config.get("host")
            port = config.get("port")
            user = config.get("user")
            password = config.get("password")
            maxconn = config.get("maxconn", 10)
            self._connection_wait_timeout = config.get("connection_wait_timeout", 30)
            self._skip_connection_health_check = config.get("skip_connection_health_check", False)
            self._warm_up_on_startup_by_full = config.get("warm_up_on_startup_by_full", False)
            self._warm_up_on_startup_by_all = config.get("warm_up_on_startup_by_all", False)
        else:
            self.db_name = config.db_name
            self.user_name = config.user_name
            host = config.host
            port = config.port
            user = config.user
            password = config.password
            maxconn = config.maxconn if hasattr(config, "maxconn") else 10
            self._connection_wait_timeout = getattr(config, "connection_wait_timeout", 30)
            self._skip_connection_health_check = getattr(
                config, "skip_connection_health_check", False
            )
            self._warm_up_on_startup_by_full = getattr(config, "warm_up_on_startup_by_full", False)
            self._warm_up_on_startup_by_all = getattr(config, "warm_up_on_startup_by_all", False)
            logger.info(
                f"polardb init config connection_wait_timeout:{self._connection_wait_timeout},_skip_connection_health_check:{self._skip_connection_health_check},warm_up_on_startup_by_full:{self._warm_up_on_startup_by_full},warm_up_on_startup_by_all:{self._warm_up_on_startup_by_all}"
            )

        logger.info(
            f" polardb init db_name: {self.db_name} && maxconn: {maxconn} && connection_wait_timeout: {self._connection_wait_timeout}s"
        )

        self._shard_count = int(
            config.get("shard_count", 400)
            if isinstance(config, dict)
            else getattr(config, "shard_count", 400)
        )
        shard_schemas = ",".join(f"{self.db_name}_graph_{i}" for i in range(self._shard_count))
        self._all_shards_search_path = (
            f'{self.db_name}_graph,{shard_schemas},ag_catalog,"$user",public'
        )
        self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=maxconn,
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=self.db_name,
            connect_timeout=10,
            keepalives_idle=120,
            keepalives_interval=15,
            keepalives_count=5,
            keepalives=1,
            options=f"-c search_path={self._all_shards_search_path}",
        )

        self._semaphore = threading.BoundedSemaphore(maxconn)

    def _get_config_value(self, key: str, default=None):
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        else:
            return getattr(self.config, key, default)

    def _warm_up_search_connections_by_full(self, user_name: str | None = None) -> None:
        logger.info("--warm_up_search_connections_by_full--start-up----")
        user_name = user_name or self.user_name
        if not user_name:
            logger.debug("[warm_up] Skipped: no user_name for warm-up")
            return
        warm_count = min(5, self.connection_pool.minconn)
        for _ in range(warm_count):
            try:
                self.search_by_fulltext(
                    query_words=["warmup"],
                    top_k=1,
                    user_name=user_name,
                )
            except Exception as e:
                logger.debug(f"[warm_up] Warm-up query failed (non-fatal): {e}")
                break
        logger.info(f"[warm_up] Pre-warmed {warm_count} connections for search_by_fulltext")

    def warm_up_search_connections_by_full(self, user_name: str | None = None) -> None:
        self._warm_up_search_connections_by_full(user_name)

    def _warm_up_connections_by_all(self):
        logger.info("--_warm_up_connections_by_all--start-up")
        warm_count = self.connection_pool.minconn
        preheated = 0
        logger.info(f"[warm_up] Pre-warming {warm_count} connections...")
        for _ in range(warm_count):
            try:
                with self._get_connection() as conn, conn.cursor() as cur:
                    cur.execute("SELECT 1")
                preheated += 1
            except Exception as e:
                logger.warning(f"[warm_up] Failed to pre-warm connection: {e}")
                continue
        logger.info(f"[warm_up] Pre-warmed {preheated}/{warm_count} connections")

    @contextmanager
    def _get_connection(self):
        import psycopg2

        timeout = self._connection_wait_timeout
        if timeout is None or timeout <= 0:
            self._semaphore.acquire()
        elif not self._semaphore.acquire(timeout=timeout):
            logger.warning(f"Timeout waiting for connection slot ({timeout}s)")
            raise RuntimeError("Connection pool busy")

        conn = None
        broken = False
        try:
            for attempt in range(2):
                conn = self.connection_pool.getconn()
                conn.autocommit = True
                if self._skip_connection_health_check:
                    break
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                    break
                except psycopg2.Error:
                    logger.warning("Dead connection detected, recreating (attempt %d)", attempt + 1)
                    self.connection_pool.putconn(conn, close=True)
                    conn = None
            else:
                raise RuntimeError("Cannot obtain valid DB connection after 2 attempts")
            with conn.cursor() as cur:
                cur.execute(f"SET search_path = {self._all_shards_search_path};")
            yield conn
        except psycopg2.Error as e:
            broken = True
            logger.error("Database connection error: %s", e)
            raise
        except Exception:
            raise
        finally:
            if conn is not None:
                try:
                    self.connection_pool.putconn(conn, close=broken)
                except Exception as e:
                    logger.warning("Failed to return connection to pool: %s", e)
            self._semaphore.release()

    def _get_shard_schema_raw(self, user_name: str | None) -> str:
        if not user_name:
            return f"{self.db_name}_graph_0"
        hash_val = int(hashlib.md5(user_name.encode("utf-8")).hexdigest(), 16)
        shard_id = hash_val % self._shard_count
        return f"{self.db_name}_graph_{shard_id}"

    def get_memory_graph_table_name(self, user_name: str | None) -> str:
        return f'"{self._get_shard_schema_raw(user_name)}"'

    def _get_all_shard_table_names(self) -> list[str]:
        return [f'"{self.db_name}_graph_{i}"' for i in range(self._shard_count)]

    def _get_all_shard_schemas(self) -> list[str]:
        return [f"{self.db_name}_graph_{i}" for i in range(self._shard_count)]

    def _get_existing_shard_schemas(self) -> list[str]:
        expected_prefix = f"{self.db_name}_graph_"
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(
                    "SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE %s",
                    (f"{expected_prefix}%",),
                )
                rows = cursor.fetchall()
            existing: list[str] = []
            for row in rows:
                name = row[0]
                if not isinstance(name, str) or not name.startswith(expected_prefix):
                    continue
                suffix = name[len(expected_prefix):]
                if suffix.isdigit() and int(suffix) < self._shard_count:
                    existing.append(name)
            return existing
        except Exception as e:
            logger.warning(
                "Failed to fetch existing shard schemas, fallback to all configured shards: %s",
                e,
            )
            return self._get_all_shard_schemas()

    def _ensure_database_exists(self):
        try:
            logger.info(f"Using database '{self.db_name}'")
        except Exception as e:
            logger.error(f"Failed to access database '{self.db_name}': {e}")
            raise

    @timed
    def _create_graph(self):
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.db_name}_graph";')
                logger.info(f"Schema '{self.db_name}_graph' ensured.")

                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS "{self.db_name}_graph"."Memory" (
                        id TEXT PRIMARY KEY,
                        properties JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                logger.info(f"Memory table created in schema '{self.db_name}_graph'.")

                try:
                    cursor.execute(f"""
                        ALTER TABLE "{self.db_name}_graph"."Memory"
                        ADD COLUMN IF NOT EXISTS embedding JSONB;
                    """)
                    logger.info("Embedding column added to Memory table.")
                except Exception as e:
                    logger.warning(f"Failed to add embedding column: {e}")

                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_memory_properties
                    ON "{self.db_name}_graph"."Memory" USING GIN (properties);
                """)

                try:
                    cursor.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_memory_embedding
                        ON "{self.db_name}_graph"."Memory" USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    """)
                    logger.info("Vector index created for Memory table.")
                except Exception as e:
                    logger.warning(f"Vector index creation failed (might not be supported): {e}")

                logger.info("Indexes created for Memory table.")

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
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_memory_properties
                    ON "{self.db_name}_graph"."Memory" USING GIN (properties);
                """)

                try:
                    cursor.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_memory_embedding
                        ON "{self.db_name}_graph"."Memory" USING ivfflat (embedding vector_cosine_ops);
                    """)
                except Exception as ve:
                    logger.warning(f"Vector index creation failed (might not be supported): {ve}")

                logger.debug("Indexes created successfully.")
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")

    def get_memory_count(self, memory_type: str, user_name: str | None = None) -> int:
        logger.info(
            "get_memory_count request: memory_type=%s, user_name=%s", memory_type, user_name
        )
        start_time = time.perf_counter()
        type_param = self.format_param_value(memory_type)
        if user_name:
            tbl = self.get_memory_graph_table_name(user_name)
            query = (
                f'SELECT COUNT(*) FROM {tbl}."Memory"'
                f" WHERE ag_catalog.agtype_access_operator(properties, '\"memory_type\"'::agtype) = %s::agtype"
                f" AND ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = %s::agtype"
            )
            params = [type_param, self.format_param_value(user_name)]
        else:
            union_parts = [
                f'SELECT COUNT(*) AS cnt FROM {tbl}."Memory"'
                f" WHERE ag_catalog.agtype_access_operator(properties, '\"memory_type\"'::agtype) = %s::agtype"
                for tbl in self._get_all_shard_table_names()
            ]
            query = f"SELECT SUM(cnt) FROM ({' UNION ALL '.join(union_parts)}) t"
            params = [type_param] * len(union_parts)

        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(query, params)
                result = cursor.fetchone()
                elapsed = (time.perf_counter() - start_time) * 1000
                logger.info("get_memory_count completed in %.2f ms", elapsed)
                return int(result[0]) if result and result[0] else 0
        except Exception as e:
            logger.error(f"[get_memory_count] Failed: {e}")
            return -1

    @timed
    def node_not_exist(self, scope: str, user_name: str | None = None) -> int:
        logger.info(" node_not_exist request: scope=%s, user_name=%s", scope, user_name)
        scope_param = self.format_param_value(scope)
        if user_name:
            tbl = self.get_memory_graph_table_name(user_name)
            query = (
                f'SELECT id FROM {tbl}."Memory"'
                f" WHERE ag_catalog.agtype_access_operator(properties, '\"memory_type\"'::agtype) = %s::agtype"
                f" AND ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = %s::agtype"
                f" LIMIT 1"
            )
            params = [scope_param, self.format_param_value(user_name)]
        else:
            union_parts = [
                f'SELECT id FROM {tbl}."Memory"'
                f" WHERE ag_catalog.agtype_access_operator(properties, '\"memory_type\"'::agtype) = %s::agtype"
                for tbl in self._get_all_shard_table_names()
            ]
            query = " UNION ALL ".join(union_parts) + " LIMIT 1"
            params = [scope_param] * len(union_parts)
        logger.info("node_not_exist query=%s, params=%s", query, params)
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(query, params)
                result = cursor.fetchone()
                return 1 if result else 0
        except Exception as e:
            logger.error(f"[node_not_exist] Query failed: {e}", exc_info=True)
            raise

    @timed
    def remove_oldest_memory(
        self, memory_type: str, keep_latest: int, user_name: str | None = None
    ) -> None:
        start_time = time.perf_counter()
        logger.info(
            "remove_oldest_memory by memory_type:%s,keep_latest: %s,user_name:%s",
            memory_type,
            keep_latest,
            user_name,
        )

        if user_name:
            shard_tables = [self.get_memory_graph_table_name(user_name)]
        else:
            shard_tables = self._get_all_shard_table_names()

        type_param = self.format_param_value(memory_type)
        total_deleted = 0

        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                for tbl in shard_tables:
                    if user_name:
                        select_query = (
                            f'SELECT id FROM {tbl}."Memory"'
                            f" WHERE ag_catalog.agtype_access_operator(properties, '\"memory_type\"'::agtype) = %s::agtype"
                            f" AND ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = %s::agtype"
                            f" ORDER BY ag_catalog.agtype_access_operator(properties, '\"updated_at\"'::agtype) DESC"
                            f" OFFSET %s"
                        )
                        select_params = [
                            type_param,
                            self.format_param_value(user_name),
                            keep_latest,
                        ]
                    else:
                        select_query = (
                            f'SELECT id FROM {tbl}."Memory"'
                            f" WHERE ag_catalog.agtype_access_operator(properties, '\"memory_type\"'::agtype) = %s::agtype"
                            f" ORDER BY ag_catalog.agtype_access_operator(properties, '\"updated_at\"'::agtype) DESC"
                            f" OFFSET %s"
                        )
                        select_params = [type_param, keep_latest]
                    logger.info(
                        "remove_oldest_memory select_query=%s, select_params=%s",
                        select_query,
                        select_params,
                    )
                    cursor.execute(select_query, select_params)
                    ids_to_delete = [row[0] for row in cursor.fetchall()]

                    if not ids_to_delete:
                        continue

                    placeholders = ",".join(["%s"] * len(ids_to_delete))
                    delete_query = f'DELETE FROM {tbl}."Memory" WHERE id IN ({placeholders})'
                    logger.info(
                        "remove_oldest_memory delete_query=%s, ids_to_delete=%s",
                        delete_query,
                        ids_to_delete,
                    )
                    cursor.execute(delete_query, ids_to_delete)
                    total_deleted += cursor.rowcount

                elapsed = (time.perf_counter() - start_time) * 1000.0
                logger.info(
                    "remove_oldest_memory removed %d %s memories, keeping %d latest, took %.1f ms",
                    total_deleted,
                    memory_type,
                    keep_latest,
                    elapsed,
                )
        except Exception as e:
            logger.error(f"[remove_oldest_memory] Failed: {e}", exc_info=True)
            raise

    @timed
    def update_node(self, id: str, fields: dict[str, Any], user_name: str | None = None) -> None:
        logger.info(
            "update_node id=%s, user_name=%s, fields_keys=%s",
            id,
            user_name,
            list(fields.keys()) if fields else [],
        )
        if not fields:
            return

        resolved_user_name = user_name

        current_node = self.get_node(id, user_name=resolved_user_name)
        if not current_node:
            logger.info("update_node Node '%s' not found, skip update", id)
            return

        if not resolved_user_name:
            resolved_user_name = current_node.get("metadata", {}).get("user_name")

        properties = current_node["metadata"].copy()
        original_id = properties.get("id", id)
        original_memory = current_node.get("memory", "")

        if "memory" in fields:
            original_memory = fields.pop("memory")

        properties.update(fields)
        properties["id"] = original_id
        properties["memory"] = original_memory

        embedding_vector = None
        if "embedding" in fields:
            embedding_vector = fields.pop("embedding")
            if not isinstance(embedding_vector, list):
                embedding_vector = None

        tbl = self.get_memory_graph_table_name(resolved_user_name)
        id_param = self.format_param_value(id)

        if embedding_vector is not None:
            query = (
                f'UPDATE {tbl}."Memory"'
                f" SET properties = %s, embedding = %s"
                f" WHERE ag_catalog.agtype_access_operator(properties, '\"id\"'::agtype) = %s::agtype"
            )
            params = [json.dumps(properties), json.dumps(embedding_vector), id_param]
        else:
            query = (
                f'UPDATE {tbl}."Memory"'
                f" SET properties = %s"
                f" WHERE ag_catalog.agtype_access_operator(properties, '\"id\"'::agtype) = %s::agtype"
            )
            params = [json.dumps(properties), id_param]

        if resolved_user_name:
            query += " AND ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = %s::agtype"
            params.append(self.format_param_value(resolved_user_name))

        logger.info("update_node query=%s, params_count=%d", query, len(params))
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(query, params)
        except Exception as e:
            logger.error(f"[update_node] Failed to update node '{id}': {e}", exc_info=True)
            raise

    @timed
    def delete_node(self, id: str, user_name: str | None = None) -> None:
        logger.info("delete_node id=%s, user_name=%s", id, user_name)
        resolved_user_name = user_name

        if not resolved_user_name:
            node = self.get_node(id)
            if node:
                resolved_user_name = node.get("metadata", {}).get("user_name")
            if not resolved_user_name:
                logger.warning("delete_node node '%s' not found, skip delete", id)
                return

        tbl = self.get_memory_graph_table_name(resolved_user_name)
        id_param = self.format_param_value(id)
        query = (
            f'DELETE FROM {tbl}."Memory"'
            f" WHERE ag_catalog.agtype_access_operator(properties, '\"id\"'::agtype) = %s::agtype"
            f" AND ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = %s::agtype"
        )
        params = [id_param, self.format_param_value(resolved_user_name)]

        logger.info("delete_node query=%s, params=%s", query, params)
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(query, params)
        except Exception as e:
            logger.error("delete_node failed to delete node '%s': %s", id, e, exc_info=True)
            raise

    @timed
    def create_extension(self):
        extensions = [("polar_age", "Graph engine"), ("vector", "Vector engine")]
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute("SELECT current_database();")
                current_db = cursor.fetchone()[0]
                logger.info(f"Current database context: {current_db}")

                for ext_name, ext_desc in extensions:
                    try:
                        cursor.execute(f"create extension if not exists {ext_name};")
                        logger.info(f"Extension '{ext_name}' ({ext_desc}) ensured.")
                    except Exception as e:
                        if "already exists" in str(e):
                            logger.info(f"Extension '{ext_name}' ({ext_desc}) already exists.")
                        else:
                            logger.warning(
                                f"Failed to create extension '{ext_name}' ({ext_desc}): {e}"
                            )
                            logger.error(
                                f"Failed to create extension '{ext_name}': {e}", exc_info=True
                            )
        except Exception as e:
            logger.warning(f"Failed to access database context: {e}")
            logger.error(f"Failed to access database context: {e}", exc_info=True)

    @timed
    def create_graph(self):
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(f"""
                        SELECT COUNT(*) FROM ag_catalog.ag_graph
                        WHERE name = '{self.db_name}_graph';
                    """)
                graph_exists = cursor.fetchone()[0] > 0

                if graph_exists:
                    logger.info(f"Graph '{self.db_name}_graph' already exists.")
                else:
                    cursor.execute(f"select create_graph('{self.db_name}_graph');")
                    logger.info(f"Graph database '{self.db_name}_graph' created.")
        except Exception as e:
            logger.warning(f"Failed to create graph '{self.db_name}_graph': {e}")
            logger.error(f"Failed to create graph '{self.db_name}_graph': {e}", exc_info=True)

    @timed
    def create_edge(self):
        valid_rel_types = {"AGGREGATE_TO", "FOLLOWS", "INFERS", "MERGED_TO", "RELATE_TO", "PARENT"}

        for label_name in valid_rel_types:
            logger.info(f"Creating elabel: {label_name}")
            try:
                with self._get_connection() as conn, conn.cursor() as cursor:
                    cursor.execute(f"select create_elabel('{self.db_name}_graph', '{label_name}');")
                    logger.info(f"Successfully created elabel: {label_name}")
            except Exception as e:
                if "already exists" in str(e):
                    logger.info(f"Label '{label_name}' already exists, skipping.")
                else:
                    logger.warning(f"Failed to create label {label_name}: {e}")
                    logger.error(f"Failed to create elabel '{label_name}': {e}", exc_info=True)

    @timed
    def add_edge(
        self, source_id: str, target_id: str, type: str, user_name: str | None = None
    ) -> None:
        logger.info(
            "add_edge source_id=%s, target_id=%s, type=%s, user_name=%s",
            source_id,
            target_id,
            type,
            user_name,
        )
        start_time = time.perf_counter()
        if not source_id or not target_id:
            logger.error("add_edge source_id and target_id must not be empty")
            return

        resolved_user_name = user_name

        source_node = self.get_node(source_id, user_name=resolved_user_name)
        target_node = self.get_node(target_id, user_name=resolved_user_name)

        if not source_node or not target_node:
            logger.warning(
                "add_edge source %s exists=%s, target %s exists=%s, skip",
                source_id,
                source_node is not None,
                target_id,
                target_node is not None,
            )
            return

        if not resolved_user_name:
            resolved_user_name = source_node.get("metadata", {}).get("user_name")

        schema_raw = self._get_shard_schema_raw(resolved_user_name)
        properties = {}
        if resolved_user_name:
            properties["user_name"] = resolved_user_name

        query = f"""
            INSERT INTO {schema_raw}."{type}"(id, start_id, end_id, properties)
            SELECT
                ag_catalog._next_graph_id('{schema_raw}'::name, '{type}'),
                ag_catalog._make_graph_id('{schema_raw}'::name, 'Memory'::name, '{source_id}'::text::cstring),
                ag_catalog._make_graph_id('{schema_raw}'::name, 'Memory'::name, '{target_id}'::text::cstring),
                jsonb_build_object('user_name', '{resolved_user_name}')::text::agtype
            WHERE NOT EXISTS (
                SELECT 1 FROM {schema_raw}."{type}"
                WHERE start_id = ag_catalog._make_graph_id('{schema_raw}'::name, 'Memory'::name, '{source_id}'::text::cstring)
                  AND end_id   = ag_catalog._make_graph_id('{schema_raw}'::name, 'Memory'::name, '{target_id}'::text::cstring)
            );
        """
        logger.info("add_edge query=%s", query)
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(query)
                elapsed_time = (time.perf_counter() - start_time) * 1000.0
                logger.info(
                    "add_edge created completed successfully in took %.1f ms",
                    elapsed_time,
                )
        except Exception as e:
            logger.error("add_edge failed: %s", e, exc_info=True)
            raise

    @timed
    def delete_edge(
        self, source_id: str, target_id: str, type: str, user_name: str | None = None
    ) -> None:
        logger.info(
            "delete_edge source_id=%s, target_id=%s, type=%s, user_name=%s",
            source_id,
            target_id,
            type,
            user_name,
        )
        resolved_user_name = user_name

        if not resolved_user_name:
            source_node = self.get_node(source_id)
            if source_node:
                resolved_user_name = source_node.get("metadata", {}).get("user_name")
            if not resolved_user_name:
                logger.warning("delete_edge cannot resolve shard for source_id=%s, skip", source_id)
                return

        tbl = self.get_memory_graph_table_name(resolved_user_name)
        query = (
            f'DELETE FROM {tbl}."Edges" WHERE source_id = %s AND target_id = %s AND edge_type = %s'
        )
        params = [source_id, target_id, type]
        logger.info("delete_edge query=%s, params=%s", query, params)
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(query, params)
                logger.info("delete_edge done %s -[%s]-> %s", source_id, type, target_id)
        except Exception as e:
            logger.error("delete_edge failed: %s", e, exc_info=True)
            raise

    @timed
    def edge_exists(
        self,
        source_id: str,
        target_id: str,
        type: str = "ANY",
        direction: str = "OUTGOING",
        user_name: str | None = None,
    ) -> bool:
        logger.info(
            "edge_exists source_id=%s, target_id=%s, type=%s, direction=%s, user_name=%s",
            source_id,
            target_id,
            type,
            direction,
            user_name,
        )
        resolved_user_name = user_name

        if direction == "OUTGOING":
            pattern = "(a:Memory)-[r]->(b:Memory)"
        elif direction == "INCOMING":
            pattern = "(a:Memory)<-[r]-(b:Memory)"
        elif direction == "ANY":
            pattern = "(a:Memory)-[r]-(b:Memory)"
        else:
            raise ValueError(
                f"Invalid direction: {direction}. Must be 'OUTGOING', 'INCOMING', or 'ANY'."
            )

        type_filter = f" AND type(r) = '{type}'" if type != "ANY" else ""

        if resolved_user_name:
            schema_raw = self._get_shard_schema_raw(resolved_user_name)
            query = (
                f"SELECT * FROM cypher('{schema_raw}', $$"
                f" MATCH {pattern}"
                f" WHERE a.user_name = '{resolved_user_name}' AND b.user_name = '{resolved_user_name}'"
                f" AND a.id = '{source_id}' AND b.id = '{target_id}'"
                f"{type_filter}"
                f" RETURN r"
                f" $$) AS (r agtype)"
            )
        else:
            union_parts = []
            for schema_raw in [f"{self.db_name}_graph_{i}" for i in range(self._shard_count)]:
                part = (
                    f"SELECT * FROM cypher('{schema_raw}', $$"
                    f" MATCH {pattern}"
                    f" WHERE a.id = '{source_id}' AND b.id = '{target_id}'"
                    f"{type_filter}"
                    f" RETURN r"
                    f" $$) AS (r agtype)"
                )
                union_parts.append(part)
            query = " UNION ALL ".join(union_parts) + " LIMIT 1"

        logger.info("edge_exists query=%s", query)
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchone()
                return result is not None and result[0] is not None
        except Exception as e:
            logger.error("edge_exists failed: %s", e, exc_info=True)
            raise

    @timed
    def get_node(
        self, id: str, include_embedding: bool = False, user_name: str | None = None
    ) -> dict[str, Any] | None:
        logger.info(
            f"polardb get_node id: {id}, include_embedding: {include_embedding}, user_name: {user_name}"
        )
        start_time = time.perf_counter()
        select_fields = "id, properties, embedding" if include_embedding else "id, properties"
        id_param = self.format_param_value(id)

        if user_name is not None:
            tbl = self.get_memory_graph_table_name(user_name)
            query = (
                f"SELECT {select_fields}"
                f' FROM {tbl}."Memory"'
                f" WHERE ag_catalog.agtype_access_operator(properties, '\"id\"'::agtype) = %s::agtype"
                f" AND ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = %s::agtype"
            )
            params = [id_param, self.format_param_value(user_name)]
        else:
            union_parts = [
                f'SELECT {select_fields} FROM {tbl}."Memory"'
                f" WHERE ag_catalog.agtype_access_operator(properties, '\"id\"'::agtype) = %s::agtype"
                for tbl in self._get_all_shard_table_names()
            ]
            query = " UNION ALL ".join(union_parts) + " LIMIT 1"
            params = [id_param] * len(union_parts)

        logger.info(f"polardb [get_node] query: {query},params: {params}")
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(query, params)
                result = cursor.fetchone()

                if result:
                    if include_embedding:
                        _, properties_json, embedding_json = result
                    else:
                        _, properties_json = result
                        embedding_json = None

                    if isinstance(properties_json, str):
                        try:
                            properties = json.loads(properties_json)
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(f"Failed to parse properties for node {id}")
                            properties = {}
                    else:
                        properties = properties_json if properties_json else {}

                    if include_embedding and embedding_json is not None:
                        try:
                            embedding = (
                                json.loads(embedding_json)
                                if isinstance(embedding_json, str)
                                else embedding_json
                            )
                            properties["embedding"] = embedding
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(f"Failed to parse embedding for node {id}")

                    elapsed_time = (time.perf_counter() - start_time) * 1000.0
                    logger.info(
                        "polardb get_node get_node completed time in took %.1f ms",
                        elapsed_time,
                    )
                    return self._parse_node(
                        {
                            "id": id,
                            "memory": properties.get("memory", ""),
                            **properties,
                        }
                    )
                return None

        except Exception as e:
            logger.error(f"[get_node] Failed to retrieve node '{id}': {e}", exc_info=True)
            return None

    @timed
    def get_nodes(
        self, ids: list[str], user_name: str | None = None, **kwargs
    ) -> list[dict[str, Any]]:
        logger.info("get_nodes ids=%s, user_name=%s", ids, user_name)
        if not ids:
            return []

        resolved_user_name = user_name
        placeholders = ",".join(["%s"] * len(ids))
        id_params = [self.format_param_value(id_val) for id_val in ids]

        if resolved_user_name:
            tbl = self.get_memory_graph_table_name(resolved_user_name)
            query = (
                f'SELECT id, properties, embedding FROM {tbl}."Memory"'
                f" WHERE ag_catalog.agtype_access_operator(properties, '\"id\"'::agtype) = ANY(ARRAY[{placeholders}]::agtype[])"
                f" AND ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = %s::agtype"
            )
            params = [*id_params, self.format_param_value(resolved_user_name)]
        else:
            union_parts = [
                f'SELECT id, properties, embedding FROM {tbl}."Memory"'
                f" WHERE ag_catalog.agtype_access_operator(properties, '\"id\"'::agtype) = ANY(ARRAY[{placeholders}]::agtype[])"
                for tbl in self._get_all_shard_table_names()
            ]
            query = " UNION ALL ".join(union_parts)
            params = []
            for _ in union_parts:
                params.extend(id_params)

        logger.info("get_nodes query=%s, params_count=%d", query, len(params))

        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()

                nodes = []
                for row in results:
                    node_id, properties_json, embedding_json = row
                    if isinstance(properties_json, str):
                        try:
                            properties = json.loads(properties_json)
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(
                                "get_nodes failed to parse properties for node %s", node_id
                            )
                            properties = {}
                    else:
                        properties = properties_json if properties_json else {}

                    if embedding_json is not None and kwargs.get("include_embedding"):
                        try:
                            embedding = (
                                json.loads(embedding_json)
                                if isinstance(embedding_json, str)
                                else embedding_json
                            )
                            properties["embedding"] = embedding
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(
                                "get_nodes failed to parse embedding for node %s", node_id
                            )
                    nodes.append(self._parse_node(properties))
                return nodes
        except Exception as e:
            logger.error("get_nodes failed: %s", e, exc_info=True)
            raise

    def get_neighbors(
        self, id: str, type: str, direction: Literal["in", "out", "both"] = "out"
    ) -> list[str]:
        raise NotImplementedError

    @timed
    def get_children_with_embeddings(
        self, id: str, user_name: str | None = None
    ) -> list[dict[str, Any]]:
        user_name = user_name if user_name else self._get_config_value("user_name")
        schema_raw = self._get_shard_schema_raw(user_name)
        tbl = self.get_memory_graph_table_name(user_name)
        where_user = f"AND p.user_name = '{user_name}' AND c.user_name = '{user_name}'"

        query = f"""
            WITH t as (
                SELECT *
                FROM cypher('{schema_raw}', $$
                MATCH (p:Memory)-[r:PARENT]->(c:Memory)
                WHERE p.id = '{id}' {where_user}
                RETURN id(c) as cid, c.id AS id, c.memory AS memory
                $$) as (cid agtype, id agtype, memory agtype)
                )
                SELECT t.id, m.embedding, t.memory FROM t,
                {tbl}."Memory" m
            WHERE t.cid::graphid = m.id;
        """

        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()

                children = []
                for row in results:
                    child_id_raw = row[0].value if hasattr(row[0], "value") else str(row[0])
                    if isinstance(child_id_raw, str):
                        if child_id_raw.startswith('"') and child_id_raw.endswith('"'):
                            child_id = child_id_raw[1:-1]
                        else:
                            child_id = child_id_raw
                    else:
                        child_id = str(child_id_raw)

                    embedding_raw = row[1]
                    embedding = []
                    if embedding_raw is not None:
                        try:
                            if isinstance(embedding_raw, str):
                                embedding = json.loads(embedding_raw)
                            elif isinstance(embedding_raw, list):
                                embedding = embedding_raw
                            else:
                                embedding = list(embedding_raw)
                        except (json.JSONDecodeError, TypeError, ValueError) as e:
                            logger.warning(
                                f"Failed to parse embedding for child node {child_id}: {e}"
                            )
                            embedding = []

                    memory_raw = row[2].value if hasattr(row[2], "value") else str(row[2])
                    if isinstance(memory_raw, str):
                        if memory_raw.startswith('"') and memory_raw.endswith('"'):
                            memory = memory_raw[1:-1]
                        else:
                            memory = memory_raw
                    else:
                        memory = str(memory_raw)

                    children.append({"id": child_id, "embedding": embedding, "memory": memory})

                return children

        except Exception as e:
            logger.error(f"[get_children_with_embeddings] Failed: {e}", exc_info=True)
            return []

    def get_path(self, source_id: str, target_id: str, max_depth: int = 3) -> list[str]:
        raise NotImplementedError

    def _build_subgraph_cypher(
        self, schema_raw: str, center_id: str, center_status: str, user_filter: str, depth: int
    ) -> str:
        if depth == 1:
            return (
                f"SELECT * FROM cypher('{schema_raw}', $$"
                f" MATCH(center:Memory)-[r]->(neighbor:Memory)"
                f" WHERE center.id = '{center_id}' AND center.status = '{center_status}' {user_filter}"
                f" RETURN collect(DISTINCT center), collect(DISTINCT neighbor), collect(DISTINCT r)"
                f" $$) AS (centers agtype, neighbors agtype, rels agtype)"
            )
        return (
            f"SELECT * FROM cypher('{schema_raw}', $$"
            f" MATCH(center:Memory)-[r]->(neighbor:Memory)"
            f" WHERE center.id = '{center_id}' AND center.status = '{center_status}' {user_filter}"
            f" RETURN collect(DISTINCT center), collect(DISTINCT neighbor), collect(DISTINCT r)"
            f" UNION ALL"
            f" MATCH(center:Memory)-[r]->(n:Memory)-[r1]->(neighbor:Memory)"
            f" WHERE center.id = '{center_id}' AND center.status = '{center_status}' {user_filter}"
            f" RETURN collect(DISTINCT center), collect(DISTINCT neighbor), collect(DISTINCT r1)"
            f" $$) AS (centers agtype, neighbors agtype, rels agtype)"
        )

    @timed
    def get_subgraph(
        self,
        center_id: str,
        depth: int = 2,
        center_status: str = "activated",
        user_name: str | None = None,
    ) -> dict[str, Any]:
        logger.info(
            "get_subgraph center_id=%s, depth=%s, center_status=%s, user_name=%s",
            center_id,
            depth,
            center_status,
            user_name,
        )
        if not 1 <= depth <= 5:
            raise ValueError("depth must be 1-5")

        resolved_user_name = user_name

        if center_id.startswith('"') and center_id.endswith('"'):
            center_id = center_id[1:-1]

        if resolved_user_name:
            schema_raw = self._get_shard_schema_raw(resolved_user_name)
            user_filter = f"AND center.user_name = '{resolved_user_name}'"
            query = self._build_subgraph_cypher(
                schema_raw, center_id, center_status, user_filter, depth
            )
        else:
            union_parts = []
            for i in range(self._shard_count):
                schema_raw = f"{self.db_name}_graph_{i}"
                query_part = self._build_subgraph_cypher(
                    schema_raw, center_id, center_status, "", depth
                )
                union_parts.append(query_part)
            query = " UNION ALL ".join(union_parts)

        logger.info("get_subgraph query=%s", query)
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()

                if not results:
                    return {"core_node": None, "neighbors": [], "edges": []}

                all_centers_list = []
                all_neighbors_list = []
                all_edges_list = []

                for result in results:
                    if not result or not result[0]:
                        continue

                    centers_data = result[0] if result[0] else "[]"
                    neighbors_data = result[1] if result[1] else "[]"
                    edges_data = result[2] if result[2] else "[]"

                    try:
                        if isinstance(centers_data, str):
                            centers_data = centers_data.replace("::vertex", "")
                        if isinstance(neighbors_data, str):
                            neighbors_data = neighbors_data.replace("::vertex", "")
                        if isinstance(edges_data, str):
                            edges_data = edges_data.replace("::edge", "")

                        centers_list = (
                            json.loads(centers_data)
                            if isinstance(centers_data, str)
                            else centers_data
                        )
                        neighbors_list = (
                            json.loads(neighbors_data)
                            if isinstance(neighbors_data, str)
                            else neighbors_data
                        )
                        edges_list = (
                            json.loads(edges_data) if isinstance(edges_data, str) else edges_data
                        )

                        if isinstance(centers_list, list):
                            all_centers_list.extend(centers_list)
                        if isinstance(neighbors_list, list):
                            all_neighbors_list.extend(neighbors_list)
                        if isinstance(edges_list, list):
                            all_edges_list.extend(edges_list)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON data: {e}")
                        continue

                centers_dict = {}
                for center_data in all_centers_list:
                    if isinstance(center_data, dict) and "properties" in center_data:
                        center_id_key = center_data["properties"].get("id")
                        if center_id_key and center_id_key not in centers_dict:
                            centers_dict[center_id_key] = center_data

                core_node = None
                if centers_dict:
                    center_data = next(iter(centers_dict.values()))
                    if isinstance(center_data, dict) and "properties" in center_data:
                        core_node = self._parse_node(center_data["properties"])

                neighbors_dict = {}
                for neighbor_data in all_neighbors_list:
                    if isinstance(neighbor_data, dict) and "properties" in neighbor_data:
                        neighbor_id = neighbor_data["properties"].get("id")
                        if neighbor_id and neighbor_id not in neighbors_dict:
                            neighbors_dict[neighbor_id] = neighbor_data

                neighbors = []
                for neighbor_data in neighbors_dict.values():
                    if isinstance(neighbor_data, dict) and "properties" in neighbor_data:
                        neighbor_parsed = self._parse_node(neighbor_data["properties"])
                        neighbors.append(neighbor_parsed)

                edges_dict = {}
                for edge_group in all_edges_list:
                    if isinstance(edge_group, list):
                        for edge_data in edge_group:
                            if isinstance(edge_data, dict):
                                edge_key = (
                                    edge_data.get("start_id", ""),
                                    edge_data.get("end_id", ""),
                                    edge_data.get("label", ""),
                                )
                                if edge_key not in edges_dict:
                                    edges_dict[edge_key] = {
                                        "type": edge_data.get("label", ""),
                                        "source": edge_data.get("start_id", ""),
                                        "target": edge_data.get("end_id", ""),
                                    }
                    elif isinstance(edge_group, dict):
                        edge_key = (
                            edge_group.get("start_id", ""),
                            edge_group.get("end_id", ""),
                            edge_group.get("label", ""),
                        )
                        if edge_key not in edges_dict:
                            edges_dict[edge_key] = {
                                "type": edge_group.get("label", ""),
                                "source": edge_group.get("start_id", ""),
                                "target": edge_group.get("end_id", ""),
                            }

                edges = list(edges_dict.values())

                return self._convert_graph_edges(
                    {"core_node": core_node, "neighbors": neighbors, "edges": edges}
                )

        except Exception as e:
            logger.error("get_subgraph failed: %s", e, exc_info=True)
            return {"core_node": None, "neighbors": [], "edges": []}

    def get_context_chain(self, id: str, type: str = "FOLLOWS") -> list[str]:
        raise NotImplementedError

    def _extract_fields_from_properties(
        self, properties: Any, return_fields: list[str]
    ) -> dict[str, Any]:
        result = {}
        return_fields = self._validate_return_fields(return_fields)
        if not properties or not return_fields:
            return result
        try:
            if isinstance(properties, str):
                props = json.loads(properties)
            elif isinstance(properties, dict):
                props = properties
            else:
                props = json.loads(str(properties))
        except (json.JSONDecodeError, TypeError, ValueError):
            return result
        for field in return_fields:
            if field != "id" and field in props:
                result[field] = props[field]
        return result

    @timed
    def search_by_keywords_like(
        self,
        query_word: str,
        scope: str | None = None,
        status: str | None = None,
        search_filter: dict | None = None,
        user_name: str | None = None,
        filter: dict | None = None,
        knowledgebase_ids: list[str] | None = None,
        return_fields: list[str] | None = None,
        **kwargs,
    ) -> list[dict]:
        logger.info(
            "search_by_keywords_like query_word=%s, scope=%s, user_name=%s",
            query_word,
            scope,
            user_name,
        )
        if not user_name:
           return []
        resolved_user_name = user_name

        where_clauses = []

        if scope:
            where_clauses.append(
                f"ag_catalog.agtype_access_operator(properties, '\"memory_type\"'::agtype) = '\"{scope}\"'::agtype"
            )
        if status:
            where_clauses.append(
                f"ag_catalog.agtype_access_operator(properties, '\"status\"'::agtype) = '\"{status}\"'::agtype"
            )
        else:
            where_clauses.append(
                "ag_catalog.agtype_access_operator(properties, '\"status\"'::agtype) = '\"activated\"'::agtype"
            )

        user_name_conditions = self._build_user_name_and_kb_ids_conditions_sql(
            user_name=resolved_user_name,
            knowledgebase_ids=knowledgebase_ids,
        )

        if user_name_conditions:
            if len(user_name_conditions) == 1:
                where_clauses.append(user_name_conditions[0])
            else:
                where_clauses.append(f"({' OR '.join(user_name_conditions)})")

        if search_filter:
            for key, value in search_filter.items():
                if isinstance(value, str):
                    where_clauses.append(
                        f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) = '\"{value}\"'::agtype"
                    )
                else:
                    where_clauses.append(
                        f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) = {value}::agtype"
                    )

        filter_conditions = self._build_filter_conditions_sql(filter)
        where_clauses.extend(filter_conditions)

        where_clauses.append("""(properties -> '"memory"')::text LIKE %s""")
        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        select_clause = (
            "SELECT ag_catalog.agtype_access_operator(properties, '\"id\"'::agtype) AS old_id,"
            " agtype_object_field_text(properties, 'memory') as memory_text"
        )
        if return_fields:
            select_clause += ", properties"

        like_pattern = f"%{query_word}%"
        if resolved_user_name:
            tbl = self.get_memory_graph_table_name(resolved_user_name)
            query = f'{select_clause} FROM {tbl}."Memory" {where_clause}'
            params = (like_pattern,)
        else:
            union_parts = [
                f'{select_clause} FROM {tbl}."Memory" {where_clause}'
                for tbl in self._get_all_shard_table_names()
            ]
            query = " UNION ALL ".join(union_parts)
            params = tuple(like_pattern for _ in union_parts)

        logger.info("search_by_keywords_like query=%s, params=%s", query, params)
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
                output = []
                for row in results:
                    oldid = row[0]
                    id_val = str(oldid)
                    if id_val.startswith('"') and id_val.endswith('"'):
                        id_val = id_val[1:-1]
                    item = {"id": id_val}
                    if return_fields:
                        properties = row[2]
                        item.update(self._extract_fields_from_properties(properties, return_fields))
                    output.append(item)
                logger.info("search_by_keywords_like recalled %d results", len(output))
                return output
        except Exception as e:
            logger.error("search_by_keywords_like failed: %s", e, exc_info=True)
            raise

    @timed
    def search_by_keywords_tfidf(
        self,
        query_words: list[str],
        scope: str | None = None,
        status: str | None = None,
        search_filter: dict | None = None,
        user_name: str | None = None,
        filter: dict | None = None,
        knowledgebase_ids: list[str] | None = None,
        tsvector_field: str = "properties_tsvector_zh",
        tsquery_config: str = "jiebaqry",
        return_fields: list[str] | None = None,
        **kwargs,
    ) -> list[dict]:
        logger.info(
            "search_by_keywords_tfidf query_words=%s, scope=%s, user_name=%s,filter=%s",
            query_words,
            scope,
            user_name,
            filter,
        )
        if not user_name:
           return []
        resolved_user_name = user_name

        where_clauses = []

        if scope:
            where_clauses.append(
                f"ag_catalog.agtype_access_operator(properties, '\"memory_type\"'::agtype) = '\"{scope}\"'::agtype"
            )
        if status:
            where_clauses.append(
                f"ag_catalog.agtype_access_operator(properties, '\"status\"'::agtype) = '\"{status}\"'::agtype"
            )
        else:
            where_clauses.append(
                "ag_catalog.agtype_access_operator(properties, '\"status\"'::agtype) = '\"activated\"'::agtype"
            )

        user_name_conditions = self._build_user_name_and_kb_ids_conditions_sql(
            user_name=resolved_user_name,
            knowledgebase_ids=knowledgebase_ids,
        )

        if user_name_conditions:
            if len(user_name_conditions) == 1:
                where_clauses.append(user_name_conditions[0])
            else:
                where_clauses.append(f"({' OR '.join(user_name_conditions)})")

        if search_filter:
            for key, value in search_filter.items():
                if isinstance(value, str):
                    where_clauses.append(
                        f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) = '\"{value}\"'::agtype"
                    )
                else:
                    where_clauses.append(
                        f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) = {value}::agtype"
                    )

        filter_conditions = self._build_filter_conditions_sql(filter)
        where_clauses.extend(filter_conditions)
        tsquery_string = " | ".join(query_words)

        where_clauses.append(f"{tsvector_field} @@ to_tsquery('{tsquery_config}', %s)")

        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        select_clause = (
            "SELECT ag_catalog.agtype_access_operator(properties, '\"id\"'::agtype) AS old_id,"
            " agtype_object_field_text(properties, 'memory') as memory_text"
        )
        if return_fields:
            select_clause += ", properties"

        if resolved_user_name:
            tbl = self.get_memory_graph_table_name(resolved_user_name)
            query = f'{select_clause} FROM {tbl}."Memory" {where_clause}'
            params = (tsquery_string,)
        else:
            union_parts = [
                f'{select_clause} FROM {tbl}."Memory" {where_clause}'
                for tbl in self._get_all_shard_table_names()
            ]
            query = " UNION ALL ".join(union_parts)
            params = tuple(tsquery_string for _ in union_parts)

        logger.info("search_by_keywords_tfidf query=%s, params=%s", query, params)
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
                output = []
                for row in results:
                    oldid = row[0]
                    id_val = str(oldid)
                    if id_val.startswith('"') and id_val.endswith('"'):
                        id_val = id_val[1:-1]
                    item = {"id": id_val}
                    if return_fields:
                        properties = row[2]
                        item.update(self._extract_fields_from_properties(properties, return_fields))
                    output.append(item)
                logger.info("search_by_keywords_tfidf recalled %d results", len(output))
                return output
        except Exception as e:
            logger.error("search_by_keywords_tfidf failed: %s", e, exc_info=True)
            raise

    @timed
    def search_by_fulltext(
        self,
        query_words: list[str],
        top_k: int = 10,
        scope: str | None = None,
        status: str | None = None,
        threshold: float | None = None,
        search_filter: dict | None = None,
        user_name: str | None = None,
        filter: dict | None = None,
        knowledgebase_ids: list[str] | None = None,
        tsvector_field: str = "properties_tsvector_zh",
        tsquery_config: str = "jiebacfg",
        return_fields: list[str] | None = None,
        **kwargs,
    ) -> list[dict]:
        resolved_user_name = user_name

        start_time = time.perf_counter()
        logger.info(
            "search_by_fulltext query_words=%s, top_k=%s, scope=%s, user_name=%s,filter=%s",
            query_words,
            top_k,
            scope,
            resolved_user_name,
            filter,
        )
        where_clauses = []

        if scope:
            where_clauses.append(
                f"ag_catalog.agtype_access_operator(properties, '\"memory_type\"'::agtype) = '\"{scope}\"'::agtype"
            )
        if status:
            where_clauses.append(
                f"ag_catalog.agtype_access_operator(properties, '\"status\"'::agtype) = '\"{status}\"'::agtype"
            )
        else:
            where_clauses.append(
                "ag_catalog.agtype_access_operator(properties, '\"status\"'::agtype) = '\"activated\"'::agtype"
            )

        user_name_conditions = self._build_user_name_and_kb_ids_conditions_sql(
            user_name=resolved_user_name,
            knowledgebase_ids=knowledgebase_ids,
        )

        if user_name_conditions:
            if len(user_name_conditions) == 1:
                where_clauses.append(user_name_conditions[0])
            else:
                where_clauses.append(f"({' OR '.join(user_name_conditions)})")

        if search_filter:
            for key, value in search_filter.items():
                if isinstance(value, str):
                    where_clauses.append(
                        f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) = '\"{value}\"'::agtype"
                    )
                else:
                    where_clauses.append(
                        f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) = {value}::agtype"
                    )

        filter_conditions = self._build_filter_conditions_sql(filter)
        where_clauses.extend(filter_conditions)
        tsquery_string = " & ".join(query_words)

        where_clauses.append(f"{tsvector_field} @@ to_tsquery('{tsquery_config}', %s)")

        select_cols = (
            f"ag_catalog.agtype_access_operator(m.properties, '\"id\"'::agtype) AS old_id,"
            f" ts_rank(m.{tsvector_field}, q.fq) AS rank"
        )
        if return_fields:
            select_cols += ", m.properties"

        where_with_q = []
        for w in where_clauses:
            if f"{tsvector_field} @@ to_tsquery(" in w:
                where_with_q.append(f"m.{tsvector_field} @@ q.fq")
            else:
                where_with_q.append(
                    w.replace("(properties,", "(m.properties,")
                    .replace("(properties)", "(m.properties)")
                    .replace("ARRAY[properties,", "ARRAY[m.properties,")
                )
        where_clause_cte = f"WHERE {' AND '.join(where_with_q)}" if where_with_q else ""

        if resolved_user_name:
            tbl = self.get_memory_graph_table_name(resolved_user_name)
            query = (
                f"/*+ Set(max_parallel_workers_per_gather 0) */"
                f" WITH q AS (SELECT to_tsquery('{tsquery_config}', %s) AS fq)"
                f" SELECT {select_cols}"
                f' FROM {tbl}."Memory" m CROSS JOIN q'
                f" {where_clause_cte}"
                f" ORDER BY rank DESC"
                f" LIMIT {top_k}"
            )
            params = [tsquery_string]
        else:
            shard_selects = []
            for tbl in self._get_all_shard_table_names():
                part = f'SELECT {select_cols} FROM {tbl}."Memory" m CROSS JOIN q {where_clause_cte}'
                shard_selects.append(part)
            inner_union = " UNION ALL ".join(shard_selects)
            query = (
                f"/*+ Set(max_parallel_workers_per_gather 0) */"
                f" WITH q AS (SELECT to_tsquery('{tsquery_config}', %s) AS fq)"
                f" {inner_union}"
                f" ORDER BY rank DESC"
                f" LIMIT {top_k}"
            )
            params = [tsquery_string]

        logger.info("search_by_fulltext query=%s, params=%s", query, params)
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
                output = []
                for row in results:
                    oldid = row[0]
                    rank = row[1]

                    id_val = str(oldid)
                    if id_val.startswith('"') and id_val.endswith('"'):
                        id_val = id_val[1:-1]
                    score_val = float(rank)

                    if threshold is None or score_val >= threshold:
                        item = {"id": id_val, "score": score_val}
                        if return_fields:
                            properties = row[2]
                            item.update(
                                self._extract_fields_from_properties(properties, return_fields)
                            )
                        output.append(item)
                elapsed = (time.perf_counter() - start_time) * 1000.0
                logger.info(
                    "search_by_fulltext recalled %d results, took %.1f ms", len(output), elapsed
                )
                return output[:top_k]
        except Exception as e:
            logger.error("search_by_fulltext failed: %s", e, exc_info=True)
            raise

    @timed
    def search_by_embedding(
        self,
        vector: list[float],
        user_name: str,
        top_k: int = 5,
        scope: str | None = None,
        status: str | None = None,
        threshold: float | None = None,
        search_filter: dict | None = None,
        filter: dict | None = None,
        knowledgebase_ids: list[str] | None = None,
        return_fields: list[str] | None = None,
        **kwargs,
    ) -> list[dict]:
        logger.info(
            "search_by_embedding by user_name:%s,knowledgebase_ids: %s,scope:%s,status:%s,search_filter:%s,filter:%s,knowledgebase_ids:%s,return_fields:%s",
            user_name,
            knowledgebase_ids,
            scope,
            status,
            search_filter,
            filter,
            knowledgebase_ids,
            return_fields,
        )
        tbl = self.get_memory_graph_table_name(user_name)

        start_time = time.perf_counter()
        where_clauses = []
        if scope:
            where_clauses.append(
                f"ag_catalog.agtype_access_operator(properties, '\"memory_type\"'::agtype) = '\"{scope}\"'::agtype"
            )
        if status:
            where_clauses.append(
                f"ag_catalog.agtype_access_operator(properties, '\"status\"'::agtype) = '\"{status}\"'::agtype"
            )
        else:
            where_clauses.append(
                "ag_catalog.agtype_access_operator(properties, '\"status\"'::agtype) = '\"activated\"'::agtype"
            )
        where_clauses.append("embedding is not null")
        user_name_conditions = self._build_user_name_and_kb_ids_conditions_sql(
            user_name=user_name,
            knowledgebase_ids=knowledgebase_ids,
        )

        if user_name_conditions:
            if len(user_name_conditions) == 1:
                where_clauses.append(user_name_conditions[0])
            else:
                where_clauses.append(f"({' OR '.join(user_name_conditions)})")

        if search_filter:
            for key, value in search_filter.items():
                if isinstance(value, str):
                    where_clauses.append(
                        f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) = '\"{value}\"'::agtype"
                    )
                else:
                    where_clauses.append(
                        f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) = {value}::agtype"
                    )

        filter_conditions = self._build_filter_conditions_sql(filter)
        where_clauses.extend(filter_conditions)

        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        query = f"""
                    set hnsw.ef_search = 100;set hnsw.iterative_scan = relaxed_order;
                    WITH t AS (
                        SELECT id,
                               properties,
                               timeline,
                               ag_catalog.agtype_access_operator(properties, '"id"'::agtype) AS old_id,
                               (embedding <=> %s::vector(1024)) AS scope_distance
                        FROM {tbl}."Memory"
                        {where_clause}
                        ORDER BY scope_distance ASC
                        LIMIT {top_k}
                    )
                    SELECT *,(1 - scope_distance) AS scope
                    FROM t
                    WHERE scope_distance < 0.9;
                """
        vector_str = convert_to_vector(vector)
        query = query.replace("%s::vector(1024)", f"'{vector_str}'::vector(1024)")
        params = []

        query_lines = query.strip().split("\n")
        for line in query_lines:
            if len(line) > 200:
                wrapped_lines = textwrap.wrap(
                    line, width=200, break_long_words=False, break_on_hyphens=False
                )
                for _wrapped_line in wrapped_lines:
                    pass
            else:
                pass

        logger.info(" search_by_embedding query: %s", query)

        with self._get_connection() as conn, conn.cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            results = cursor.fetchall()
            output = []
            for row in results:
                if len(row) < 5:
                    logger.warning(f"Row has {len(row)} columns, expected 5. Row: {row}")
                    continue
                oldid = row[3]
                score = row[4]
                id_val = str(oldid)
                if id_val.startswith('"') and id_val.endswith('"'):
                    id_val = id_val[1:-1]
                score_val = float(score)
                score_val = (score_val + 1) / 2
                if threshold is None or score_val >= threshold:
                    item = {"id": id_val, "score": score_val}
                    if return_fields:
                        properties = row[1]
                        item.update(self._extract_fields_from_properties(properties, return_fields))
                    output.append(item)
            elapsed_time = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                "search_by_embedding query by embedding completed time took %.1f ms", elapsed_time
            )
            return output[:top_k]

    @timed
    def get_by_metadata(
        self,
        filters: list[dict[str, Any]],
        user_name: str | None = None,
        filter: dict | None = None,
        knowledgebase_ids: list | None = None,
        user_name_flag: bool = True,
        **kwargs,
    ) -> list[str]:
        start_time = time.perf_counter()
        resolved_user_name = user_name
        logger.info(
            "get_by_metadata user_name=%s, filter=%s, knowledgebase_ids=%s, filters=%s",
            resolved_user_name,
            filter,
            knowledgebase_ids,
            filters,
        )

        if not resolved_user_name:
            raise ValueError("get_by_metadata requires user_name && user_name is not null ")

        where_conditions = []

        for f in filters:
            field = f["field"]
            op = f.get("op", "=")
            value = f["value"]

            if isinstance(value, str):
                escaped_str = value.replace("'", "\\'")
                escaped_value = f"'{escaped_str}'"
            elif isinstance(value, list):
                list_items = []
                for v in value:
                    if isinstance(v, str):
                        escaped_str = v.replace('"', '\\"')
                        list_items.append(f'"{escaped_str}"')
                    else:
                        list_items.append(str(v))
                escaped_value = f"[{', '.join(list_items)}]"
            else:
                escaped_value = f"'{value}'" if isinstance(value, str) else str(value)
            if op == "=":
                where_conditions.append(f"n.{field} = {escaped_value}")
            elif op == "in":
                where_conditions.append(f"n.{field} IN {escaped_value}")
            elif op == "contains":
                where_conditions.append(f"{escaped_value} IN n.{field}")
            elif op == "starts_with":
                where_conditions.append(f"n.{field} STARTS WITH {escaped_value}")
            elif op == "ends_with":
                where_conditions.append(f"n.{field} ENDS WITH {escaped_value}")
            elif op == "like":
                where_conditions.append(f"n.{field} CONTAINS {escaped_value}")
            elif op in [">", ">=", "<", "<="]:
                where_conditions.append(f"n.{field} {op} {escaped_value}")
            else:
                raise ValueError(f"Unsupported operator: {op}")

        user_name_conditions = self._build_user_name_and_kb_ids_conditions_cypher(
            user_name=resolved_user_name,
            knowledgebase_ids=knowledgebase_ids,
        )
        logger.info("get_by_metadata user_name_conditions=%s", user_name_conditions)

        if user_name_conditions:
            if len(user_name_conditions) == 1:
                where_conditions.append(user_name_conditions[0])
            else:
                where_conditions.append(f"({' OR '.join(user_name_conditions)})")

        filter_where_clause = self._build_filter_conditions_cypher(filter)
        logger.info("get_by_metadata filter_where_clause=%s", filter_where_clause)

        where_str = " AND ".join(where_conditions) + filter_where_clause

        if resolved_user_name:
            schema_raw = self._get_shard_schema_raw(resolved_user_name)
            target_shards = [schema_raw]
        else:
            target_shards = [f"{self.db_name}_graph_{i}" for i in range(self._shard_count)]

        ids = []
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                for shard in target_shards:
                    cypher_query = f"""
                        SELECT * FROM cypher('{shard}', $$
                        MATCH (n:Memory)
                        WHERE {where_str}
                        RETURN n.id AS id
                        $$) AS (id agtype)
                    """
                    logger.info("get_by_metadata shard=%s, cypher_query=%s", shard, cypher_query)
                    cursor.execute(cypher_query)
                    results = cursor.fetchall()
                    ids.extend(str(item[0]).strip('"') for item in results)
        except Exception as e:
            logger.warning("get_by_metadata failed: %s", e, exc_info=True)
        elapsed = (time.perf_counter() - start_time) * 1000.0
        logger.info("get_by_metadata recalled %d ids, took %.1f ms", len(ids), elapsed)
        return ids

    @timed
    def get_grouped_counts(
        self,
        group_fields: list[str],
        where_clause: str = "",
        params: dict[str, Any] | None = None,
        user_name: str | None = None,
    ) -> list[dict[str, Any]]:
        start_time = time.perf_counter()
        logger.info(
            "get_grouped_counts group_fields=%s, where_clause=%s, params=%s, user_name=%s",
            group_fields,
            where_clause,
            params,
            user_name,
        )
        if not group_fields:
            raise ValueError("group_fields cannot be empty")

        resolved_user_name = user_name

        effective_where = where_clause.strip() if where_clause else ""

        if resolved_user_name:
            user_clause = (
                f"ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype)"
                f" = '\"{resolved_user_name}\"'::agtype"
            )
            if effective_where:
                if effective_where.upper().startswith("WHERE"):
                    effective_where += f" AND {user_clause}"
                else:
                    effective_where = f"WHERE {effective_where} AND {user_clause}"
            else:
                effective_where = f"WHERE {user_clause}"

        if params and isinstance(params, dict):
            for key, value in params.items():
                if isinstance(value, str):
                    value = f"'{value}'"
                effective_where = effective_where.replace(f"${key}", str(value))

        if "user_name = %s" in effective_where and resolved_user_name:
            effective_where = effective_where.replace(
                "user_name = %s",
                f"ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype)"
                f" = '\"{resolved_user_name}\"'::agtype",
            )

        cte_select_list = []
        aliases = []
        for field in group_fields:
            alias = field.replace(".", "_")
            aliases.append(alias)
            cte_select_list.append(
                f"ag_catalog.agtype_access_operator(properties, '\"{field}\"'::agtype) AS {alias}"
            )
        cte_cols = ", ".join(cte_select_list)
        outer_select = ", ".join(f"{a}::text" for a in aliases)
        outer_group_by = ", ".join(aliases)

        if resolved_user_name:
            tbl = self.get_memory_graph_table_name(resolved_user_name)
            query = (
                f"WITH t AS ("
                f' SELECT {cte_cols} FROM {tbl}."Memory" {effective_where} LIMIT 100'
                f") SELECT {outer_select}, count(*) AS count FROM t GROUP BY {outer_group_by}"
            )
        else:
            if not effective_where:
                effective_where_inner = ""
            elif effective_where.upper().startswith("WHERE"):
                effective_where_inner = effective_where
            else:
                effective_where_inner = f"WHERE {effective_where}"

            union_parts = [
                f'SELECT {cte_cols} FROM {tbl}."Memory" {effective_where_inner}'
                for tbl in self._get_all_shard_table_names()
            ]
            query = (
                f"WITH t AS ({' UNION ALL '.join(union_parts)} LIMIT 100)"
                f" SELECT {outer_select}, count(*) AS count FROM t GROUP BY {outer_group_by}"
            )

        logger.info("get_grouped_counts query=%s", query)

        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
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
                        if hasattr(value, "value"):
                            group_values[field] = value.value
                        else:
                            group_values[field] = str(value)
                    count_value = row[-1]
                    output.append({**group_values, "count": int(count_value)})

                elapsed = (time.perf_counter() - start_time) * 1000.0
                logger.info("get_grouped_counts took %.1f ms, results=%d", elapsed, len(output))
                return output

        except Exception as e:
            logger.error("get_grouped_counts failed: %s", e, exc_info=True)
            return []

    def deduplicate_nodes(self) -> None:
        raise NotImplementedError

    def detect_conflicts(self) -> list[tuple[str, str]]:
        raise NotImplementedError

    def merge_nodes(self, id1: str, id2: str) -> str:
        raise NotImplementedError

    @timed
    def clear(self, user_name: str | None = None) -> None:
        user_name = user_name if user_name else self._get_config_value("user_name")
        schema_raw = self._get_shard_schema_raw(user_name)

        try:
            query = f"""
                SELECT * FROM cypher('{schema_raw}', $$
                MATCH (n:Memory)
                WHERE n.user_name = '{user_name}'
                DETACH DELETE n
                $$) AS (result agtype)
            """
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(query)
                logger.info("Cleared all nodes from database.")

        except Exception as e:
            logger.error(f"[ERROR] Failed to clear database: {e}")

    @timed
    def export_graph(
        self,
        user_name: str | None = None,
        include_embedding: bool = False,
        user_id: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
        filter: dict | None = None,
        memory_type: list[str] | None = None,
        status: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        start_time = time.perf_counter()
        logger.info(
            "export_graph include_embedding=%s, user_name=%s, user_id=%s, page=%s, page_size=%s, filter=%s, memory_type=%s, status=%s",
            include_embedding,
            user_name,
            user_id,
            page,
            page_size,
            filter,
            memory_type,
            status,
        )
        resolved_user_name = user_name
        user_id = user_id if user_id else self._get_config_value("user_id")

        extracted_object_type: str | None = None
        extracted_mem_cube_id: str | None = None

        def _extract_special_filter_values(filter_obj):
            nonlocal extracted_object_type, extracted_mem_cube_id

            if isinstance(filter_obj, dict):
                if "and" in filter_obj and isinstance(filter_obj["and"], list):
                    cleaned_items = []
                    for item in filter_obj["and"]:
                        cleaned_item = _extract_special_filter_values(item)
                        if cleaned_item not in (None, {}, []):
                            cleaned_items.append(cleaned_item)
                    return {"and": cleaned_items} if cleaned_items else None

                if "or" in filter_obj and isinstance(filter_obj["or"], list):
                    cleaned_items = []
                    for item in filter_obj["or"]:
                        cleaned_item = _extract_special_filter_values(item)
                        if cleaned_item not in (None, {}, []):
                            cleaned_items.append(cleaned_item)
                    return {"or": cleaned_items} if cleaned_items else None

                cleaned_dict = {}
                for key, value in filter_obj.items():
                    if key == "object_type" and isinstance(value, str):
                        if extracted_object_type is None:
                            extracted_object_type = value
                        continue
                    if key == "mem_cube_id" and isinstance(value, str):
                        if extracted_mem_cube_id is None:
                            extracted_mem_cube_id = value
                        continue
                    cleaned_dict[key] = value
                return cleaned_dict if cleaned_dict else None

            return filter_obj

        filter_for_sql = _extract_special_filter_values(filter)

        total_nodes = 0
        total_edges = 0

        use_pagination = page is not None and page_size is not None

        if use_pagination:
            if page < 1:
                page = 1
            if page_size < 1:
                page_size = 10
            offset = (page - 1) * page_size
        else:
            offset = None

        where_conditions = []
        has_object_type_filter = (
            isinstance(extracted_object_type, str)
            and isinstance(extracted_mem_cube_id, str)
            and extracted_mem_cube_id.strip() != ""
        )

        if resolved_user_name and not has_object_type_filter:
            where_conditions.append(
                f"ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = '\"{resolved_user_name}\"'::agtype"
            )

        if has_object_type_filter:
            object_type_value = extracted_object_type.strip().lower()
            escaped_mem_cube_id = extracted_mem_cube_id.replace("'", "''")
            if object_type_value == "user":
                where_conditions.append(
                    f"ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) <> '\"{escaped_mem_cube_id}\"'::agtype"
                )
            elif object_type_value == "public":
                where_conditions.append(
                    f"ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = '\"{escaped_mem_cube_id}\"'::agtype"
                )

        if user_id:
            where_conditions.append(
                f"ag_catalog.agtype_access_operator(properties, '\"user_id\"'::agtype) = '\"{user_id}\"'::agtype"
            )

        if memory_type and isinstance(memory_type, list) and len(memory_type) > 0:
            memory_type_values = []
            for mt in memory_type:
                escaped_memory_type = str(mt).replace("'", "''")
                memory_type_values.append(f"'\"{escaped_memory_type}\"'::agtype")
            memory_type_in_clause = ", ".join(memory_type_values)
            where_conditions.append(
                f"ag_catalog.agtype_access_operator(properties, '\"memory_type\"'::agtype) IN ({memory_type_in_clause})"
            )

        if status is None:
            where_conditions.append(
                "ag_catalog.agtype_access_operator(properties, '\"status\"'::agtype) <> '\"deleted\"'::agtype"
            )
        elif isinstance(status, list) and len(status) > 0:
            status_values = []
            for st in status:
                escaped_status = str(st).replace("'", "''")
                status_values.append(f"'\"{escaped_status}\"'::agtype")
            status_in_clause = ", ".join(status_values)
            where_conditions.append(
                f"ag_catalog.agtype_access_operator(properties, '\"status\"'::agtype) IN ({status_in_clause})"
            )

        filter_conditions = self._build_filter_conditions_sql(filter_for_sql)
        logger.info("export_graph filter_conditions=%s", filter_conditions)
        if filter_conditions:
            where_conditions.extend(filter_conditions)

        where_clause = ""
        if where_conditions:
            where_clause = f"WHERE {' AND '.join(where_conditions)}"

        pagination_clause = ""
        if use_pagination:
            pagination_clause = f"LIMIT {page_size} OFFSET {offset}"

        order_clause = (
            " ORDER BY ag_catalog.agtype_access_operator(properties, '\"created_at\"'::agtype)"
            " DESC NULLS LAST, id DESC"
        )

        select_cols = "id, properties, embedding" if include_embedding else "id, properties"

        if resolved_user_name:
            tbl = self.get_memory_graph_table_name(resolved_user_name)
            count_query = f'SELECT COUNT(*) AS total_count FROM {tbl}."Memory" {where_clause}'
            data_query = (
                f'SELECT {select_cols} FROM {tbl}."Memory"'
                f" {where_clause} {order_clause} {pagination_clause}"
            )
        else:
            count_parts = [
                f'SELECT COUNT(*) AS cnt FROM {tbl}."Memory" {where_clause}'
                for tbl in self._get_all_shard_table_names()
            ]
            count_query = f"SELECT SUM(cnt) FROM ({' UNION ALL '.join(count_parts)}) t"

            data_parts = [
                f'SELECT {select_cols} FROM {tbl}."Memory" {where_clause}'
                for tbl in self._get_all_shard_table_names()
            ]
            data_query = (
                f"SELECT {select_cols} FROM ({' UNION ALL '.join(data_parts)}) t"
                f" {order_clause} {pagination_clause}"
            )

        logger.info("export_graph count_query=%s", count_query)
        logger.info("export_graph data_query=%s", data_query)

        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(count_query)
                count_row = cursor.fetchone()
                total_nodes = int(count_row[0]) if count_row and count_row[0] is not None else 0

                cursor.execute(data_query)
                node_results = cursor.fetchall()
            nodes = []

            for row in node_results:
                if include_embedding:
                    row_id, properties_json, embedding_json = row
                else:
                    row_id, properties_json = row
                    embedding_json = None

                if row_id is None:
                    continue

                if isinstance(properties_json, str):
                    try:
                        properties = json.loads(properties_json)
                    except json.JSONDecodeError:
                        properties = {}
                else:
                    properties = properties_json if properties_json else {}

                if not include_embedding:
                    properties.pop("embedding", None)
                elif include_embedding and embedding_json is not None:
                    properties["embedding"] = embedding_json

                nodes.append(self._parse_node(properties))

        except Exception as e:
            logger.error("export_graph nodes failed: %s", e, exc_info=True)
            raise RuntimeError(f"export_graph nodes failed: {e}") from e
        elapsed = (time.perf_counter() - start_time) * 1000.0
        logger.info("export_graph took %.1f ms, total_nodes=%d", elapsed, total_nodes)

        edges = []
        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
        }

    @timed
    def count_nodes(self, scope: str, user_name: str | None = None) -> int:
        user_name = user_name if user_name else self.config.user_name
        schema_raw = self._get_shard_schema_raw(user_name)

        query = f"""
            SELECT * FROM cypher('{schema_raw}', $$
                MATCH (n:Memory)
                WHERE n.memory_type = '{scope}'
                AND n.user_name = '{user_name}'
                RETURN count(n)
            $$) AS (count agtype)
        """
        with self._get_connection() as conn:
            result = self.execute_query(query, conn)
            return int(result.one_or_none()["count"].value)

    @timed
    def get_all_memory_items(
        self,
        scope: str,
        user_name: str | None = None,
        include_embedding: bool = False,
        filter: dict | None = None,
        knowledgebase_ids: list | None = None,
        status: str | None = None,
    ) -> list[dict]:
        logger.info(
            "get_all_memory_items scope=%s, user_name=%s, filter=%s, knowledgebase_ids=%s, status=%s",
            scope,
            user_name,
            filter,
            knowledgebase_ids,
            status,
        )

        resolved_user_name = user_name
        if scope not in {"WorkingMemory", "LongTermMemory", "UserMemory", "OuterMemory"}:
            raise ValueError(f"Unsupported memory type scope: {scope}")

        user_name_conditions = self._build_user_name_and_kb_ids_conditions_cypher(
            user_name=resolved_user_name,
            knowledgebase_ids=knowledgebase_ids,
        )

        if user_name_conditions:
            if len(user_name_conditions) == 1:
                user_name_where = user_name_conditions[0]
            else:
                user_name_where = f"({' OR '.join(user_name_conditions)})"
        else:
            user_name_where = ""

        filter_where_clause = self._build_filter_conditions_cypher(filter)
        logger.info("get_all_memory_items filter_where_clause=%s", filter_where_clause)

        where_parts = [f"n.memory_type = '{scope}'"]
        if status:
            where_parts.append(f"n.status = '{status}'")
        if user_name_where:
            where_parts.append(user_name_where)
        if filter_where_clause:
            where_clause = " AND ".join(where_parts) + filter_where_clause
        else:
            where_clause = " AND ".join(where_parts)

        if resolved_user_name:
            shard_schemas = [self._get_shard_schema_raw(resolved_user_name)]
        else:
            shard_schemas = [f"{self.db_name}_graph_{i}" for i in range(self._shard_count)]

        if include_embedding:
            if len(shard_schemas) == 1:
                sr = shard_schemas[0]
                cypher_query = (
                    f"WITH t AS ("
                    f" SELECT * FROM cypher('{sr}', $$"
                    f" MATCH (n:Memory) WHERE {where_clause}"
                    f" RETURN id(n) as id1, n LIMIT 100"
                    f" $$) AS (id1 agtype, n agtype)"
                    f') SELECT m.embedding, t.n FROM t, {sr}."Memory" m WHERE t.id1 = m.id'
                )
            else:
                union_parts = []
                for sr in shard_schemas:
                    part = (
                        f"SELECT m.embedding, t.n FROM ("
                        f" SELECT * FROM cypher('{sr}', $$"
                        f" MATCH (n:Memory) WHERE {where_clause}"
                        f" RETURN id(n) as id1, n LIMIT 100"
                        f" $$) AS (id1 agtype, n agtype)"
                        f') t, {sr}."Memory" m WHERE t.id1 = m.id'
                    )
                    union_parts.append(part)
                cypher_query = " UNION ALL ".join(union_parts)

            nodes = []
            node_ids = set()
            logger.info("get_all_memory_items cypher_query=%s", cypher_query)
            try:
                with self._get_connection() as conn, conn.cursor() as cursor:
                    cursor.execute(cypher_query)
                    results = cursor.fetchall()

                    for row in results:
                        if isinstance(row, list | tuple) and len(row) >= 2:
                            embedding_val, node_val = row[0], row[1]
                        else:
                            embedding_val, node_val = None, row[0]

                        node = self._build_node_from_agtype(node_val, embedding_val)
                        if node:
                            node_id = node["id"]
                            if node_id not in node_ids:
                                nodes.append(node)
                                node_ids.add(node_id)

            except Exception as e:
                logger.warning("get_all_memory_items failed: %s", e, exc_info=True)

            return nodes
        else:
            if len(shard_schemas) == 1:
                sr = shard_schemas[0]
                cypher_query = (
                    f"SELECT * FROM cypher('{sr}', $$"
                    f" MATCH (n:Memory) WHERE {where_clause}"
                    f" RETURN properties(n) as props LIMIT 100"
                    f" $$) AS (nprops agtype)"
                )
            else:
                union_parts = []
                for sr in shard_schemas:
                    part = (
                        f"SELECT * FROM cypher('{sr}', $$"
                        f" MATCH (n:Memory) WHERE {where_clause}"
                        f" RETURN properties(n) as props LIMIT 100"
                        f" $$) AS (nprops agtype)"
                    )
                    union_parts.append(part)
                cypher_query = " UNION ALL ".join(union_parts)

            nodes = []
            logger.info("get_all_memory_items cypher_query=%s", cypher_query)
            try:
                with self._get_connection() as conn, conn.cursor() as cursor:
                    cursor.execute(cypher_query)
                    results = cursor.fetchall()

                    for row in results:
                        memory_data = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                        nodes.append(self._parse_node(memory_data))

            except Exception as e:
                logger.error("get_all_memory_items failed: %s", e, exc_info=True)

            return nodes

    @timed
    def get_structure_optimization_candidates(
        self, scope: str, include_embedding: bool = False, user_name: str | None = None
    ) -> list[dict]:
        logger.info(
            "get_structure_optimization_candidates scope=%s, include_embedding=%s, user_name=%s",
            scope,
            include_embedding,
            user_name,
        )
        resolved_user_name = user_name
        if not resolved_user_name:
            raise ValueError(
                "get_structure_optimization_candidates requires user_name && user_name is not null "
            )

        if include_embedding:
            return_fields = "id(n) as id1,n"
            return_fields_agtype = " id1 agtype,n agtype"
        else:
            return_fields = ",".join(
                [
                    "n.id AS id",
                    "n.memory AS memory",
                    "n.user_name AS user_name",
                    "n.user_id AS user_id",
                    "n.session_id AS session_id",
                    "n.status AS status",
                    "n.key AS key",
                    "n.confidence AS confidence",
                    "n.tags AS tags",
                    "n.created_at AS created_at",
                    "n.updated_at AS updated_at",
                    "n.memory_type AS memory_type",
                    "n.sources AS sources",
                    "n.source AS source",
                    "n.node_type AS node_type",
                    "n.visibility AS visibility",
                    "n.usage AS usage",
                    "n.background AS background",
                    "n.graph_id as graph_id",
                ]
            )
            fields = [
                "id",
                "memory",
                "user_name",
                "user_id",
                "session_id",
                "status",
                "key",
                "confidence",
                "tags",
                "created_at",
                "updated_at",
                "memory_type",
                "sources",
                "source",
                "node_type",
                "visibility",
                "usage",
                "background",
                "graph_id",
            ]
            return_fields_agtype = ", ".join([f"{field} agtype" for field in fields])

        if resolved_user_name:
            shard_schemas = [self._get_shard_schema_raw(resolved_user_name)]
            user_filter = f"AND n.user_name = '{resolved_user_name}'"
        else:
            shard_schemas = [f"{self.db_name}_graph_{i}" for i in range(self._shard_count)]
            user_filter = ""

        def _build_shard_query(sr: str) -> str:
            base = (
                f"SELECT * FROM cypher('{sr}', $$"
                f" MATCH (n:Memory)"
                f" WHERE n.memory_type = '{scope}' AND n.status = 'activated' {user_filter}"
                f" OPTIONAL MATCH (n)-[:PARENT]->(c:Memory)"
                f" OPTIONAL MATCH (p:Memory)-[:PARENT]->(n)"
                f" WITH n, c, p WHERE c IS NULL AND p IS NULL"
                f" RETURN {return_fields}"
                f" $$) AS ({return_fields_agtype})"
            )
            if include_embedding:
                return (
                    f'SELECT m.embedding, t.n FROM ({base}) t, {sr}."Memory" m WHERE t.id1 = m.id'
                )
            return base

        if len(shard_schemas) == 1:
            cypher_query = _build_shard_query(shard_schemas[0])
        else:
            cypher_query = " UNION ALL ".join(_build_shard_query(sr) for sr in shard_schemas)

        logger.info("get_structure_optimization_candidates query=%s", cypher_query)

        candidates = []
        node_ids = set()
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(cypher_query)
                results = cursor.fetchall()
                logger.info(f"Found {len(results)} structure optimization candidates")
                for row in results:
                    if include_embedding:
                        """
                            if isinstance(row, (list, tuple)) and len(row) >= 2:
                            """
                        if isinstance(row, list | tuple) and len(row) >= 2:
                            embedding_val, node_val = row[0], row[1]
                        else:
                            embedding_val, node_val = None, row[0]

                        node = self._build_node_from_agtype(node_val, embedding_val)
                        if node:
                            node_id = node["id"]
                            if node_id not in node_ids:
                                candidates.append(node)
                                node_ids.add(node_id)
                    else:
                        field_names = [
                            "id",
                            "memory",
                            "user_name",
                            "user_id",
                            "session_id",
                            "status",
                            "key",
                            "confidence",
                            "tags",
                            "created_at",
                            "updated_at",
                            "memory_type",
                            "sources",
                            "source",
                            "node_type",
                            "visibility",
                            "usage",
                            "background",
                            "graph_id",
                        ]

                        node_data = {}
                        for i, field_name in enumerate(field_names):
                            if i < len(row):
                                value = row[i]
                                if field_name in ["tags", "sources", "usage"] and isinstance(
                                    value, str
                                ):
                                    try:
                                        node_data[field_name] = json.loads(value)
                                    except (json.JSONDecodeError, TypeError):
                                        node_data[field_name] = value
                                else:
                                    node_data[field_name] = value

                        try:
                            node = self._parse_node_new(node_data)
                            node_id = node["id"]

                            if node_id not in node_ids:
                                candidates.append(node)
                                node_ids.add(node_id)
                                logger.debug(f"Parsed node successfully: {node_id}")
                        except Exception as e:
                            logger.error(f"Failed to parse node: {e}")

        except Exception as e:
            logger.error("get_structure_optimization_candidates failed: %s", e, exc_info=True)

        return candidates

    def drop_database(self) -> None:
        return
        if self._get_config_value("use_multi_db", True):
            with self.connection.cursor() as cursor:
                cursor.execute(f"SELECT drop_graph('{self.db_name}_graph', true)")
                logger.info(f"Graph '{self.db_name}_graph' has been dropped.")
        else:
            raise ValueError(
                f"Refusing to drop graph '{self.db_name}_graph' in "
                f"Shared Database Multi-Tenant mode"
            )

    def _parse_node(self, node_data: dict[str, Any]) -> dict[str, Any]:
        node = node_data.copy()

        for time_field in ("created_at", "updated_at"):
            if time_field in node and hasattr(node[time_field], "isoformat"):
                node[time_field] = node[time_field].isoformat()

        if "sources" in node and node.get("sources"):
            sources = node["sources"]
            if isinstance(sources, list):
                deserialized_sources = []
                for source_item in sources:
                    if isinstance(source_item, str):
                        try:
                            parsed = json.loads(source_item)
                            deserialized_sources.append(parsed)
                        except (json.JSONDecodeError, TypeError):
                            deserialized_sources.append({"type": "doc", "content": source_item})
                    elif isinstance(source_item, dict):
                        deserialized_sources.append(source_item)
                    else:
                        deserialized_sources.append({"type": "doc", "content": str(source_item)})
                node["sources"] = deserialized_sources

        return {"id": node.get("id"), "memory": node.get("memory", ""), "metadata": node}

    def _parse_node_new(self, node_data: dict[str, Any]) -> dict[str, Any]:
        node = node_data.copy()

        def _strip_wrapping_quotes(value: Any) -> Any:
            if (
                isinstance(value, str)
                and len(value) >= 2
                and value[0] == value[-1]
                and value[0] in ("'", '"')
            ):
                return value[1:-1]
            return value

        for k, v in list(node.items()):
            if isinstance(v, str):
                node[k] = _strip_wrapping_quotes(v)

        for time_field in ("created_at", "updated_at"):
            if time_field in node and hasattr(node[time_field], "isoformat"):
                node[time_field] = node[time_field].isoformat()

        if "sources" in node and node.get("sources"):
            sources = node["sources"]
            if isinstance(sources, list):
                deserialized_sources = []
                for source_item in sources:
                    if isinstance(source_item, str):
                        try:
                            parsed = json.loads(source_item)
                            deserialized_sources.append(parsed)
                        except (json.JSONDecodeError, TypeError):
                            deserialized_sources.append({"type": "doc", "content": source_item})
                    elif isinstance(source_item, dict):
                        deserialized_sources.append(source_item)
                    else:
                        deserialized_sources.append({"type": "doc", "content": str(source_item)})
                node["sources"] = deserialized_sources

        return {"id": node.pop("id"), "memory": node.pop("memory", ""), "metadata": node}

    def __del__(self):
        if hasattr(self, "connection") and self.connection:
            self.connection.close()

    @timed
    def add_node(self, id: str, memory: str, metadata: dict[str, Any], user_name: str) -> None:
        logger.info(f"[add_node] id: {id}, memory: {memory}, metadata: {metadata}")

        user_name = user_name if user_name else self.config.user_name
        schema_raw = self._get_shard_schema_raw(user_name)
        metadata["user_name"] = user_name

        metadata = _prepare_node_metadata(metadata)

        created_at = metadata.pop("created_at", datetime.utcnow().isoformat())
        updated_at = metadata.pop("updated_at", datetime.utcnow().isoformat())

        properties = {
            "id": id,
            "memory": memory,
            "created_at": created_at,
            "updated_at": updated_at,
            "delete_time": "",
            "delete_record_id": "",
            **metadata,
        }

        if "embedding" not in properties or not properties["embedding"]:
            properties["embedding"] = generate_vector(
                self._get_config_value("embedding_dimension", 1024)
            )

        for field_name in ["sources", "usage"]:
            if properties.get(field_name):
                if isinstance(properties[field_name], list):
                    for idx in range(len(properties[field_name])):
                        if not isinstance(properties[field_name][idx], str):
                            properties[field_name][idx] = json.dumps(properties[field_name][idx])
                elif isinstance(properties[field_name], str):
                    pass

        embedding_vector = properties.pop("embedding", [])
        if not isinstance(embedding_vector, list):
            embedding_vector = []

        embedding_column = "embedding"
        if len(embedding_vector) == 3072:
            embedding_column = "embedding_3072"
        elif len(embedding_vector) == 1024:
            embedding_column = "embedding"
        elif len(embedding_vector) == 768:
            embedding_column = "embedding_768"

        insert_query = None
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    delete_query = f"""
                        DELETE FROM {schema_raw}."Memory"
                        WHERE id = ag_catalog._make_graph_id('{schema_raw}'::name, 'Memory'::name, %s::text::cstring)
                    """
                    cursor.execute(delete_query, (id,))
                    get_graph_id_query = f"""
                                      SELECT ag_catalog._make_graph_id('{schema_raw}'::name, 'Memory'::name, %s::text::cstring)
                                  """
                    cursor.execute(get_graph_id_query, (id,))
                    graph_id = cursor.fetchone()[0]
                    properties["graph_id"] = str(graph_id)

                    if embedding_vector:
                        insert_query = f"""
                            INSERT INTO {schema_raw}."Memory"(id, properties, {embedding_column})
                            VALUES (
                                ag_catalog._make_graph_id('{schema_raw}'::name, 'Memory'::name, %s::text::cstring),
                                %s,
                                %s
                            )
                        """
                        cursor.execute(
                            insert_query, (id, json.dumps(properties), json.dumps(embedding_vector))
                        )
                        logger.info(
                            f"[add_node] [embedding_vector-true] insert_query: {insert_query}, properties: {json.dumps(properties)}"
                        )
                    else:
                        insert_query = f"""
                            INSERT INTO {schema_raw}."Memory"(id, properties)
                            VALUES (
                                ag_catalog._make_graph_id('{schema_raw}'::name, 'Memory'::name, %s::text::cstring),
                                %s
                            )
                        """
                        cursor.execute(insert_query, (id, json.dumps(properties)))
                        logger.info(
                            f"[add_node] [embedding_vector-false] insert_query: {insert_query}, properties: {json.dumps(properties)}"
                        )
                if insert_query:
                    logger.info(
                        f"In add node polardb: id-{id} memory-{memory} query-{insert_query}"
                    )
        except Exception as e:
            logger.error(f"[add_node] Failed to add node: {e}", exc_info=True)
            raise

    @timed
    def add_nodes_batch(
        self,
        nodes: list[dict[str, Any]],
        user_name: str,
    ) -> None:
        batch_start_time = time.perf_counter()
        if not nodes:
            logger.warning("[add_nodes_batch] Empty nodes list, skipping")
            return

        effective_user_name = user_name if user_name else self.config.user_name
        schema_raw = self._get_shard_schema_raw(effective_user_name)
        logger.info(
            "add_nodes_batch start count=%d user_name=%s schema=%s",
            len(nodes),
            user_name,
            schema_raw,
        )

        prepared_nodes = []
        for node_data in nodes:
            try:
                id = node_data["id"]
                memory = node_data["memory"]
                metadata = node_data.get("metadata", {})

                logger.debug(f"[add_nodes_batch] Processing node id: {id}")

                metadata["user_name"] = effective_user_name

                metadata = _prepare_node_metadata(metadata)

                created_at = metadata.pop("created_at", datetime.utcnow().isoformat())
                updated_at = metadata.pop("updated_at", datetime.utcnow().isoformat())

                properties = {
                    "id": id,
                    "memory": memory,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "delete_time": "",
                    "delete_record_id": "",
                    **metadata,
                }

                if "embedding" not in properties or not properties["embedding"]:
                    properties["embedding"] = generate_vector(
                        self._get_config_value("embedding_dimension", 1024)
                    )

                for field_name in ["sources", "usage"]:
                    if properties.get(field_name):
                        if isinstance(properties[field_name], list):
                            for idx in range(len(properties[field_name])):
                                if not isinstance(properties[field_name][idx], str):
                                    properties[field_name][idx] = json.dumps(
                                        properties[field_name][idx]
                                    )
                        elif isinstance(properties[field_name], str):
                            pass

                embedding_vector = properties.pop("embedding", [])
                if not isinstance(embedding_vector, list):
                    embedding_vector = []

                embedding_column = "embedding"
                if len(embedding_vector) == 3072:
                    embedding_column = "embedding_3072"
                elif len(embedding_vector) == 1024:
                    embedding_column = "embedding"
                elif len(embedding_vector) == 768:
                    embedding_column = "embedding_768"

                prepared_nodes.append(
                    {
                        "id": id,
                        "memory": memory,
                        "properties": properties,
                        "embedding_vector": embedding_vector,
                        "embedding_column": embedding_column,
                    }
                )
            except Exception as e:
                logger.error(
                    f"[add_nodes_batch] Failed to prepare node {node_data.get('id', 'unknown')}: {e}",
                    exc_info=True,
                )
                continue

        if not prepared_nodes:
            logger.warning("[add_nodes_batch] No valid nodes to insert after preparation")
            return

        nodes_by_embedding_column = {}
        for node in prepared_nodes:
            col = node["embedding_column"]
            if col not in nodes_by_embedding_column:
                nodes_by_embedding_column[col] = []
            nodes_by_embedding_column[col].append(node)

        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                for embedding_column, nodes_group in nodes_by_embedding_column.items():
                    ids_to_delete = [node["id"] for node in nodes_group]
                    if ids_to_delete:
                        delete_query = f"""
                            DELETE FROM {schema_raw}."Memory"
                            WHERE id IN (
                                SELECT ag_catalog._make_graph_id('{schema_raw}'::name, 'Memory'::name, unnest(%s::text[])::cstring)
                            )
                        """
                        cursor.execute(delete_query, (ids_to_delete,))

                    get_graph_ids_query = f"""
                        SELECT
                            id_val,
                            ag_catalog._make_graph_id('{schema_raw}'::name, 'Memory'::name, id_val::text::cstring) as graph_id
                        FROM unnest(%s::text[]) as id_val
                    """
                    cursor.execute(get_graph_ids_query, (ids_to_delete,))
                    graph_id_map = {row[0]: row[1] for row in cursor.fetchall()}

                    for node in nodes_group:
                        graph_id = graph_id_map.get(node["id"])
                        if graph_id:
                            node["properties"]["graph_id"] = str(graph_id)

                    has_embedding = bool(embedding_column) and any(
                        node["embedding_vector"] for node in nodes_group
                    )

                    if has_embedding:
                        cols = f"(id, properties, {embedding_column})"
                        value_tpl = (
                            f"(ag_catalog._make_graph_id('{schema_raw}'::name, 'Memory'::name, %s::text::cstring),"
                            " %s::text::agtype,"
                            " %s::vector)"
                        )
                    else:
                        cols = "(id, properties)"
                        value_tpl = (
                            f"(ag_catalog._make_graph_id('{schema_raw}'::name, 'Memory'::name, %s::text::cstring),"
                            " %s::text::agtype)"
                        )

                    values_clause = ",".join([value_tpl] * len(nodes_group))
                    sql = (
                        f'INSERT INTO {schema_raw}."Memory"{cols} '
                        f"VALUES {values_clause}"
                    )

                    params: list = []
                    for node in nodes_group:
                        params.append(node["id"])
                        params.append(json.dumps(node["properties"]))
                        if has_embedding:
                            embedding = node["embedding_vector"]
                            params.append(
                                json.dumps(embedding) if embedding else None
                            )
                    cursor.execute(sql, params)

            elapsed_time = (time.perf_counter() - batch_start_time) * 1000.0
            logger.info(
                "add_nodes_batch completed in %.1f ms (count=%d)",
                elapsed_time,
                len(prepared_nodes),
            )

        except Exception as e:
            logger.error(f"[add_nodes_batch] Failed to add nodes: {e}", exc_info=True)
            raise

    def _build_node_from_agtype(self, node_agtype, embedding=None):
        try:
            if isinstance(node_agtype, str):
                json_str = node_agtype.replace("::vertex", "")
                obj = json.loads(json_str)
                if not (isinstance(obj, dict) and "properties" in obj):
                    return None
                props = obj["properties"]
            elif node_agtype and hasattr(node_agtype, "value"):
                val = node_agtype.value
                if not (isinstance(val, dict) and "properties" in val):
                    return None
                props = val["properties"]
            else:
                return None

            if embedding is not None:
                if isinstance(embedding, str):
                    try:
                        embedding = json.loads(embedding)
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Failed to parse embedding for node")
                props["embedding"] = embedding

            return {"id": props.get("id", ""), "memory": props.get("memory", ""), "metadata": props}
        except Exception:
            return None

    @timed
    def get_neighbors_by_tag(
        self,
        tags: list[str],
        exclude_ids: list[str],
        top_k: int = 5,
        min_overlap: int = 1,
        include_embedding: bool = False,
        user_name: str | None = None,
    ) -> list[dict[str, Any]]:
        if not tags:
            return []

        user_name = user_name if user_name else self._get_config_value("user_name")
        tbl = self.get_memory_graph_table_name(user_name)

        where_clauses = []
        params = []

        if exclude_ids:
            exclude_conditions = []
            for exclude_id in exclude_ids:
                exclude_conditions.append(
                    "ag_catalog.agtype_access_operator(properties, '\"id\"'::agtype) != %s::agtype"
                )
                params.append(self.format_param_value(exclude_id))
            where_clauses.append(f"({' AND '.join(exclude_conditions)})")

        where_clauses.append(
            "ag_catalog.agtype_access_operator(properties, '\"status\"'::agtype) = '\"activated\"'::agtype"
        )

        where_clauses.append(
            "ag_catalog.agtype_access_operator(properties, '\"node_type\"'::agtype) != '\"reasoning\"'::agtype"
        )

        where_clauses.append(
            "ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = %s::agtype"
        )
        params.append(self.format_param_value(user_name))

        where_clauses.append(
            "ag_catalog.agtype_access_operator(properties, '\"memory_type\"'::agtype) != '\"WorkingMemory\"'::agtype"
        )

        where_clause = " AND ".join(where_clauses)

        query = f"""
            SELECT id, properties, embedding
            FROM {tbl}."Memory"
            WHERE {where_clause}
        """

        logger.debug(f"[get_neighbors_by_tag] query: {query}, params: {params}")

        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()

                nodes_with_overlap = []
                for row in results:
                    node_id, properties_json, embedding_json = row
                    properties = properties_json if properties_json else {}

                    if include_embedding and embedding_json is not None:
                        try:
                            embedding = (
                                json.loads(embedding_json)
                                if isinstance(embedding_json, str)
                                else embedding_json
                            )
                            properties["embedding"] = embedding
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(f"Failed to parse embedding for node {node_id}")

                    node_tags = properties.get("tags", [])
                    if isinstance(node_tags, str):
                        try:
                            node_tags = json.loads(node_tags)
                        except (json.JSONDecodeError, TypeError):
                            node_tags = []

                    overlap_tags = [tag for tag in tags if tag in node_tags]
                    overlap_count = len(overlap_tags)

                    if overlap_count >= min_overlap:
                        node_data = self._parse_node(
                            {
                                "id": properties.get("id", node_id),
                                "memory": properties.get("memory", ""),
                                "metadata": properties,
                            }
                        )
                        nodes_with_overlap.append((node_data, overlap_count))

                nodes_with_overlap.sort(key=lambda x: x[1], reverse=True)
                return [node for node, _ in nodes_with_overlap[:top_k]]

        except Exception as e:
            logger.error(f"Failed to get neighbors by tag: {e}", exc_info=True)
            return []

    @timed
    def import_graph(self, data: dict[str, Any], user_name: str | None = None) -> None:
        logger.info(
            "import_graph user_name=%s, nodes=%d, edges=%d",
            user_name,
            len(data.get("nodes", [])),
            len(data.get("edges", [])),
        )
        resolved_user_name = user_name

        for node in data.get("nodes", []):
            try:
                id, memory, metadata = _compose_node(node)
                node_user_name = resolved_user_name or metadata.get("user_name")
                if node_user_name:
                    metadata["user_name"] = node_user_name
                metadata = _prepare_node_metadata(metadata)
                metadata.update({"id": id, "memory": memory})

                self.add_node(id, memory, metadata, user_name=node_user_name)

            except Exception as e:
                logger.error(
                    "import_graph fail to load node: %s, error: %s", node.get("id", "unknown"), e
                )

        for edge in data.get("edges", []):
            try:
                source_id, target_id = edge["source"], edge["target"]
                edge_type = edge["type"]

                self.add_edge(source_id, target_id, edge_type, resolved_user_name)

            except Exception as e:
                logger.error("import_graph fail to load edge: %s, error: %s", edge, e)

    def _build_cypher_edge_body(
        self, id_esc: str, user_esc: str | None, type: str, type_filter: str, direction: str
    ) -> str:
        user_cond = f" AND a.user_name = '{user_esc}'" if user_esc else ""
        if direction == "OUTGOING":
            return (
                f"MATCH (a:Memory)-[r:{type}]->(b:Memory)\n"
                f"WHERE a.id = '{id_esc}'{user_cond}\n"
                f"RETURN a.id AS from_id, b.id AS to_id, type(r) AS edge_type"
            )
        elif direction == "INCOMING":
            return (
                f"MATCH (b:Memory)<-[r:{type}]-(a:Memory)\n"
                f"WHERE a.id = '{id_esc}'{user_cond}\n"
                f"RETURN a.id AS from_id, b.id AS to_id, type(r) AS edge_type"
            )
        else:
            return (
                f"MATCH (a:Memory)-[r]->(b:Memory)\n"
                f"WHERE a.id = '{id_esc}'{user_cond}{type_filter}\n"
                f"RETURN a.id AS from_id, b.id AS to_id, type(r) AS edge_type\n"
                f"UNION ALL\n"
                f"MATCH (b:Memory)<-[r]-(a:Memory)\n"
                f"WHERE a.id = '{id_esc}'{user_cond}{type_filter}\n"
                f"RETURN a.id AS from_id, b.id AS to_id, type(r) AS edge_type"
            )

    @staticmethod
    def _parse_edge_rows(rows: list) -> list[dict[str, str]]:
        edges = []
        for row in rows:
            from_id_raw = row[0].value if hasattr(row[0], "value") else row[0]
            if (
                isinstance(from_id_raw, str)
                and from_id_raw.startswith('"')
                and from_id_raw.endswith('"')
            ):
                from_id = from_id_raw[1:-1]
            else:
                from_id = str(from_id_raw)

            to_id_raw = row[1].value if hasattr(row[1], "value") else row[1]
            if isinstance(to_id_raw, str) and to_id_raw.startswith('"') and to_id_raw.endswith('"'):
                to_id = to_id_raw[1:-1]
            else:
                to_id = str(to_id_raw)

            edge_type_raw = row[2].value if hasattr(row[2], "value") else row[2]
            if (
                isinstance(edge_type_raw, str)
                and edge_type_raw.startswith('"')
                and edge_type_raw.endswith('"')
            ):
                edge_type = edge_type_raw[1:-1]
            else:
                edge_type = str(edge_type_raw)

            edges.append({"from": from_id, "to": to_id, "type": edge_type})
        return edges

    @timed
    def get_edges(
        self, id: str, type: str = "ANY", direction: str = "ANY", user_name: str | None = None
    ) -> list[dict[str, str]]:
        start_time = time.perf_counter()
        logger.info(f" get_edges id:{id},type:{type},direction:{direction},user_name:{user_name}")
        resolved_user_name = user_name
        if direction not in ("OUTGOING", "INCOMING", "ANY"):
            raise ValueError("Invalid direction. Must be 'OUTGOING', 'INCOMING', or 'ANY'.")

        id_esc = (id or "").replace("'", "''")
        type_esc = (type or "").replace("'", "''")
        type_filter = f" AND type(r) = '{type_esc}'" if type != "ANY" else ""

        if resolved_user_name:
            schema_raw = self._get_shard_schema_raw(resolved_user_name)
            user_esc = resolved_user_name.replace("'", "''")
            cypher_body = self._build_cypher_edge_body(
                id_esc, user_esc, type, type_filter, direction
            )
            query = (
                f"SELECT * FROM cypher('{schema_raw}', $$\n"
                f"{cypher_body}\n"
                f"$$) AS (from_id agtype, to_id agtype, edge_type agtype)"
            )
        else:
            cypher_body = self._build_cypher_edge_body(id_esc, None, type, type_filter, direction)
            union_parts = [
                (
                    f"SELECT * FROM cypher('{schema}', $$\n"
                    f"{cypher_body}\n"
                    f"$$) AS (from_id agtype, to_id agtype, edge_type agtype)"
                )
                for schema in self._get_all_shard_schemas()
            ]
            query = " UNION ALL ".join(union_parts)

        logger.info(f"get_edges query length:{len(query)}")
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(query)
                edges = self._parse_edge_rows(cursor.fetchall())
                elapsed_time = (time.perf_counter() - start_time) * 1000.0
                logger.info(
                    "polardb get_edges completed time in took %.1f ms",
                    elapsed_time,
                )
                return edges
        except Exception as e:
            logger.error(f"Failed to get edges: {e}", exc_info=True)
            return []

    def _convert_graph_edges(self, core_node: dict) -> dict:
        import copy

        data = copy.deepcopy(core_node)
        id_map = {}
        core_node = data.get("core_node", {})
        if not core_node:
            return {
                "core_node": None,
                "neighbors": data.get("neighbors", []),
                "edges": data.get("edges", []),
            }
        core_meta = core_node.get("metadata", {})
        if "graph_id" in core_meta and "id" in core_node:
            id_map[core_meta["graph_id"]] = core_node["id"]
        for neighbor in data.get("neighbors", []):
            n_meta = neighbor.get("metadata", {})
            if "graph_id" in n_meta and "id" in neighbor:
                id_map[n_meta["graph_id"]] = neighbor["id"]
        for edge in data.get("edges", []):
            src = edge.get("source")
            tgt = edge.get("target")
            if src in id_map:
                edge["source"] = id_map[src]
            if tgt in id_map:
                edge["target"] = id_map[tgt]
        return data

    def format_param_value(self, value: str | None) -> str:
        if value is None:
            logger.warning("format_param_value: value is None")
            return "null"

        if value.startswith('"') and value.endswith('"'):
            return value
        else:
            return f'"{value}"'

    def _build_user_name_and_kb_ids_conditions_cypher(
        self,
        user_name: str | None,
        knowledgebase_ids: list | None,
    ) -> list[str]:
        user_name_conditions = []

        if user_name:
            escaped_user_name = user_name.replace("'", "''")
            user_name_conditions.append(f"n.user_name = '{escaped_user_name}'")

        if knowledgebase_ids and isinstance(knowledgebase_ids, list) and len(knowledgebase_ids) > 0:
            for kb_id in knowledgebase_ids:
                if isinstance(kb_id, str):
                    escaped_kb_id = kb_id.replace("'", "''")
                    user_name_conditions.append(f"n.user_name = '{escaped_kb_id}'")

        return user_name_conditions

    def _build_user_name_and_kb_ids_conditions_sql(
        self,
        user_name: str | None,
        knowledgebase_ids: list | None,
    ) -> list[str]:
        user_name_conditions = []

        if user_name:
            user_name_conditions.append(
                f"ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = '\"{user_name}\"'::agtype"
            )

        if knowledgebase_ids and isinstance(knowledgebase_ids, list) and len(knowledgebase_ids) > 0:
            for kb_id in knowledgebase_ids:
                if isinstance(kb_id, str):
                    user_name_conditions.append(
                        f"ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = '\"{kb_id}\"'::agtype"
                    )

        return user_name_conditions

    def _build_filter_conditions_cypher(
        self,
        filter: dict | None,
    ) -> str:
        filter_where_clause = ""
        filter = self.parse_filter(filter)
        if filter:

            def escape_cypher_string(value: str) -> str:
                return value.replace("'", "\\'")

            def build_cypher_filter_condition(condition_dict: dict) -> str:
                condition_parts = []
                for key, value in condition_dict.items():
                    if isinstance(value, dict):
                        for op, op_value in value.items():
                            if op in ("gt", "lt", "gte", "lte"):
                                cypher_op_map = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<="}
                                cypher_op = cypher_op_map[op]

                                is_datetime = key in ("created_at", "updated_at") or key.endswith(
                                    "_at"
                                )

                                if key.startswith("info."):
                                    info_field = key[5:]
                                    is_info_datetime = info_field in (
                                        "created_at",
                                        "updated_at",
                                    ) or info_field.endswith("_at")
                                    if isinstance(op_value, str):
                                        escaped_value = escape_cypher_string(op_value)
                                        if is_info_datetime:
                                            condition_parts.append(
                                                f"n.info.{info_field}::timestamp {cypher_op} '{escaped_value}'::timestamp"
                                            )
                                        else:
                                            condition_parts.append(
                                                f"n.info.{info_field} {cypher_op} '{escaped_value}'"
                                            )
                                    else:
                                        condition_parts.append(
                                            f"n.info.{info_field} {cypher_op} {op_value}"
                                        )
                                else:
                                    if isinstance(op_value, str):
                                        escaped_value = escape_cypher_string(op_value)
                                        if is_datetime:
                                            condition_parts.append(
                                                f"n.{key}::timestamp {cypher_op} '{escaped_value}'::timestamp"
                                            )
                                        else:
                                            condition_parts.append(
                                                f"n.{key} {cypher_op} '{escaped_value}'"
                                            )
                                    else:
                                        condition_parts.append(f"n.{key} {cypher_op} {op_value}")
                            elif op == "=":
                                if key.startswith("info."):
                                    info_field = key[5:]
                                    if isinstance(op_value, str):
                                        escaped_value = escape_cypher_string(op_value)
                                        if info_field in ("tags", "sources"):
                                            condition_parts.append(
                                                f"n.info.{info_field} = ['{escaped_value}']"
                                            )
                                        else:
                                            condition_parts.append(
                                                f"n.info.{info_field} = '{escaped_value}'"
                                            )
                                    elif isinstance(op_value, list):
                                        if info_field in ("tags", "sources"):
                                            escaped_items = [
                                                f"'{escape_cypher_string(str(item))}'"
                                                for item in op_value
                                            ]
                                            array_str = "[" + ", ".join(escaped_items) + "]"
                                            condition_parts.append(
                                                f"n.info.{info_field} = {array_str}"
                                            )
                                        else:
                                            condition_parts.append(
                                                f"n.info.{info_field} = {op_value}"
                                            )
                                    else:
                                        if info_field in ("tags", "sources"):
                                            condition_parts.append(
                                                f"n.info.{info_field} = [{op_value}]"
                                            )
                                        else:
                                            condition_parts.append(
                                                f"n.info.{info_field} = {op_value}"
                                            )
                                else:
                                    if isinstance(op_value, str):
                                        escaped_value = escape_cypher_string(op_value)
                                        if key in ("tags", "sources"):
                                            condition_parts.append(f"n.{key} = ['{escaped_value}']")
                                        else:
                                            condition_parts.append(f"n.{key} = '{escaped_value}'")
                                    elif isinstance(op_value, list):
                                        if key in ("tags", "sources"):
                                            escaped_items = [
                                                f"'{escape_cypher_string(str(item))}'"
                                                for item in op_value
                                            ]
                                            array_str = "[" + ", ".join(escaped_items) + "]"
                                            condition_parts.append(f"n.{key} = {array_str}")
                                        else:
                                            condition_parts.append(f"n.{key} = {op_value}")
                                    else:
                                        if key in ("tags", "sources"):
                                            condition_parts.append(f"n.{key} = [{op_value}]")
                                        else:
                                            condition_parts.append(f"n.{key} = {op_value}")
                            elif op == "contains":
                                if key.startswith("info."):
                                    info_field = key[5:]
                                    if isinstance(op_value, str):
                                        escaped_value = escape_cypher_string(op_value)
                                        condition_parts.append(
                                            f"'{escaped_value}' IN n.info.{info_field}"
                                        )
                                    else:
                                        condition_parts.append(f"{op_value} IN n.info.{info_field}")
                                else:
                                    if isinstance(op_value, str):
                                        escaped_value = escape_cypher_string(op_value)
                                        condition_parts.append(f"'{escaped_value}' IN n.{key}")
                                    else:
                                        condition_parts.append(f"{op_value} IN n.{key}")
                            elif op == "in":
                                if not isinstance(op_value, list):
                                    raise ValueError(
                                        f"in operator only supports array format. "
                                        f"Use {{'{key}': {{'in': ['{op_value}']}}}} instead of {{'{key}': {{'in': '{op_value}'}}}}"
                                    )
                                is_array_field = key in ("file_ids", "tags", "sources")

                                if key.startswith("info."):
                                    info_field = key[5:]
                                    is_info_array = info_field in ("tags", "sources", "file_ids")

                                    if len(op_value) == 0:
                                        condition_parts.append("false")
                                    elif len(op_value) == 1:
                                        item = op_value[0]
                                        if is_info_array:
                                            if isinstance(item, str):
                                                escaped_value = escape_cypher_string(item)
                                                condition_parts.append(
                                                    f"'{escaped_value}' IN n.info.{info_field}"
                                                )
                                            else:
                                                condition_parts.append(
                                                    f"{item} IN n.info.{info_field}"
                                                )
                                        else:
                                            if isinstance(item, str):
                                                escaped_value = escape_cypher_string(item)
                                                condition_parts.append(
                                                    f"n.info.{info_field} = '{escaped_value}'"
                                                )
                                            else:
                                                condition_parts.append(
                                                    f"n.info.{info_field} = {item}"
                                                )
                                    else:
                                        or_conditions = []
                                        for item in op_value:
                                            if is_info_array:
                                                if isinstance(item, str):
                                                    escaped_value = escape_cypher_string(item)
                                                    or_conditions.append(
                                                        f"'{escaped_value}' IN n.info.{info_field}"
                                                    )
                                                else:
                                                    or_conditions.append(
                                                        f"{item} IN n.info.{info_field}"
                                                    )
                                            else:
                                                if isinstance(item, str):
                                                    escaped_value = escape_cypher_string(item)
                                                    or_conditions.append(
                                                        f"n.info.{info_field} = '{escaped_value}'"
                                                    )
                                                else:
                                                    or_conditions.append(
                                                        f"n.info.{info_field} = {item}"
                                                    )
                                        if or_conditions:
                                            condition_parts.append(
                                                f"({' OR '.join(or_conditions)})"
                                            )
                                else:
                                    if len(op_value) == 0:
                                        condition_parts.append("false")
                                    elif len(op_value) == 1:
                                        item = op_value[0]
                                        if is_array_field:
                                            if isinstance(item, str):
                                                escaped_value = escape_cypher_string(item)
                                                condition_parts.append(
                                                    f"'{escaped_value}' IN n.{key}"
                                                )
                                            else:
                                                condition_parts.append(f"{item} IN n.{key}")
                                        else:
                                            if isinstance(item, str):
                                                escaped_value = escape_cypher_string(item)
                                                condition_parts.append(
                                                    f"n.{key} = '{escaped_value}'"
                                                )
                                            else:
                                                condition_parts.append(f"n.{key} = {item}")
                                    else:
                                        if is_array_field:
                                            or_conditions = []
                                            for item in op_value:
                                                if isinstance(item, str):
                                                    escaped_value = escape_cypher_string(item)
                                                    or_conditions.append(
                                                        f"'{escaped_value}' IN n.{key}"
                                                    )
                                                else:
                                                    or_conditions.append(f"{item} IN n.{key}")
                                            if or_conditions:
                                                condition_parts.append(
                                                    f"({' OR '.join(or_conditions)})"
                                                )
                                        else:
                                            escaped_items = [
                                                f"'{escape_cypher_string(str(item))}'"
                                                if isinstance(item, str)
                                                else str(item)
                                                for item in op_value
                                            ]
                                            array_str = "[" + ", ".join(escaped_items) + "]"
                                            condition_parts.append(f"n.{key} IN {array_str}")
                            elif op == "like":
                                if key.startswith("info."):
                                    info_field = key[5:]
                                    if isinstance(op_value, str):
                                        escaped_value = escape_cypher_string(op_value)
                                        condition_parts.append(
                                            f"n.info.{info_field} CONTAINS '{escaped_value}'"
                                        )
                                    else:
                                        condition_parts.append(
                                            f"n.info.{info_field} CONTAINS {op_value}"
                                        )
                                else:
                                    if isinstance(op_value, str):
                                        escaped_value = escape_cypher_string(op_value)
                                        condition_parts.append(
                                            f"n.{key} CONTAINS '{escaped_value}'"
                                        )
                                    else:
                                        condition_parts.append(f"n.{key} CONTAINS {op_value}")
                    elif key.startswith("info."):
                        info_field = key[5:]
                        if isinstance(value, str):
                            escaped_value = escape_cypher_string(value)
                            condition_parts.append(f"n.info.{info_field} = '{escaped_value}'")
                        else:
                            condition_parts.append(f"n.info.{info_field} = {value}")
                    else:
                        if isinstance(value, str):
                            escaped_value = escape_cypher_string(value)
                            condition_parts.append(f"n.{key} = '{escaped_value}'")
                        else:
                            condition_parts.append(f"n.{key} = {value}")
                return " AND ".join(condition_parts)

            if isinstance(filter, dict):
                if "or" in filter:
                    or_conditions = []
                    for condition in filter["or"]:
                        if isinstance(condition, dict):
                            condition_str = build_cypher_filter_condition(condition)
                            if condition_str:
                                or_conditions.append(f"({condition_str})")
                    if or_conditions:
                        filter_where_clause = " AND " + f"({' OR '.join(or_conditions)})"

                elif "and" in filter:
                    and_conditions = []
                    for condition in filter["and"]:
                        if isinstance(condition, dict):
                            condition_str = build_cypher_filter_condition(condition)
                            if condition_str:
                                and_conditions.append(f"({condition_str})")
                    if and_conditions:
                        filter_where_clause = " AND " + " AND ".join(and_conditions)
                else:
                    condition_str = build_cypher_filter_condition(filter)
                    if condition_str:
                        filter_where_clause = " AND " + condition_str

        return filter_where_clause

    def _build_filter_conditions_sql(
        self,
        filter: dict | None,
    ) -> list[str]:
        filter_conditions = []
        filter = self.parse_filter(filter)
        if filter:

            def escape_sql_string(value: str) -> str:
                return value.replace("'", "''")

            def build_filter_condition(condition_dict: dict) -> str:
                condition_parts = []
                for key, value in condition_dict.items():
                    if isinstance(value, dict):
                        for op, op_value in value.items():
                            if op in ("gt", "lt", "gte", "lte"):
                                sql_op_map = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<="}
                                sql_op = sql_op_map[op]

                                is_datetime = key in ("created_at", "updated_at") or key.endswith(
                                    "_at"
                                )

                                if key.startswith("info."):
                                    info_field = key[5:]
                                    is_info_datetime = info_field in (
                                        "created_at",
                                        "updated_at",
                                    ) or info_field.endswith("_at")
                                    if isinstance(op_value, str):
                                        escaped_value = escape_sql_string(op_value)
                                        if is_info_datetime:
                                            condition_parts.append(
                                                f"TRIM(BOTH '\"' FROM ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"info\"'::ag_catalog.agtype, '\"{info_field}\"'::ag_catalog.agtype)::text)::timestamp {sql_op} '{escaped_value}'::timestamp"
                                            )
                                        else:
                                            condition_parts.append(
                                                f"ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"info\"'::ag_catalog.agtype, '\"{info_field}\"'::ag_catalog.agtype]) {sql_op} '\"{escaped_value}\"'::agtype"
                                            )
                                    else:
                                        value_json = json.dumps(op_value)
                                        condition_parts.append(
                                            f"ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"info\"'::ag_catalog.agtype, '\"{info_field}\"'::ag_catalog.agtype]) {sql_op} ag_catalog.agtype_in('{value_json}')"
                                        )
                                else:
                                    if isinstance(op_value, str):
                                        escaped_value = escape_sql_string(op_value)
                                        if is_datetime:
                                            condition_parts.append(
                                                f"TRIM(BOTH '\"' FROM ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype)::text)::timestamp {sql_op} '{escaped_value}'::timestamp"
                                            )
                                        else:
                                            condition_parts.append(
                                                f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) {sql_op} '\"{escaped_value}\"'::agtype"
                                            )
                                    else:
                                        value_json = json.dumps(op_value)
                                        condition_parts.append(
                                            f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) {sql_op} ag_catalog.agtype_in('{value_json}')"
                                        )
                            elif op == "=":
                                if key.startswith("info."):
                                    info_field = key[5:]
                                    if isinstance(op_value, str):
                                        escaped_value = escape_sql_string(op_value)
                                        if info_field in ("tags", "sources"):
                                            condition_parts.append(
                                                f"ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"info\"'::ag_catalog.agtype, '\"{info_field}\"'::ag_catalog.agtype]) = '[\"{escaped_value}\"]'::agtype"
                                            )
                                        else:
                                            condition_parts.append(
                                                f"ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"info\"'::ag_catalog.agtype, '\"{info_field}\"'::ag_catalog.agtype]) = '\"{escaped_value}\"'::agtype"
                                            )
                                    elif isinstance(op_value, list):
                                        if info_field in ("tags", "sources"):
                                            escaped_items = [
                                                escape_sql_string(str(item)) for item in op_value
                                            ]
                                            json_array = json.dumps(escaped_items)
                                            condition_parts.append(
                                                f"ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"info\"'::ag_catalog.agtype, '\"{info_field}\"'::ag_catalog.agtype]) = '{json_array}'::agtype"
                                            )
                                        else:
                                            condition_parts.append(
                                                f"ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"info\"'::ag_catalog.agtype, '\"{info_field}\"'::ag_catalog.agtype]) = {op_value}::agtype"
                                            )
                                    else:
                                        if info_field in ("tags", "sources"):
                                            condition_parts.append(
                                                f"ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"info\"'::ag_catalog.agtype, '\"{info_field}\"'::ag_catalog.agtype]) = '[{op_value}]'::agtype"
                                            )
                                        else:
                                            value_json = json.dumps(op_value)
                                            condition_parts.append(
                                                f"ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"info\"'::ag_catalog.agtype, '\"{info_field}\"'::ag_catalog.agtype]) = ag_catalog.agtype_in('{value_json}')"
                                            )
                                else:
                                    if isinstance(op_value, str):
                                        escaped_value = escape_sql_string(op_value)
                                        if key in ("tags", "sources"):
                                            condition_parts.append(
                                                f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) = '[\"{escaped_value}\"]'::agtype"
                                            )
                                        else:
                                            condition_parts.append(
                                                f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) = '\"{escaped_value}\"'::agtype"
                                            )
                                    elif isinstance(op_value, list):
                                        if key in ("tags", "sources"):
                                            escaped_items = [
                                                escape_sql_string(str(item)) for item in op_value
                                            ]
                                            json_array = json.dumps(escaped_items)
                                            condition_parts.append(
                                                f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) = '{json_array}'::agtype"
                                            )
                                        else:
                                            value_json = json.dumps(op_value)
                                            condition_parts.append(
                                                f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) = ag_catalog.agtype_in('{value_json}')"
                                            )
                                    else:
                                        if key in ("tags", "sources"):
                                            condition_parts.append(
                                                f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) = '[{op_value}]'::agtype"
                                            )
                                        else:
                                            value_json = json.dumps(op_value)
                                            condition_parts.append(
                                                f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) = ag_catalog.agtype_in('{value_json}')"
                                            )
                            elif op == "contains":
                                if key.startswith("info."):
                                    info_field = key[5:]
                                    escaped_value = escape_sql_string(str(op_value))
                                    condition_parts.append(
                                        f"ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"info\"'::ag_catalog.agtype, '\"{info_field}\"'::ag_catalog.agtype]) @> '[\"{escaped_value}\"]'::agtype"
                                    )
                                else:
                                    escaped_value = escape_sql_string(str(op_value))
                                    condition_parts.append(
                                        f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) @> '[\"{escaped_value}\"]'::agtype"
                                    )
                            elif op == "in":
                                if not isinstance(op_value, list):
                                    raise ValueError(
                                        f"in operator only supports array format. "
                                        f"Use {{'{key}': {{'in': ['{op_value}']}}}} instead of {{'{key}': {{'in': '{op_value}'}}}}"
                                    )
                                is_array_field = key in ("file_ids", "tags", "sources")

                                if key.startswith("info."):
                                    info_field = key[5:]
                                    is_info_array = info_field in ("tags", "sources", "file_ids")

                                    if len(op_value) == 0:
                                        condition_parts.append("false")
                                    elif len(op_value) == 1:
                                        item = op_value[0]
                                        if is_info_array:
                                            escaped_value = escape_sql_string(str(item))
                                            condition_parts.append(
                                                f"ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"info\"'::ag_catalog.agtype, '\"{info_field}\"'::ag_catalog.agtype]) @> '[\"{escaped_value}\"]'::agtype"
                                            )
                                        else:
                                            if isinstance(item, str):
                                                escaped_value = escape_sql_string(item)
                                                condition_parts.append(
                                                    f"ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"info\"'::ag_catalog.agtype, '\"{info_field}\"'::ag_catalog.agtype]) = '\"{escaped_value}\"'::agtype"
                                                )
                                            else:
                                                condition_parts.append(
                                                    f"ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"info\"'::ag_catalog.agtype, '\"{info_field}\"'::ag_catalog.agtype]) = {item}::agtype"
                                                )
                                    else:
                                        or_conditions = []
                                        for item in op_value:
                                            if is_info_array:
                                                escaped_value = escape_sql_string(str(item))
                                                or_conditions.append(
                                                    f"ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"info\"'::ag_catalog.agtype, '\"{info_field}\"'::ag_catalog.agtype]) @> '[\"{escaped_value}\"]'::agtype"
                                                )
                                            else:
                                                if isinstance(item, str):
                                                    escaped_value = escape_sql_string(item)
                                                    or_conditions.append(
                                                        f"ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"info\"'::ag_catalog.agtype, '\"{info_field}\"'::ag_catalog.agtype]) = '\"{escaped_value}\"'::agtype"
                                                    )
                                                else:
                                                    or_conditions.append(
                                                        f"ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"info\"'::ag_catalog.agtype, '\"{info_field}\"'::ag_catalog.agtype]) = {item}::agtype"
                                                    )
                                        if or_conditions:
                                            condition_parts.append(
                                                f"({' OR '.join(or_conditions)})"
                                            )
                                else:
                                    if len(op_value) == 0:
                                        condition_parts.append("false")
                                    elif len(op_value) == 1:
                                        item = op_value[0]
                                        if is_array_field:
                                            escaped_value = escape_sql_string(str(item))
                                            condition_parts.append(
                                                f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) @> '[\"{escaped_value}\"]'::agtype"
                                            )
                                        else:
                                            if isinstance(item, str):
                                                escaped_value = escape_sql_string(item)
                                                condition_parts.append(
                                                    f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) = '\"{escaped_value}\"'::agtype"
                                                )
                                            else:
                                                condition_parts.append(
                                                    f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) = {item}::agtype"
                                                )
                                    else:
                                        or_conditions = []
                                        for item in op_value:
                                            if is_array_field:
                                                escaped_value = escape_sql_string(str(item))
                                                or_conditions.append(
                                                    f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) @> '[\"{escaped_value}\"]'::agtype"
                                                )
                                            else:
                                                if isinstance(item, str):
                                                    escaped_value = escape_sql_string(item)
                                                    or_conditions.append(
                                                        f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) = '\"{escaped_value}\"'::agtype"
                                                    )
                                                else:
                                                    or_conditions.append(
                                                        f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) = {item}::agtype"
                                                    )
                                        if or_conditions:
                                            condition_parts.append(
                                                f"({' OR '.join(or_conditions)})"
                                            )
                            elif op == "like":
                                if key.startswith("info."):
                                    info_field = key[5:]
                                    if isinstance(op_value, str):
                                        escaped_value = (
                                            escape_sql_string(op_value)
                                            .replace("%", "\\%")
                                            .replace("_", "\\_")
                                        )
                                        condition_parts.append(
                                            f"ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"info\"'::ag_catalog.agtype, '\"{info_field}\"'::ag_catalog.agtype])::text LIKE '%{escaped_value}%'"
                                        )
                                    else:
                                        condition_parts.append(
                                            f"ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"info\"'::ag_catalog.agtype, '\"{info_field}\"'::ag_catalog.agtype])::text LIKE '%{op_value}%'"
                                        )
                                else:
                                    if isinstance(op_value, str):
                                        escaped_value = (
                                            escape_sql_string(op_value)
                                            .replace("%", "\\%")
                                            .replace("_", "\\_")
                                        )
                                        condition_parts.append(
                                            f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype)::text LIKE '%{escaped_value}%'"
                                        )
                                    else:
                                        condition_parts.append(
                                            f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype)::text LIKE '%{op_value}%'"
                                        )
                            elif op == "nolike":
                                if key.startswith("info."):
                                    info_field = key[5:]
                                    if isinstance(op_value, str):
                                        escaped_value = (
                                            escape_sql_string(op_value)
                                            .replace("%", "\\%")
                                            .replace("_", "\\_")
                                        )
                                        condition_parts.append(
                                            f"ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"info\"'::ag_catalog.agtype, '\"{info_field}\"'::ag_catalog.agtype])::text NOT LIKE '%{escaped_value}%'"
                                        )
                                    else:
                                        condition_parts.append(
                                            f"ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"info\"'::ag_catalog.agtype, '\"{info_field}\"'::ag_catalog.agtype])::text NOT LIKE '%{op_value}%'"
                                        )
                                else:
                                    if isinstance(op_value, str):
                                        escaped_value = (
                                            escape_sql_string(op_value)
                                            .replace("%", "\\%")
                                            .replace("_", "\\_")
                                        )
                                        condition_parts.append(
                                            f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype)::text NOT LIKE '%{escaped_value}%'"
                                        )
                                    else:
                                        condition_parts.append(
                                            f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype)::text NOT LIKE '%{op_value}%'"
                                        )
                    elif key.startswith("info."):
                        info_field = key[5:]
                        if isinstance(value, str):
                            escaped_value = escape_sql_string(value)
                            condition_parts.append(
                                f"ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"info\"'::ag_catalog.agtype, '\"{info_field}\"'::ag_catalog.agtype]) = '\"{escaped_value}\"'::agtype"
                            )
                        else:
                            value_json = json.dumps(value)
                            condition_parts.append(
                                f"ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '\"info\"'::ag_catalog.agtype, '\"{info_field}\"'::ag_catalog.agtype]) = ag_catalog.agtype_in('{value_json}')"
                            )
                    else:
                        if isinstance(value, str):
                            escaped_value = escape_sql_string(value)
                            condition_parts.append(
                                f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) = '\"{escaped_value}\"'::agtype"
                            )
                        else:
                            value_json = json.dumps(value)
                            condition_parts.append(
                                f"ag_catalog.agtype_access_operator(properties, '\"{key}\"'::agtype) = ag_catalog.agtype_in('{value_json}')"
                            )
                return " AND ".join(condition_parts)

            if isinstance(filter, dict):
                if "or" in filter:
                    or_conditions = []
                    for condition in filter["or"]:
                        if isinstance(condition, dict):
                            condition_str = build_filter_condition(condition)
                            if condition_str:
                                or_conditions.append(f"({condition_str})")
                    if or_conditions:
                        filter_conditions.append(f"({' OR '.join(or_conditions)})")

                elif "and" in filter:
                    for condition in filter["and"]:
                        if isinstance(condition, dict):
                            condition_str = build_filter_condition(condition)
                            if condition_str:
                                filter_conditions.append(f"({condition_str})")
                else:
                    condition_str = build_filter_condition(filter)
                    if condition_str:
                        filter_conditions.append(condition_str)

        return filter_conditions

    def parse_filter(
        self,
        filter_dict: dict | None = None,
    ):
        if filter_dict is None:
            return None
        full_fields = {
            "id",
            "key",
            "tags",
            "type",
            "usage",
            "memory",
            "status",
            "sources",
            "user_id",
            "graph_id",
            "user_name",
            "background",
            "confidence",
            "created_at",
            "session_id",
            "updated_at",
            "memory_type",
            "node_type",
            "info",
            "source",
            "file_ids",
            "project_id",
            "manager_user_id",
            "delete_time",
            "related_id",
        }

        def process_condition(condition):
            if not isinstance(condition, dict):
                return condition

            new_condition = {}

            for key, value in condition.items():
                if key.lower() in ["or", "and"]:
                    if isinstance(value, list):
                        processed_items = []
                        for item in value:
                            if isinstance(item, dict):
                                processed_item = {}
                                for item_key, item_value in item.items():
                                    if item_key not in full_fields and not item_key.startswith(
                                        "info."
                                    ):
                                        new_item_key = f"info.{item_key}"
                                    else:
                                        new_item_key = item_key
                                    processed_item[new_item_key] = item_value
                                processed_items.append(processed_item)
                            else:
                                processed_items.append(item)
                        new_condition[key] = processed_items
                    else:
                        new_condition[key] = value
                else:
                    if key not in full_fields and not key.startswith("info."):
                        new_key = f"info.{key}"
                    else:
                        new_key = key

                    new_condition[new_key] = value

            return new_condition

        return process_condition(filter_dict)

    @timed
    def delete_node_by_prams(
        self,
        writable_cube_ids: list[str] | None = None,
        memory_ids: list[str] | None = None,
        file_ids: list[str] | None = None,
        filter: dict | None = None,
    ) -> int:
        logger.info(
            "delete_node_by_prams memory_ids=%s file_ids=%s filter=%s writable_cube_ids=%s",
            memory_ids,
            file_ids,
            filter,
            writable_cube_ids,
        )

        if memory_ids:
            return self._delete_by_memory_ids(memory_ids)

        if writable_cube_ids and file_ids:
            return self._delete_by_cube_and_file_ids(writable_cube_ids, file_ids)

        if filter:
            return self._delete_by_filter(filter)

        logger.warning(
            "delete_node_by_prams no matching scenario: memory_ids/cube+file/filter all empty"
        )
        return 0

    def _delete_by_memory_ids(self, memory_ids: list[str]) -> int:
        start_time = time.perf_counter()
        if not memory_ids:
            return 0

        existing_schemas = self._get_existing_shard_schemas()
        if not existing_schemas:
            logger.info(
                "delete_by_memory_ids skipped: no existing shard schemas found"
                " (configured_shards=%d)",
                self._shard_count,
            )
            return 0

        batch_start_time = time.time()
        cte_parts: list[str] = ["ids AS (SELECT unnest(%s::text[])::text AS mid)"]
        count_parts: list[str] = []

        for idx, schema_raw in enumerate(existing_schemas):
            cte_name = f"d{idx}"
            cte_parts.append(
                f"{cte_name} AS ("
                f'DELETE FROM "{schema_raw}"."Memory" WHERE id IN ('
                f"SELECT ag_catalog._make_graph_id("
                f"'{schema_raw}'::name, 'Memory'::name, mid::cstring) "
                f"FROM ids"
                f") RETURNING 1"
                f")"
            )
            count_parts.append(f"(SELECT count(*) FROM {cte_name})")

        sql = (
            "WITH "
            + ", ".join(cte_parts)
            + " SELECT "
            + " + ".join(count_parts)
            + " AS total_deleted"
        )

        total_deleted = 0
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(sql, [memory_ids])
                row = cursor.fetchone()
                total_deleted = int(row[0]) if row and row[0] is not None else 0
        except Exception as e:
            logger.error("delete_by_memory_ids failed: %s", e, exc_info=True)
            raise

        elapsed_ms = (time.time() - batch_start_time) * 1000.0
        logger.info(
            "delete_by_memory_ids completed in %.2fms, deleted %d nodes",
            elapsed_ms,
            total_deleted,
        )
        return total_deleted

    def _delete_by_cube_and_file_ids(
        self,
        writable_cube_ids: list[str],
        file_ids: list[str],
    ) -> int:

        batch_start_time = time.time()
        shard_to_cube_ids: dict[str, list[str]] = {}
        for cube_id in writable_cube_ids:
            shard = self.get_memory_graph_table_name(cube_id)
            shard_to_cube_ids.setdefault(shard, []).append(cube_id)

        target_tables = list(shard_to_cube_ids.keys())
        logger.info(f"_delete_by_cube_and_file_ids target_tables:{target_tables}")
        if not target_tables:
            return 0

        if not file_ids:
            logger.warning("_delete_by_cube_and_file_ids skipped: file_ids is empty")
            return 0


        cte_parts: list[str] = []
        count_parts: list[str] = []
        params: list = []

        for idx, tbl in enumerate(target_tables):
            cube_ids_on_shard = shard_to_cube_ids[tbl]
            cte_name = f"d{idx}"

            params.append(list(cube_ids_on_shard))
            params.append(list(file_ids))

            cte_parts.append(
                f"{cte_name} AS ("
                f'DELETE FROM {tbl}."Memory" WHERE '
                f"ag_catalog.agtype_access_operator("
                f"VARIADIC ARRAY[properties, '\"user_name\"'::agtype])::text"
                f" = ANY(%s::text[])"
                f" AND (ag_catalog.agtype_access_operator("
                f"VARIADIC ARRAY[properties, '\"file_ids\"'::agtype])::jsonb)"
                f" ?| %s::text[]"
                f" RETURNING 1"
                f")"
            )
            count_parts.append(f"(SELECT count(*) FROM {cte_name})")

        sql = (
                "WITH "
                + ", ".join(cte_parts)
                + " SELECT "
                + " + ".join(count_parts)
                + " AS total_deleted"
        )
        logger.info(f"delete_by_cube_and_file_ids query sql: {sql},params:{params}")

        total_deleted = 0
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(sql, params)
                row = cursor.fetchone()
                total_deleted = int(row[0]) if row and row[0] is not None else 0
        except Exception as e:
            logger.error("delete_by_cube_and_file_ids failed: %s", e, exc_info=True)
            raise

        elapsed_ms = (time.time() - batch_start_time) * 1000.0
        logger.info(
            "delete_by_cube_and_file_ids completed in %.2fms",
            elapsed_ms,
        )
        return total_deleted


    def _delete_by_filter(self, filter: dict) -> int:
        filter_conditions = self._build_filter_conditions_sql(filter)
        if not filter_conditions:
            logger.warning("_delete_by_filter produced no WHERE conditions, skip")
            return 0
        logger.info("_delete_by_filter filter_conditions=%s", filter_conditions)

        where_clause = " AND ".join(filter_conditions)
        target_tables = self._resolve_shards_from_filter(filter)
        if not target_tables:
            return 0

        batch_start_time = time.time()

        if len(target_tables) == 1:
            tbl = target_tables[0]
            sql = f'DELETE FROM {tbl}."Memory" WHERE {where_clause}'
        else:
            cte_parts: list[str] = []
            count_parts: list[str] = []
            for idx, tbl in enumerate(target_tables):
                cte_name = f"d{idx}"
                cte_parts.append(
                    f"{cte_name} AS ("
                    f'DELETE FROM {tbl}."Memory" WHERE {where_clause}'
                    f" RETURNING 1"
                    f")"
                )
                count_parts.append(f"(SELECT count(*) FROM {cte_name})")
            sql = (
                "WITH "
                + ", ".join(cte_parts)
                + " SELECT "
                + " + ".join(count_parts)
                + " AS total_deleted"
            )
        logger.info("_delete_by_filter sql=%s", sql)
        total_deleted = 0
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(sql)
                if len(target_tables) == 1:
                    total_deleted = cursor.rowcount
                else:
                    row = cursor.fetchone()
                    total_deleted = int(row[0]) if row and row[0] is not None else 0
        except Exception as e:
            logger.error("delete_by_filter failed: %s", e, exc_info=True)
            raise

        elapsed_ms = (time.time() - batch_start_time) * 1000.0
        logger.info(
            "delete_by_filter completed in %.2fms, deleted %d nodes"
            " (shards=%d, single round-trip)",
            elapsed_ms,
            total_deleted,
            len(target_tables),
        )
        return total_deleted

    def _resolve_shards_from_filter(self, filter: dict) -> list[str]:
        if not isinstance(filter, dict):
            return self._get_all_shard_table_names()

        user_name_value = filter.get("user_name")
        if isinstance(user_name_value, str) and user_name_value:
            shard = self.get_memory_graph_table_name(user_name_value)
            logger.info(
                "_delete_by_filter routed to single shard %s via user_name=%s",
                shard,
                user_name_value,
            )
            return [shard]

        if isinstance(user_name_value, list) and user_name_value and all(
            isinstance(v, str) and v for v in user_name_value
        ):
            shards = list({self.get_memory_graph_table_name(v) for v in user_name_value})
            logger.info(
                "_delete_by_filter routed to %d shards via user_name in %s",
                len(shards),
                user_name_value,
            )
            return shards

        return self._get_all_shard_table_names()

    @timed
    def get_user_names_by_memory_ids(self, memory_ids: list[str]) -> dict[str, str | None]:
        logger.info(f"[get_user_names_by_memory_ids] Querying memory_ids {memory_ids}")
        if not memory_ids:
            return {}

        normalized_memory_ids = []
        for mid in memory_ids:
            if not isinstance(mid, str):
                mid = str(mid)
            mid = mid.strip()
            if mid:
                normalized_memory_ids.append(mid)

        if not normalized_memory_ids:
            return {}

        def escape_memory_id(mid: str) -> str:
            mid_str = mid.replace("\\", "\\\\")
            mid_str = mid_str.replace('"', '\\"')
            return mid_str

        id_conditions = []
        for mid in normalized_memory_ids:
            escaped_mid = escape_memory_id(mid)
            id_conditions.append(
                f"ag_catalog.agtype_access_operator(properties, '\"id\"'::agtype) = '\"{escaped_mid}\"'::agtype"
            )

        where_clause = f"({' OR '.join(id_conditions)})"

        shard_queries = []
        for shard_tbl in self._get_all_shard_table_names():
            shard_queries.append(f"""
            SELECT
                ag_catalog.agtype_access_operator(properties, '\"id\"'::agtype)::text AS memory_id,
                ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype)::text AS user_name
            FROM {shard_tbl}."Memory"
            WHERE {where_clause}
            """)
        query = " UNION ALL ".join(shard_queries)

        logger.info(f"[get_user_names_by_memory_ids] query: {query}")
        result_dict = {}
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()

                for row in results:
                    memory_id_raw = row[0]
                    user_name_raw = row[1]

                    if isinstance(memory_id_raw, str):
                        memory_id = memory_id_raw.strip('"').strip("'")
                    else:
                        memory_id = str(memory_id_raw).strip('"').strip("'")

                    if isinstance(user_name_raw, str):
                        user_name = user_name_raw.strip('"').strip("'")
                    else:
                        user_name = (
                            str(user_name_raw).strip('"').strip("'") if user_name_raw else None
                        )

                    result_dict[memory_id] = user_name if user_name else None

                for mid in normalized_memory_ids:
                    if mid not in result_dict:
                        result_dict[mid] = None

                logger.info(
                    f"[get_user_names_by_memory_ids] Found {len([v for v in result_dict.values() if v is not None])} memory_ids with user_names, "
                    f"{len([v for v in result_dict.values() if v is None])} memory_ids without user_names"
                )

                return result_dict
        except Exception as e:
            logger.error(
                f"[get_user_names_by_memory_ids] Failed to get user names: {e}", exc_info=True
            )
            raise

    def exist_user_name(self, user_name: str) -> dict[str, bool]:
        logger.info(f"[exist_user_name] Querying user_name {user_name}")
        if not user_name:
            return {user_name: False}

        tbl = self.get_memory_graph_table_name(user_name)

        def escape_user_name(un: str) -> str:
            un_str = un.replace("\\", "\\\\")
            un_str = un_str.replace('"', '\\"')
            return un_str

        escaped_un = escape_user_name(user_name)

        query = f"""
            SELECT COUNT(*)
            FROM {tbl}."Memory"
            WHERE ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = '\"{escaped_un}\"'::agtype
        """
        logger.info(f"[exist_user_name] query: {query}")
        result_dict = {}
        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                cursor.execute(query)
                count = cursor.fetchone()[0]
                result = count > 0
                result_dict[user_name] = result
                return result_dict
        except Exception as e:
            logger.error(
                f"[exist_user_name] Failed to check user_name existence: {e}", exc_info=True
            )
            raise

    @timed
    def delete_node_by_mem_cube_id(
        self,
        mem_cube_id: str,
        delete_record_id: str,
        hard_delete: bool = False,
    ) -> int:
        logger.info(
            f"delete_node_by_mem_cube_id mem_cube_id:{mem_cube_id}, "
            f"delete_record_id:{delete_record_id}, hard_delete:{hard_delete}"
        )

        if not mem_cube_id:
            raise ValueError("delete_node_by_mem_cube_id mem_cube_id is required but not provided")
        if not delete_record_id:
            raise ValueError(
                "delete_node_by_mem_cube_id delete_record_id is required but not provided"
            )
        tbl = self.get_memory_graph_table_name(mem_cube_id)

        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                user_name_condition = "ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = %s::agtype"

                user_name_param = self.format_param_value(mem_cube_id)

                if hard_delete:
                    delete_record_id_condition = "ag_catalog.agtype_access_operator(properties, '\"delete_record_id\"'::agtype) = %s::agtype"
                    where_clause = f"{user_name_condition} AND {delete_record_id_condition}"

                    where_params = [user_name_param, self.format_param_value(delete_record_id)]

                    delete_query = f"""
                        DELETE FROM {tbl}."Memory"
                        WHERE {where_clause}
                    """
                    logger.info(f"[delete_node_by_mem_cube_id] Hard delete query: {delete_query}")

                    cursor.execute(delete_query, where_params)
                    deleted_count = cursor.rowcount

                    logger.info(f"[delete_node_by_mem_cube_id] Hard deleted {deleted_count} nodes")
                    return deleted_count
                else:
                    delete_time_empty_condition = (
                        "(ag_catalog.agtype_access_operator(properties, '\"delete_time\"'::agtype) IS NULL "
                        "OR ag_catalog.agtype_access_operator(properties, '\"delete_time\"'::agtype) = '\"\"'::agtype)"
                    )
                    delete_record_id_empty_condition = (
                        "(ag_catalog.agtype_access_operator(properties, '\"delete_record_id\"'::agtype) IS NULL "
                        "OR ag_catalog.agtype_access_operator(properties, '\"delete_record_id\"'::agtype) = '\"\"'::agtype)"
                    )
                    where_clause = f"{user_name_condition} AND {delete_time_empty_condition} AND {delete_record_id_empty_condition}"

                    current_time = datetime.utcnow().isoformat()
                    update_query = f"""
                        UPDATE {tbl}."Memory"
                        SET properties = (
                            properties::jsonb || %s::jsonb
                        )::text::agtype,
                        deletetime = %s
                        WHERE {where_clause}
                    """
                    update_properties = {
                        "status": "deleted",
                        "delete_time": current_time,
                        "delete_record_id": delete_record_id,
                    }
                    logger.info(
                        f"delete_node_by_mem_cube_id Soft delete update_query:{update_query},update_properties:{update_properties},deletetime:{current_time}"
                    )
                    update_params = [
                        json.dumps(update_properties),
                        current_time,
                        user_name_param,
                    ]
                    cursor.execute(update_query, update_params)
                    updated_count = cursor.rowcount

                    logger.info(
                        f"delete_node_by_mem_cube_id Soft deleted (updated) {updated_count} nodes"
                    )
                    return updated_count

        except Exception as e:
            logger.error(
                f"[delete_node_by_mem_cube_id] Failed to delete/update nodes: {e}", exc_info=True
            )
            raise

    @timed
    def recover_memory_by_mem_cube_id(
        self,
        mem_cube_id: str,
        delete_record_id: str,
    ) -> int:
        logger.info(
            f"recover_memory_by_mem_cube_id mem_cube_id:{mem_cube_id},delete_record_id:{delete_record_id}"
        )
        if not mem_cube_id:
            raise ValueError(
                "recover_memory_by_mem_cube_id mem_cube_id is required but not provided"
            )

        if not delete_record_id:
            raise ValueError(
                "recover_memory_by_mem_cube_id delete_record_id is required but not provided"
            )
        tbl = self.get_memory_graph_table_name(mem_cube_id)

        logger.info(
            f"recover_memory_by_mem_cube_id mem_cube_id={mem_cube_id}, "
            f"delete_record_id={delete_record_id}"
        )

        try:
            with self._get_connection() as conn, conn.cursor() as cursor:
                user_name_condition = "ag_catalog.agtype_access_operator(properties, '\"user_name\"'::agtype) = %s::agtype"
                delete_record_id_condition = "ag_catalog.agtype_access_operator(properties, '\"delete_record_id\"'::agtype) = %s::agtype"
                where_clause = f"{user_name_condition} AND {delete_record_id_condition}"

                where_params = [
                    self.format_param_value(mem_cube_id),
                    self.format_param_value(delete_record_id),
                ]

                update_properties = {
                    "status": "activated",
                    "delete_record_id": "",
                    "delete_time": "",
                }

                update_query = f"""
                    UPDATE {tbl}."Memory"
                    SET properties = (
                        properties::jsonb || %s::jsonb
                    )::text::agtype,
                    deletetime = NULL
                    WHERE {where_clause}
                """

                logger.info(f"[recover_memory_by_mem_cube_id] Update query: {update_query}")
                logger.info(
                    f"[recover_memory_by_mem_cube_id] update_properties: {update_properties}"
                )

                update_params = [json.dumps(update_properties), *where_params]
                cursor.execute(update_query, update_params)
                updated_count = cursor.rowcount

                logger.info(
                    f"[recover_memory_by_mem_cube_id] Recovered (updated) {updated_count} nodes"
                )
                return updated_count

        except Exception as e:
            logger.error(
                f"[recover_memory_by_mem_cube_id] Failed to recover nodes: {e}", exc_info=True
            )
            raise
