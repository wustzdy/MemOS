import time

from datetime import datetime
from typing import TYPE_CHECKING, Any

from memos.configs.memory import TreeTextMemoryConfig
from memos.embedders.base import BaseEmbedder
from memos.graph_dbs.base import BaseGraphDB
from memos.llms.base import BaseLLM
from memos.log import get_logger
from memos.mem_reader.base import BaseMemReader
from memos.memories.textual.item import TextualMemoryItem, TreeNodeTextualMemoryMetadata
from memos.memories.textual.tree import TreeTextMemory
from memos.memories.textual.tree_text_memory.organize.manager import MemoryManager
from memos.memories.textual.tree_text_memory.retrieve.bm25_util import EnhancedBM25
from memos.memories.textual.tree_text_memory.retrieve.searcher import Searcher
from memos.reranker.base import BaseReranker
from memos.types import MessageList


if TYPE_CHECKING:
    from memos.embedders.factory import OllamaEmbedder
    from memos.graph_dbs.factory import Neo4jGraphDB
    from memos.llms.factory import AzureLLM, OllamaLLM, OpenAILLM


logger = get_logger(__name__)


class SimpleTreeTextMemory(TreeTextMemory):
    """General textual memory implementation for storing and retrieving memories."""

    def __init__(
        self,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        mem_reader: BaseMemReader,
        graph_db: BaseGraphDB,
        reranker: BaseReranker,
        memory_manager: MemoryManager,
        config: TreeTextMemoryConfig,
        internet_retriever: None = None,
        is_reorganize: bool = False,
    ):
        """Initialize memory with the given configuration."""
        time_start = time.time()
        self.config: TreeTextMemoryConfig = config
        self.mode = self.config.mode
        logger.info(f"Tree mode is {self.mode}")

        self.extractor_llm: OpenAILLM | OllamaLLM | AzureLLM = llm
        logger.info(f"time init: extractor_llm time is: {time.time() - time_start}")

        time_start_ex = time.time()
        self.dispatcher_llm: OpenAILLM | OllamaLLM | AzureLLM = llm
        logger.info(f"time init: dispatcher_llm time is: {time.time() - time_start_ex}")

        time_start_em = time.time()
        self.embedder: OllamaEmbedder = embedder
        logger.info(f"time init: embedder time is: {time.time() - time_start_em}")

        time_start_gs = time.time()
        self.graph_store: Neo4jGraphDB = graph_db
        logger.info(f"time init: graph_store time is: {time.time() - time_start_gs}")

        time_start_bm = time.time()
        self.search_strategy = config.search_strategy
        self.bm25_retriever = (
            EnhancedBM25() if self.search_strategy and self.search_strategy["bm25"] else None
        )
        logger.info(f"time init: bm25_retriever time is: {time.time() - time_start_bm}")

        time_start_rr = time.time()
        self.reranker = reranker
        logger.info(f"time init: reranker time is: {time.time() - time_start_rr}")

        time_start_mm = time.time()
        self.memory_manager: MemoryManager = memory_manager
        logger.info(f"time init: memory_manager time is: {time.time() - time_start_mm}")
        time_start_ir = time.time()
        # Create internet retriever if configured
        self.internet_retriever = None
        if config.internet_retriever is not None:
            self.internet_retriever = internet_retriever
            logger.info(
                f"Internet retriever initialized with backend: {config.internet_retriever.backend}"
            )
        else:
            logger.info("No internet retriever configured")
        logger.info(f"time init: internet_retriever time is: {time.time() - time_start_ir}")

    def replace_working_memory(
        self, memories: list[TextualMemoryItem], user_name: str | None = None
    ) -> None:
        self.memory_manager.replace_working_memory(memories, user_name=user_name)

    def get_working_memory(self, user_name: str | None = None) -> list[TextualMemoryItem]:
        working_memories = self.graph_store.get_all_memory_items(
            scope="WorkingMemory", user_name=user_name
        )
        items = [TextualMemoryItem.from_dict(record) for record in (working_memories)]
        # Sort by updated_at in descending order
        sorted_items = sorted(
            items, key=lambda x: x.metadata.updated_at or datetime.min, reverse=True
        )
        return sorted_items

    def get_current_memory_size(self, user_name: str | None = None) -> dict[str, int]:
        """
        Get the current size of each memory type.
        This delegates to the MemoryManager.
        """
        return self.memory_manager.get_current_memory_size(user_name=user_name)

    def get_searcher(
        self,
        manual_close_internet: bool = False,
        moscube: bool = False,
    ):
        if (self.internet_retriever is not None) and manual_close_internet:
            logger.warning(
                "Internet retriever is init by config , but  this search set manual_close_internet is True  and will close it"
            )
            searcher = Searcher(
                self.dispatcher_llm,
                self.graph_store,
                self.embedder,
                self.reranker,
                internet_retriever=None,
                moscube=moscube,
            )
        else:
            searcher = Searcher(
                self.dispatcher_llm,
                self.graph_store,
                self.embedder,
                self.reranker,
                internet_retriever=self.internet_retriever,
                moscube=moscube,
            )
        return searcher

    def search(
        self,
        query: str,
        top_k: int,
        info=None,
        mode: str = "fast",
        memory_type: str = "All",
        manual_close_internet: bool = False,
        moscube: bool = False,
        search_filter: dict | None = None,
        user_name: str | None = None,
    ) -> list[TextualMemoryItem]:
        """Search for memories based on a query.
        User query -> TaskGoalParser -> MemoryPathResolver ->
        GraphMemoryRetriever -> MemoryReranker -> MemoryReasoner -> Final output
        Args:
            query (str): The query to search for.
            top_k (int): The number of top results to return.
            info (dict): Leave a record of memory consumption.
            mode (str, optional): The mode of the search.
            - 'fast': Uses a faster search process, sacrificing some precision for speed.
            - 'fine': Uses a more detailed search process, invoking large models for higher precision, but slower performance.
            memory_type (str): Type restriction for search.
            ['All', 'WorkingMemory', 'LongTermMemory', 'UserMemory']
            manual_close_internet (bool): If True, the internet retriever will be closed by this search, it high priority than config.
            moscube (bool): whether you use moscube to answer questions
            search_filter (dict, optional): Optional metadata filters for search results.
                - Keys correspond to memory metadata fields (e.g., "user_id", "session_id").
                - Values are exact-match conditions.
                Example: {"user_id": "123", "session_id": "abc"}
                If None, no additional filtering is applied.
        Returns:
            list[TextualMemoryItem]: List of matching memories.
        """
        if (self.internet_retriever is not None) and manual_close_internet:
            searcher = Searcher(
                self.dispatcher_llm,
                self.graph_store,
                self.embedder,
                self.reranker,
                bm25_retriever=self.bm25_retriever,
                internet_retriever=None,
                moscube=moscube,
                search_strategy=self.search_strategy,
            )
        else:
            searcher = Searcher(
                self.dispatcher_llm,
                self.graph_store,
                self.embedder,
                self.reranker,
                bm25_retriever=self.bm25_retriever,
                internet_retriever=self.internet_retriever,
                moscube=moscube,
                search_strategy=self.search_strategy,
            )
        return searcher.search(
            query, top_k, info, mode, memory_type, search_filter, user_name=user_name
        )

    def get_relevant_subgraph(
        self, query: str, top_k: int = 5, depth: int = 2, center_status: str = "activated"
    ) -> dict[str, Any]:
        """
        Find and merge the local neighborhood sub-graphs of the top-k
        nodes most relevant to the query.
         Process:
             1. Embed the user query into a vector representation.
             2. Use vector similarity search to find the top-k similar nodes.
             3. For each similar node:
                 - Ensure its status matches `center_status` (e.g., 'active').
                 - Retrieve its local subgraph up to `depth` hops.
                 - Collect the center node, its neighbors, and connecting edges.
             4. Merge all retrieved subgraphs into a single unified subgraph.
             5. Return the merged subgraph structure.

         Args:
             query (str): The user input or concept to find relevant memories for.
             top_k (int, optional): How many top similar nodes to retrieve. Default is 5.
             depth (int, optional): The neighborhood depth (number of hops). Default is 2.
             center_status (str, optional): Status condition the center node must satisfy (e.g., 'active').

         Returns:
             dict[str, Any]: A subgraph dict with:
                 - 'core_id': ID of the top matching core node, or None if none found.
                 - 'nodes': List of unique nodes (core + neighbors) in the merged subgraph.
                 - 'edges': List of unique edges (as dicts with 'from', 'to', 'type') in the merged subgraph.
        """
        # Step 1: Embed query
        query_embedding = self.embedder.embed([query])[0]

        # Step 2: Get top-1 similar node
        similar_nodes = self.graph_store.search_by_embedding(query_embedding, top_k=top_k)
        if not similar_nodes:
            logger.info("No similar nodes found for query embedding.")
            return {"core_id": None, "nodes": [], "edges": []}

        # Step 3: Fetch neighborhood
        all_nodes = {}
        all_edges = set()
        cores = []

        for node in similar_nodes:
            core_id = node["id"]
            score = node["score"]

            subgraph = self.graph_store.get_subgraph(
                center_id=core_id, depth=depth, center_status=center_status
            )

            if not subgraph["core_node"]:
                logger.info(f"Skipping node {core_id} (inactive or not found).")
                continue

            core_node = subgraph["core_node"]
            neighbors = subgraph["neighbors"]
            edges = subgraph["edges"]

            # Collect nodes
            all_nodes[core_node["id"]] = core_node
            for n in neighbors:
                all_nodes[n["id"]] = n

            # Collect edges
            for e in edges:
                all_edges.add((e["source"], e["target"], e["type"]))

            cores.append(
                {"id": core_id, "score": score, "core_node": core_node, "neighbors": neighbors}
            )

        top_core = cores[0]
        return {
            "core_id": top_core["id"],
            "nodes": list(all_nodes.values()),
            "edges": [{"source": f, "target": t, "type": ty} for (f, t, ty) in all_edges],
        }

    def extract(self, messages: MessageList) -> list[TextualMemoryItem]:
        raise NotImplementedError

    def update(self, memory_id: str, new_memory: TextualMemoryItem | dict[str, Any]) -> None:
        raise NotImplementedError

    def get(self, memory_id: str) -> TextualMemoryItem:
        """Get a memory by its ID."""
        result = self.graph_store.get_node(memory_id)
        if result is None:
            raise ValueError(f"Memory with ID {memory_id} not found")
        metadata_dict = result.get("metadata", {})
        return TextualMemoryItem(
            id=result["id"],
            memory=result["memory"],
            metadata=TreeNodeTextualMemoryMetadata(**metadata_dict),
        )

    def get_by_ids(self, memory_ids: list[str]) -> list[TextualMemoryItem]:
        raise NotImplementedError

    def delete_all(self) -> None:
        """Delete all memories and their relationships from the graph store."""
        try:
            self.graph_store.clear()
            logger.info("All memories and edges have been deleted from the graph.")
        except Exception as e:
            logger.error(f"An error occurred while deleting all memories: {e}")
            raise
