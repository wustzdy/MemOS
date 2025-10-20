import os
import traceback

from typing import Any

from fastapi import APIRouter, HTTPException

from memos.api.config import APIConfig
from memos.api.product_models import (
    APIADDRequest,
    APIChatCompleteRequest,
    APISearchRequest,
    MemoryResponse,
    SearchResponse,
)
from memos.configs.embedder import EmbedderConfigFactory
from memos.configs.graph_db import GraphDBConfigFactory
from memos.configs.internet_retriever import InternetRetrieverConfigFactory
from memos.configs.llm import LLMConfigFactory
from memos.configs.mem_reader import MemReaderConfigFactory
from memos.configs.reranker import RerankerConfigFactory
from memos.embedders.factory import EmbedderFactory
from memos.graph_dbs.factory import GraphStoreFactory
from memos.llms.factory import LLMFactory
from memos.log import get_logger
from memos.mem_cube.navie import NaiveMemCube
from memos.mem_os.product_server import MOSServer
from memos.mem_reader.factory import MemReaderFactory
from memos.memories.textual.tree_text_memory.organize.manager import MemoryManager
from memos.memories.textual.tree_text_memory.retrieve.internet_retriever_factory import (
    InternetRetrieverFactory,
)
from memos.reranker.factory import RerankerFactory
from memos.types import MOSSearchResult, UserContext


logger = get_logger(__name__)

router = APIRouter(prefix="/product", tags=["Server API"])


def _build_graph_db_config(user_id: str = "default") -> dict[str, Any]:
    """Build graph database configuration."""
    graph_db_backend_map = {
        "neo4j-community": APIConfig.get_neo4j_community_config(user_id=user_id),
        "neo4j": APIConfig.get_neo4j_config(user_id=user_id),
        "nebular": APIConfig.get_nebular_config(user_id=user_id),
        "polardb": APIConfig.get_polardb_config(user_id=user_id),
    }

    graph_db_backend = os.getenv("NEO4J_BACKEND", "nebular").lower()
    return GraphDBConfigFactory.model_validate(
        {
            "backend": graph_db_backend,
            "config": graph_db_backend_map[graph_db_backend],
        }
    )


def _build_llm_config() -> dict[str, Any]:
    """Build LLM configuration."""
    return LLMConfigFactory.model_validate(
        {
            "backend": "openai",
            "config": APIConfig.get_openai_config(),
        }
    )


def _build_embedder_config() -> dict[str, Any]:
    """Build embedder configuration."""
    return EmbedderConfigFactory.model_validate(APIConfig.get_embedder_config())


def _build_mem_reader_config() -> dict[str, Any]:
    """Build memory reader configuration."""
    return MemReaderConfigFactory.model_validate(
        APIConfig.get_product_default_config()["mem_reader"]
    )


def _build_reranker_config() -> dict[str, Any]:
    """Build reranker configuration."""
    return RerankerConfigFactory.model_validate(APIConfig.get_reranker_config())


def _build_internet_retriever_config() -> dict[str, Any]:
    """Build internet retriever configuration."""
    return InternetRetrieverConfigFactory.model_validate(APIConfig.get_internet_config())


def _get_default_memory_size(cube_config) -> dict[str, int]:
    """Get default memory size configuration."""
    return getattr(cube_config.text_mem.config, "memory_size", None) or {
        "WorkingMemory": 20,
        "LongTermMemory": 1500,
        "UserMemory": 480,
    }


def init_server():
    """Initialize server components and configurations."""
    # Get default cube configuration
    default_cube_config = APIConfig.get_default_cube_config()

    # Build component configurations
    graph_db_config = _build_graph_db_config()
    print(graph_db_config)
    llm_config = _build_llm_config()
    embedder_config = _build_embedder_config()
    mem_reader_config = _build_mem_reader_config()
    reranker_config = _build_reranker_config()
    internet_retriever_config = _build_internet_retriever_config()

    # Create component instances
    graph_db = GraphStoreFactory.from_config(graph_db_config)
    llm = LLMFactory.from_config(llm_config)
    embedder = EmbedderFactory.from_config(embedder_config)
    mem_reader = MemReaderFactory.from_config(mem_reader_config)
    reranker = RerankerFactory.from_config(reranker_config)
    internet_retriever = InternetRetrieverFactory.from_config(
        internet_retriever_config, embedder=embedder
    )

    # Initialize memory manager
    memory_manager = MemoryManager(
        graph_db,
        embedder,
        llm,
        memory_size=_get_default_memory_size(default_cube_config),
        is_reorganize=getattr(default_cube_config.text_mem.config, "reorganize", False),
    )
    mos_server = MOSServer(
        mem_reader=mem_reader,
        llm=llm,
        online_bot=False,
    )
    return (
        graph_db,
        mem_reader,
        llm,
        embedder,
        reranker,
        internet_retriever,
        memory_manager,
        default_cube_config,
        mos_server,
    )


# Initialize global components
(
    graph_db,
    mem_reader,
    llm,
    embedder,
    reranker,
    internet_retriever,
    memory_manager,
    default_cube_config,
    mos_server,
) = init_server()


def _create_naive_mem_cube() -> NaiveMemCube:
    """Create a NaiveMemCube instance with initialized components."""
    naive_mem_cube = NaiveMemCube(
        llm=llm,
        embedder=embedder,
        mem_reader=mem_reader,
        graph_db=graph_db,
        reranker=reranker,
        internet_retriever=internet_retriever,
        memory_manager=memory_manager,
        default_cube_config=default_cube_config,
    )
    return naive_mem_cube


def _format_memory_item(memory_data: Any) -> dict[str, Any]:
    """Format a single memory item for API response."""
    memory = memory_data.model_dump()
    memory_id = memory["id"]
    ref_id = f"[{memory_id.split('-')[0]}]"

    memory["ref_id"] = ref_id
    memory["metadata"]["embedding"] = []
    memory["metadata"]["sources"] = []
    memory["metadata"]["ref_id"] = ref_id
    memory["metadata"]["id"] = memory_id
    memory["metadata"]["memory"] = memory["memory"]

    return memory


@router.post("/search", summary="Search memories", response_model=SearchResponse)
def search_memories(search_req: APISearchRequest):
    """Search memories for a specific user."""
    # Create UserContext object - how to assign values
    user_context = UserContext(
        user_id=search_req.user_id,
        mem_cube_id=search_req.mem_cube_id,
        session_id=search_req.session_id or "default_session",
    )
    logger.info(f"Search user_id is: {user_context.mem_cube_id}")
    memories_result: MOSSearchResult = {
        "text_mem": [],
        "act_mem": [],
        "para_mem": [],
    }
    target_session_id = search_req.session_id
    if not target_session_id:
        target_session_id = "default_session"
    search_filter = {"session_id": search_req.session_id} if search_req.session_id else None

    # Create MemCube and perform search
    naive_mem_cube = _create_naive_mem_cube()
    search_results = naive_mem_cube.text_mem.search(
        query=search_req.query,
        user_name=user_context.mem_cube_id,
        top_k=search_req.top_k,
        mode=search_req.mode,
        manual_close_internet=not search_req.internet_search,
        moscube=search_req.moscube,
        search_filter=search_filter,
        info={
            "user_id": search_req.user_id,
            "session_id": target_session_id,
            "chat_history": search_req.chat_history,
        },
    )
    formatted_memories = [_format_memory_item(data) for data in search_results]

    memories_result["text_mem"].append(
        {
            "cube_id": search_req.mem_cube_id,
            "memories": formatted_memories,
        }
    )

    return SearchResponse(
        message="Search completed successfully",
        data=memories_result,
    )


@router.post("/add", summary="Add memories", response_model=MemoryResponse)
def add_memories(add_req: APIADDRequest):
    """Add memories for a specific user."""
    # Create UserContext object - how to assign values
    user_context = UserContext(
        user_id=add_req.user_id,
        mem_cube_id=add_req.mem_cube_id,
        session_id=add_req.session_id or "default_session",
    )
    naive_mem_cube = _create_naive_mem_cube()
    target_session_id = add_req.session_id
    if not target_session_id:
        target_session_id = "default_session"
    memories = mem_reader.get_memory(
        [add_req.messages],
        type="chat",
        info={
            "user_id": add_req.user_id,
            "session_id": target_session_id,
        },
    )

    # Flatten memory list
    flattened_memories = [mm for m in memories for mm in m]
    logger.info(f"Memory extraction completed for user {add_req.user_id}")
    mem_id_list: list[str] = naive_mem_cube.text_mem.add(
        flattened_memories,
        user_name=user_context.mem_cube_id,
    )

    logger.info(
        f"Added {len(mem_id_list)} memories for user {add_req.user_id} "
        f"in session {add_req.session_id}: {mem_id_list}"
    )
    response_data = [
        {
            "memory": memory.memory,
            "memory_id": memory_id,
            "memory_type": memory.metadata.memory_type,
        }
        for memory_id, memory in zip(mem_id_list, flattened_memories, strict=False)
    ]
    return MemoryResponse(
        message="Memory added successfully",
        data=response_data,
    )


@router.post("/chat/complete", summary="Chat with MemOS (Complete Response)")
def chat_complete(chat_req: APIChatCompleteRequest):
    """Chat with MemOS for a specific user. Returns complete response (non-streaming)."""
    try:
        # Collect all responses from the generator
        naive_mem_cube = _create_naive_mem_cube()
        content, references = mos_server.chat(
            query=chat_req.query,
            user_id=chat_req.user_id,
            cube_id=chat_req.mem_cube_id,
            mem_cube=naive_mem_cube,
            history=chat_req.history,
            internet_search=chat_req.internet_search,
            moscube=chat_req.moscube,
            base_prompt=chat_req.base_prompt,
            top_k=chat_req.top_k,
            threshold=chat_req.threshold,
            session_id=chat_req.session_id,
        )

        # Return the complete response
        return {
            "message": "Chat completed successfully",
            "data": {"response": content, "references": references},
        }

    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(traceback.format_exc())) from err
    except Exception as err:
        logger.error(f"Failed to start chat: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(traceback.format_exc())) from err
