import json
import os

from collections import OrderedDict
from typing import TYPE_CHECKING

from memos.api.product_models import APISearchRequest
from memos.configs.mem_scheduler import GeneralSchedulerConfig
from memos.log import get_logger
from memos.mem_cube.general import GeneralMemCube
from memos.mem_cube.navie import NaiveMemCube
from memos.mem_scheduler.general_modules.api_misc import SchedulerAPIModule
from memos.mem_scheduler.general_scheduler import GeneralScheduler
from memos.mem_scheduler.schemas.general_schemas import (
    API_MIX_SEARCH_LABEL,
    MemCubeID,
    SearchMode,
    UserID,
)
from memos.mem_scheduler.schemas.message_schemas import ScheduleMessageItem
from memos.mem_scheduler.utils.api_utils import format_textual_memory_item
from memos.mem_scheduler.utils.db_utils import get_utc_now
from memos.memories.textual.tree import TextualMemoryItem, TreeTextMemory
from memos.types import UserContext


if TYPE_CHECKING:
    from memos.mem_scheduler.schemas.monitor_schemas import MemoryMonitorItem
    from memos.memories.textual.tree_text_memory.retrieve.searcher import Searcher
    from memos.reranker.http_bge import HTTPBGEReranker


logger = get_logger(__name__)


class OptimizedScheduler(GeneralScheduler):
    """Optimized scheduler with improved working memory management and support for api"""

    def __init__(self, config: GeneralSchedulerConfig):
        super().__init__(config)
        self.window_size = int(os.getenv("API_SEARCH_WINDOW_SIZE", 5))
        self.history_memory_turns = int(os.getenv("API_SEARCH_HISTORY_TURNS", 5))
        self.session_counter = OrderedDict()
        self.max_session_history = 5

        self.api_module = SchedulerAPIModule(
            window_size=self.window_size,
            history_memory_turns=self.history_memory_turns,
        )
        self.register_handlers(
            {
                API_MIX_SEARCH_LABEL: self._api_mix_search_message_consumer,
            }
        )

    def submit_memory_history_async_task(
        self,
        search_req: APISearchRequest,
        user_context: UserContext,
        session_id: str | None = None,
    ):
        # Create message for async fine search
        message_content = {
            "search_req": {
                "query": search_req.query,
                "user_id": search_req.user_id,
                "session_id": session_id,
                "top_k": search_req.top_k,
                "internet_search": search_req.internet_search,
                "moscube": search_req.moscube,
                "chat_history": search_req.chat_history,
            },
            "user_context": {"mem_cube_id": user_context.mem_cube_id},
        }

        async_task_id = f"mix_search_{search_req.user_id}_{get_utc_now().timestamp()}"

        # Get mem_cube for the message
        mem_cube = self.current_mem_cube

        message = ScheduleMessageItem(
            item_id=async_task_id,
            user_id=search_req.user_id,
            mem_cube_id=user_context.mem_cube_id,
            label=API_MIX_SEARCH_LABEL,
            mem_cube=mem_cube,
            content=json.dumps(message_content),
            timestamp=get_utc_now(),
        )

        # Submit async task
        self.submit_messages([message])
        logger.info(f"Submitted async fine search task for user {search_req.user_id}")
        return async_task_id

    def search_memories(
        self,
        search_req: APISearchRequest,
        user_context: UserContext,
        mem_cube: NaiveMemCube,
        mode: SearchMode,
    ):
        """Fine search memories function copied from server_router to avoid circular import"""
        target_session_id = search_req.session_id
        if not target_session_id:
            target_session_id = "default_session"
        search_filter = {"session_id": search_req.session_id} if search_req.session_id else None

        # Create MemCube and perform search
        search_results = mem_cube.text_mem.search(
            query=search_req.query,
            user_name=user_context.mem_cube_id,
            top_k=search_req.top_k,
            mode=mode,
            manual_close_internet=not search_req.internet_search,
            moscube=search_req.moscube,
            search_filter=search_filter,
            info={
                "user_id": search_req.user_id,
                "session_id": target_session_id,
                "chat_history": search_req.chat_history,
            },
        )
        return search_results

    def mix_search_memories(
        self,
        search_req: APISearchRequest,
        user_context: UserContext,
    ):
        """
        Mix search memories: fast search + async fine search
        """

        # Get mem_cube for fast search
        mem_cube = self.current_mem_cube

        target_session_id = search_req.session_id
        if not target_session_id:
            target_session_id = "default_session"
        search_filter = {"session_id": search_req.session_id} if search_req.session_id else None

        text_mem: TreeTextMemory = mem_cube.text_mem
        searcher: Searcher = text_mem.get_searcher(
            manual_close_internet=not search_req.internet_search,
            moscube=False,
        )
        # Rerank Memories - reranker expects TextualMemoryItem objects
        reranker: HTTPBGEReranker = text_mem.reranker
        info = {
            "user_id": search_req.user_id,
            "session_id": target_session_id,
            "chat_history": search_req.chat_history,
        }

        fast_retrieved_memories = searcher.retrieve(
            query=search_req.query,
            user_name=user_context.mem_cube_id,
            top_k=search_req.top_k,
            mode=SearchMode.FAST,
            manual_close_internet=not search_req.internet_search,
            moscube=search_req.moscube,
            search_filter=search_filter,
            info=info,
        )

        self.submit_memory_history_async_task(
            search_req=search_req,
            user_context=user_context,
            session_id=search_req.session_id,
        )

        # Try to get pre-computed fine memories if available
        history_memories = self.api_module.get_history_memories(
            user_id=search_req.user_id,
            mem_cube_id=user_context.mem_cube_id,
            turns=self.history_memory_turns,
        )

        if not history_memories:
            fast_memories = searcher.post_retrieve(
                retrieved_results=fast_retrieved_memories,
                top_k=search_req.top_k,
                user_name=user_context.mem_cube_id,
                info=info,
            )
            # Format fast memories for return
            formatted_memories = [format_textual_memory_item(data) for data in fast_memories]
            return formatted_memories

        sorted_history_memories = reranker.rerank(
            query=search_req.query,  # Use search_req.query instead of undefined query
            graph_results=history_memories,  # Pass TextualMemoryItem objects directly
            top_k=search_req.top_k,  # Use search_req.top_k instead of undefined top_k
            search_filter=search_filter,
        )

        sorted_results = fast_retrieved_memories + sorted_history_memories
        final_results = searcher.post_retrieve(
            retrieved_results=sorted_results,
            top_k=search_req.top_k,
            user_name=user_context.mem_cube_id,
            info=info,
        )

        formatted_memories = [
            format_textual_memory_item(item) for item in final_results[: search_req.top_k]
        ]

        return formatted_memories

    def update_search_memories_to_redis(
        self,
        messages: list[ScheduleMessageItem],
    ):
        mem_cube: NaiveMemCube = self.current_mem_cube

        for msg in messages:
            content_dict = json.loads(msg.content)
            search_req = content_dict["search_req"]
            user_context = content_dict["user_context"]

            session_id = search_req.get("session_id")
            if session_id:
                if session_id not in self.session_counter:
                    self.session_counter[session_id] = 0
                else:
                    self.session_counter[session_id] += 1
                session_turn = self.session_counter[session_id]

                # Move the current session to the end to mark it as recently used
                self.session_counter.move_to_end(session_id)

                # If the counter exceeds the max size, remove the oldest item
                if len(self.session_counter) > self.max_session_history:
                    self.session_counter.popitem(last=False)
            else:
                session_turn = 0

            memories: list[TextualMemoryItem] = self.search_memories(
                search_req=APISearchRequest(**content_dict["search_req"]),
                user_context=UserContext(**content_dict["user_context"]),
                mem_cube=mem_cube,
                mode=SearchMode.FAST,
            )
            formatted_memories = [format_textual_memory_item(data) for data in memories]

            # Sync search data to Redis
            self.api_module.sync_search_data(
                item_id=msg.item_id,
                user_id=search_req["user_id"],
                mem_cube_id=user_context["mem_cube_id"],
                query=search_req["query"],
                memories=memories,
                formatted_memories=formatted_memories,
                session_id=session_id,
                conversation_turn=session_turn,
            )

    def _api_mix_search_message_consumer(self, messages: list[ScheduleMessageItem]) -> None:
        """
        Process and handle query trigger messages from the queue.

        Args:
            messages: List of query messages to process
        """
        logger.info(f"Messages {messages} assigned to {API_MIX_SEARCH_LABEL} handler.")

        # Process the query in a session turn
        grouped_messages = self.dispatcher._group_messages_by_user_and_mem_cube(messages=messages)

        self.validate_schedule_messages(messages=messages, label=API_MIX_SEARCH_LABEL)

        for user_id in grouped_messages:
            for mem_cube_id in grouped_messages[user_id]:
                messages = grouped_messages[user_id][mem_cube_id]
                if len(messages) == 0:
                    return
                self.update_search_memories_to_redis(messages=messages)

    def replace_working_memory(
        self,
        user_id: UserID | str,
        mem_cube_id: MemCubeID | str,
        mem_cube: GeneralMemCube,
        original_memory: list[TextualMemoryItem],
        new_memory: list[TextualMemoryItem],
    ) -> None | list[TextualMemoryItem]:
        """Replace working memory with new memories after reranking."""
        text_mem_base = mem_cube.text_mem
        if isinstance(text_mem_base, TreeTextMemory):
            text_mem_base: TreeTextMemory = text_mem_base

            # process rerank memories with llm
            query_db_manager = self.monitor.query_monitors[user_id][mem_cube_id]
            # Sync with database to get latest query history
            query_db_manager.sync_with_orm()

            query_history = query_db_manager.obj.get_queries_with_timesort()
            memories_with_new_order, rerank_success_flag = (
                self.retriever.process_and_rerank_memories(
                    queries=query_history,
                    original_memory=original_memory,
                    new_memory=new_memory,
                    top_k=self.top_k,
                )
            )

            # Apply combined filtering (unrelated + redundant)
            logger.info(
                f"Applying combined unrelated and redundant memory filtering to {len(memories_with_new_order)} memories"
            )
            filtered_memories, filtering_success_flag = (
                self.retriever.filter_unrelated_and_redundant_memories(
                    query_history=query_history,
                    memories=memories_with_new_order,
                )
            )

            if filtering_success_flag:
                logger.info(
                    f"Combined filtering completed successfully. "
                    f"Filtered from {len(memories_with_new_order)} to {len(filtered_memories)} memories"
                )
                memories_with_new_order = filtered_memories
            else:
                logger.warning(
                    "Combined filtering failed - keeping memories as fallback. "
                    f"Count: {len(memories_with_new_order)}"
                )

            # Update working memory monitors
            query_keywords = query_db_manager.obj.get_keywords_collections()
            logger.info(
                f"Processing {len(memories_with_new_order)} memories with {len(query_keywords)} query keywords"
            )
            new_working_memory_monitors = self.transform_working_memories_to_monitors(
                query_keywords=query_keywords,
                memories=memories_with_new_order,
            )

            if not rerank_success_flag:
                for one in new_working_memory_monitors:
                    one.sorting_score = 0

            logger.info(f"update {len(new_working_memory_monitors)} working_memory_monitors")
            self.monitor.update_working_memory_monitors(
                new_working_memory_monitors=new_working_memory_monitors,
                user_id=user_id,
                mem_cube_id=mem_cube_id,
                mem_cube=mem_cube,
            )

            # Use the filtered and reranked memories directly
            text_mem_base.replace_working_memory(memories=memories_with_new_order)

            # Update monitor after replacing working memory
            mem_monitors: list[MemoryMonitorItem] = self.monitor.working_memory_monitors[user_id][
                mem_cube_id
            ].obj.get_sorted_mem_monitors(reverse=True)
            new_working_memories = [mem_monitor.tree_memory_item for mem_monitor in mem_monitors]

            logger.info(
                f"The working memory has been replaced with {len(memories_with_new_order)} new memories."
            )
            self.log_working_memory_replacement(
                original_memory=original_memory,
                new_memory=new_working_memories,
                user_id=user_id,
                mem_cube_id=mem_cube_id,
                mem_cube=mem_cube,
                log_func_callback=self._submit_web_logs,
            )
        else:
            logger.error("memory_base is not supported")
            memories_with_new_order = new_memory

        return memories_with_new_order
