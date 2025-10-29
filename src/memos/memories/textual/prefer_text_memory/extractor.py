import json
import uuid

from abc import ABC, abstractmethod
from concurrent.futures import as_completed
from datetime import datetime
from typing import Any

from memos.context.context import ContextThreadPoolExecutor
from memos.log import get_logger
from memos.memories.textual.item import PreferenceTextualMemoryMetadata, TextualMemoryItem
from memos.memories.textual.prefer_text_memory.spliter import Splitter
from memos.memories.textual.prefer_text_memory.utils import convert_messages_to_string
from memos.templates.prefer_complete_prompt import (
    NAIVE_EXPLICIT_PREFERENCE_EXTRACT_PROMPT,
    NAIVE_IMPLICIT_PREFERENCE_EXTRACT_PROMPT,
)
from memos.types import MessageList


logger = get_logger(__name__)


class BaseExtractor(ABC):
    """Abstract base class for extractors."""

    @abstractmethod
    def __init__(self, llm_provider=None, embedder=None, vector_db=None):
        """Initialize the extractor."""


class NaiveExtractor(BaseExtractor):
    """Extractor."""

    def __init__(self, llm_provider=None, embedder=None, vector_db=None):
        """Initialize the extractor."""
        super().__init__(llm_provider, embedder, vector_db)
        self.llm_provider = llm_provider
        self.embedder = embedder
        self.vector_db = vector_db
        self.splitter = Splitter()

    def extract_basic_info(self, qa_pair: MessageList) -> dict[str, Any]:
        """Extract basic information from a QA pair (no LLM needed)."""
        basic_info = {
            "dialog_id": str(uuid.uuid4()),
            "dialog_str": convert_messages_to_string(qa_pair),
            "created_at": datetime.now().isoformat(),
        }

        return basic_info

    def extract_explicit_preference(self, qa_pair: MessageList | str) -> dict[str, Any] | None:
        """Extract explicit preference from a QA pair."""
        qa_pair_str = convert_messages_to_string(qa_pair) if isinstance(qa_pair, list) else qa_pair
        prompt = NAIVE_EXPLICIT_PREFERENCE_EXTRACT_PROMPT.replace("{qa_pair}", qa_pair_str)

        try:
            response = self.llm_provider.generate([{"role": "user", "content": prompt}])
            response = response.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(response)
            return result
        except Exception as e:
            logger.error(f"Error extracting explicit preference: {e}, return None")
            return None

    def extract_implicit_preference(self, qa_pair: MessageList | str) -> dict[str, Any] | None:
        """Extract implicit preferences from cluster qa pairs."""
        if not qa_pair:
            return None
        qa_pair_str = convert_messages_to_string(qa_pair) if isinstance(qa_pair, list) else qa_pair
        prompt = NAIVE_IMPLICIT_PREFERENCE_EXTRACT_PROMPT.replace("{qa_pair}", qa_pair_str)

        try:
            response = self.llm_provider.generate([{"role": "user", "content": prompt}])
            response = response.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(response)
            return result
        except Exception as e:
            logger.error(f"Error extracting implicit preferences: {e}, return None")
            return None

    def _process_single_chunk_explicit(
        self, chunk: MessageList, msg_type: str, info: dict[str, Any]
    ) -> TextualMemoryItem | None:
        """Process a single chunk and return a TextualMemoryItem."""
        basic_info = self.extract_basic_info(chunk)
        if not basic_info["dialog_str"]:
            return None

        explicit_pref = self.extract_explicit_preference(basic_info["dialog_str"])
        if not explicit_pref:
            return None

        memories = []
        for pref in explicit_pref:
            vector_info = {
                "embedding": self.embedder.embed([pref["context_summary"]])[0],
            }
            extract_info = {**basic_info, **pref, **vector_info, **info}

            metadata = PreferenceTextualMemoryMetadata(
                type=msg_type, preference_type="explicit_preference", **extract_info
            )
            memory = TextualMemoryItem(
                id=str(uuid.uuid4()), memory=pref["context_summary"], metadata=metadata
            )

            memories.append(memory)

        return memories

    def _process_single_chunk_implicit(
        self, chunk: MessageList, msg_type: str, info: dict[str, Any]
    ) -> TextualMemoryItem | None:
        basic_info = self.extract_basic_info(chunk)
        if not basic_info["dialog_str"]:
            return None
        implicit_pref = self.extract_implicit_preference(basic_info["dialog_str"])
        if not implicit_pref:
            return None

        vector_info = {
            "embedding": self.embedder.embed([implicit_pref["context_summary"]])[0],
        }

        extract_info = {**basic_info, **implicit_pref, **vector_info, **info}

        metadata = PreferenceTextualMemoryMetadata(
            type=msg_type, preference_type="implicit_preference", **extract_info
        )
        memory = TextualMemoryItem(
            id=extract_info["dialog_id"], memory=implicit_pref["context_summary"], metadata=metadata
        )

        return memory

    def extract(
        self,
        messages: list[MessageList],
        msg_type: str,
        info: dict[str, Any],
        max_workers: int = 10,
    ) -> list[TextualMemoryItem]:
        """Extract preference memories based on the messages using thread pool for acceleration."""
        chunks: list[MessageList] = []
        for message in messages:
            chunk = self.splitter.split_chunks(message, split_type="overlap")
            chunks.extend(chunk)
        if not chunks:
            return []

        memories = []
        with ContextThreadPoolExecutor(max_workers=min(max_workers, len(chunks))) as executor:
            futures = {
                executor.submit(self._process_single_chunk_explicit, chunk, msg_type, info): (
                    "explicit",
                    chunk,
                )
                for chunk in chunks
            }
            futures.update(
                {
                    executor.submit(self._process_single_chunk_implicit, chunk, msg_type, info): (
                        "implicit",
                        chunk,
                    )
                    for chunk in chunks
                }
            )

            for future in as_completed(futures):
                try:
                    memory = future.result()
                    if memory:
                        if isinstance(memory, list):
                            memories.extend(memory)
                        else:
                            memories.append(memory)
                except Exception as e:
                    task_type, chunk = futures[future]
                    logger.error(f"Error processing {task_type} chunk: {chunk}\n{e}")
                    continue

        return memories
