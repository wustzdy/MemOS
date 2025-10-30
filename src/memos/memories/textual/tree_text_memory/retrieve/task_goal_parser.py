import traceback

from string import Template

from memos.llms.base import BaseLLM
from memos.log import get_logger
from memos.memories.textual.tree_text_memory.retrieve.retrieval_mid_structs import ParsedTaskGoal
from memos.memories.textual.tree_text_memory.retrieve.retrieve_utils import FastTokenizer
from memos.memories.textual.tree_text_memory.retrieve.utils import TASK_PARSE_PROMPT


logger = get_logger(__name__)


class TaskGoalParser:
    """
    Unified TaskGoalParser:
    - mode == 'fast': directly use origin task_description
    - mode == 'fine': use LLM to parse structured topic/keys/tags
    """

    def __init__(self, llm=BaseLLM):
        self.llm = llm
        self.tokenizer = FastTokenizer()

    def parse(
        self,
        task_description: str,
        context: str = "",
        conversation: list[dict] | None = None,
        mode: str = "fast",
        **kwargs,
    ) -> ParsedTaskGoal:
        """
        Parse user input into structured semantic layers.
        Returns:
            ParsedTaskGoal: object containing topic/concept/fact levels and optional metadata
        - mode == 'fast': use jieba to split words only
        - mode == 'fine': use LLM to parse structured topic/keys/tags
        """
        if mode == "fast":
            return self._parse_fast(task_description, **kwargs)
        elif mode == "fine":
            if not self.llm:
                raise ValueError("LLM not provided for slow mode.")
            return self._parse_fine(task_description, context, conversation)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _parse_fast(self, task_description: str, **kwargs) -> ParsedTaskGoal:
        """
        Fast mode: simple jieba word split.
        """
        use_fast_graph = kwargs.get("use_fast_graph", False)
        if use_fast_graph:
            desc_tokenized = self.tokenizer.tokenize_mixed(task_description)
            return ParsedTaskGoal(
                memories=[task_description],
                keys=desc_tokenized,
                tags=desc_tokenized,
                goal_type="default",
                rephrased_query=task_description,
                internet_search=False,
            )
        else:
            return ParsedTaskGoal(
                memories=[task_description],
                keys=[task_description],
                tags=[],
                goal_type="default",
                rephrased_query=task_description,
                internet_search=False,
            )

    def _parse_fine(
        self, query: str, context: str = "", conversation: list[dict] | None = None
    ) -> ParsedTaskGoal:
        """
        Slow mode: LLM structured parse.
        """
        try:
            if conversation:
                conversation_prompt = "\n".join(
                    [f"{each['role']}: {each['content']}" for each in conversation]
                )
            else:
                conversation_prompt = ""
            prompt = Template(TASK_PARSE_PROMPT).substitute(
                task=query.strip(), context=context, conversation=conversation_prompt
            )
            logger.info(f"Parsing Goal... LLM input is {prompt}")
            response = self.llm.generate(messages=[{"role": "user", "content": prompt}])
            logger.info(f"Parsing Goal... LLM Response is {response}")
            return self._parse_response(response)
        except Exception:
            logger.warning(f"Fail to fine-parse query {query}: {traceback.format_exc()}")
            return self._parse_fast(query)

    def _parse_response(self, response: str) -> ParsedTaskGoal:
        """
        Parse LLM JSON output safely.
        """
        try:
            response = response.replace("```", "").replace("json", "").strip()
            response_json = eval(response)
            return ParsedTaskGoal(
                memories=response_json.get("memories", []),
                keys=response_json.get("keys", []),
                tags=response_json.get("tags", []),
                rephrased_query=response_json.get("rephrased_instruction", None),
                internet_search=response_json.get("internet_search", False),
                goal_type=response_json.get("goal_type", "default"),
            )
        except Exception as e:
            raise ValueError(f"Failed to parse LLM output: {e}\nRaw response:\n{response}") from e
