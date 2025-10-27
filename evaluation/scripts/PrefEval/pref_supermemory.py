import argparse
import concurrent.futures
import json
import os
import sys
import time

import tiktoken

from dotenv import load_dotenv
from irrelevant_conv import irre_10, irre_300
from openai import OpenAI
from tqdm import tqdm


ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
EVAL_SCRIPTS_DIR = os.path.join(ROOT_DIR, "evaluation", "scripts")

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, EVAL_SCRIPTS_DIR)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
tokenizer = tiktoken.get_encoding("cl100k_base")


def add_memory_for_line(
    line_data: tuple, mem_client, num_irrelevant_turns: int, lib: str, version: str
) -> dict:
    """
    Adds conversation memory for a single line of data to MemOS and returns the data with a persistent user_id.
    """
    i, line = line_data
    user_id = f"{lib}_user_pref_eval_{i}_{version}"

    try:
        original_data = json.loads(line)
        conversation = original_data.get("conversation", [])

        if num_irrelevant_turns == 10:
            conversation = conversation + irre_10
        elif num_irrelevant_turns == 300:
            conversation = conversation + irre_300

        turns_add = 5
        start_time_add = time.monotonic()
        if conversation:
            if os.getenv("PRE_SPLIT_CHUNK", "false").lower() == "true":
                for chunk_start in range(0, len(conversation), turns_add * 2):
                    chunk = conversation[chunk_start : chunk_start + turns_add * 2]
                    mem_client.add(messages=chunk, user_id=user_id)
            else:
                mem_client.add(messages=conversation, user_id=user_id)
        end_time_add = time.monotonic()
        add_duration = end_time_add - start_time_add

        original_data["user_id"] = user_id
        original_data["metrics"] = {"add_memories_duration_seconds": add_duration}
        return original_data

    except Exception as e:
        print(f"Error adding memory for line {i + 1} (user_id: {user_id}): {e}")
        return None


def search_memory_for_line(line_data: tuple, mem_client, top_k_value: int) -> dict:
    """
    Processes a single line of data, searching memory based on the question.
    """
    i, line = line_data
    try:
        original_data = json.loads(line)

        user_id = original_data.get("user_id")
        question = original_data.get("question")
        metrics_dict = original_data.get("metrics", {})

        if not user_id:
            original_data["error"] = (
                "Error: user_id not found in this line. Please run 'add' mode first."
            )
            return original_data
        if not question:
            original_data["error"] = "Question not found in this line."
            return original_data

        start_time_search = time.monotonic()
        relevant_memories = mem_client.search(query=question, user_id=user_id, top_k=top_k_value)
        search_memories_duration = time.monotonic() - start_time_search
        memories_str = "\n".join(
            f"- {entry.get('memory', '')}" for entry in relevant_memories["text_mem"][0]["memories"]
        )

        memory_tokens_used = len(tokenizer.encode(memories_str))

        metrics_dict.update(
            {
                "search_memories_duration_seconds": search_memories_duration,
                "memory_tokens_used": memory_tokens_used,
                "retrieved_memories_text": memories_str,
            }
        )
        original_data["metrics"] = metrics_dict

        return original_data

    except Exception as e:
        user_id_from_data = json.loads(line).get("user_id", "N/A")
        print(f"Error searching memory for line {i + 1} (user_id: {user_id_from_data}): {e}")
        return None


def generate_response_for_line(line_data: tuple, openai_client: OpenAI) -> dict:
    """
    Generates a response for a single line of data using pre-fetched memories.
    """
    i, line = line_data
    try:
        original_data = json.loads(line)

        question = original_data.get("question")
        metrics_dict = original_data.get("metrics", {})
        memories_str = metrics_dict.get("retrieved_memories_text")

        # If an error occurred in 'add' or 'search' mode, just pass the line through
        if original_data.get("error"):
            return original_data

        if not question:
            original_data["error"] = "Question not found in this line."
            return original_data

        # Check for None, as an empty string (no memories found) is a valid result
        if memories_str is None:
            original_data["error"] = (
                "Error: retrieved_memories_text not found in metrics. "
                "Please run 'search' mode first."
            )
            return original_data

        system_prompt = f"You are a helpful AI. Answer the question based on the query and the following memories:\nUser Memories:\n{memories_str}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        response = openai_client.chat.completions.create(model=MODEL_NAME, messages=messages)
        assistant_response = response.choices[0].message.content
        original_data["response"] = assistant_response

        return original_data

    except Exception as e:
        user_id_from_data = json.loads(line).get("user_id", "N/A")
        print(f"Error generating response for line {i + 1} (user_id: {user_id_from_data}): {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Process conversations with MemOS. Run 'add', then 'search', then 'response'."
    )
    parser.add_argument(
        "mode",
        choices=["add", "search", "response"],
        help="The mode to run the script in ('add', 'search', or 'response').",
    )
    parser.add_argument("--input", required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output", required=True, help="Path to the output JSONL file.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of memories to retrieve (used in 'search' mode).",
    )
    parser.add_argument(
        "--add-turn",
        type=int,
        choices=[0, 10, 300],
        default=0,
        help="Number of irrelevant turns to add (used in 'add' mode).",
    )
    parser.add_argument(
        "--lib",
        type=str,
        choices=["supermemory"],
        default="supermemory",
        help="Which Supermemory library to use (used in 'add' mode).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="0929-1",
        help="Version identifier for user_id generation (used in 'add' mode).",
    )
    parser.add_argument(
        "--max-workers", type=int, default=20, help="Maximum number of concurrent workers."
    )

    args = parser.parse_args()

    try:
        with open(args.input, encoding="utf-8") as infile:
            lines = infile.readlines()
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found")
        return

    class SupermemoryClient:
        def __init__(self):
            from supermemory import Supermemory

            self.client = Supermemory(api_key=os.getenv("SUPERMEMORY_API_KEY"))

        def add(self, messages, user_id):
            content = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    self.client.memories.add(content=content, container_tag=user_id)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                    else:
                        raise e

        def search(self, query, user_id, top_k):
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    results = self.client.search.memories(
                        q=query,
                        container_tag=user_id,
                        threshold=0,
                        rerank=True,
                        rewrite_query=True,
                        limit=top_k,
                    )
                    context = "\n\n".join([r.memory for r in results.results])
                    return context
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                    else:
                        raise e

    mem_client = SupermemoryClient()

    if args.mode == "add":
        print(f"Running in 'add' mode. Ingesting memories from '{args.input}'...")
        print(f"Adding {args.add_turn} irrelevant turns.")
        print(f"Using {args.max_workers} workers.")
        with (
            open(args.output, "w", encoding="utf-8") as outfile,
            concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor,
        ):
            futures = [
                executor.submit(
                    add_memory_for_line,
                    (i, line),
                    mem_client,
                    args.add_turn,
                    args.lib,
                    args.version,
                )
                for i, line in enumerate(lines)
            ]

            pbar = tqdm(
                concurrent.futures.as_completed(futures),
                total=len(lines),
                desc="Adding memories...",
            )
            for future in pbar:
                result = future.result()
                if result:
                    outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"\n'add' mode complete! Data with user_id written to '{args.output}'.")

    elif args.mode == "search":
        print(f"Running in 'search' mode. Searching memories based on '{args.input}'...")
        print(f"Retrieving top {args.top_k} memories for each query.")
        print(f"Using {args.max_workers} workers.")
        with (
            open(args.output, "w", encoding="utf-8") as outfile,
            concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor,
        ):
            futures = [
                executor.submit(search_memory_for_line, (i, line), mem_client, args.top_k)
                for i, line in enumerate(lines)
            ]

            pbar = tqdm(
                concurrent.futures.as_completed(futures),
                total=len(lines),
                desc="Searching memories...",
            )
            for future in pbar:
                result = future.result()
                if result:
                    outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(
            f"\n'search' mode complete! Results with retrieved memories written to '{args.output}'."
        )

    elif args.mode == "response":
        print(f"Running in 'response' mode. Generating responses based on '{args.input}'...")
        print(f"Using {args.max_workers} workers.")
        openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
        with (
            open(args.output, "w", encoding="utf-8") as outfile,
            concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor,
        ):
            futures = [
                executor.submit(generate_response_for_line, (i, line), openai_client)
                for i, line in enumerate(lines)
            ]

            pbar = tqdm(
                concurrent.futures.as_completed(futures),
                total=len(lines),
                desc="Generating responses...",
            )
            for future in pbar:
                result = future.result()
                if result:
                    outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"\n'response' mode complete! Final results written to '{args.output}'.")


if __name__ == "__main__":
    main()
