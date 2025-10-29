from enum import Enum
from pathlib import Path
from typing import NewType


class SearchMode(str, Enum):
    """Enumeration for search modes."""

    FAST = "fast"
    FINE = "fine"
    MIXTURE = "mixture"


FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent.parent.parent

QUERY_LABEL = "query"
ANSWER_LABEL = "answer"
ADD_LABEL = "add"
MEM_READ_LABEL = "mem_read"
MEM_ORGANIZE_LABEL = "mem_organize"
API_MIX_SEARCH_LABEL = "api_mix_search"
PREF_ADD_LABEL = "pref_add"

TreeTextMemory_SEARCH_METHOD = "tree_text_memory_search"
TreeTextMemory_FINE_SEARCH_METHOD = "tree_text_memory_fine_search"
TextMemory_SEARCH_METHOD = "text_memory_search"
DIRECT_EXCHANGE_TYPE = "direct"
FANOUT_EXCHANGE_TYPE = "fanout"
DEFAULT_WORKING_MEM_MONITOR_SIZE_LIMIT = 30
DEFAULT_ACTIVATION_MEM_MONITOR_SIZE_LIMIT = 20
DEFAULT_ACT_MEM_DUMP_PATH = f"{BASE_DIR}/outputs/mem_scheduler/mem_cube_scheduler_test.kv_cache"
DEFAULT_THREAD_POOL_MAX_WORKERS = 30
DEFAULT_CONSUME_INTERVAL_SECONDS = 0.05
DEFAULT_DISPATCHER_MONITOR_CHECK_INTERVAL = 300
DEFAULT_DISPATCHER_MONITOR_MAX_FAILURES = 2
DEFAULT_STUCK_THREAD_TOLERANCE = 10
DEFAULT_MAX_INTERNAL_MESSAGE_QUEUE_SIZE = 100000
DEFAULT_TOP_K = 10
DEFAULT_CONTEXT_WINDOW_SIZE = 5
DEFAULT_USE_REDIS_QUEUE = False
DEFAULT_MULTI_TASK_RUNNING_TIMEOUT = 30

# startup mode configuration
STARTUP_BY_THREAD = "thread"
STARTUP_BY_PROCESS = "process"
DEFAULT_STARTUP_MODE = STARTUP_BY_THREAD  # default to thread mode

NOT_INITIALIZED = -1


# web log
LONG_TERM_MEMORY_TYPE = "LongTermMemory"
USER_MEMORY_TYPE = "UserMemory"
WORKING_MEMORY_TYPE = "WorkingMemory"
TEXT_MEMORY_TYPE = "TextMemory"
ACTIVATION_MEMORY_TYPE = "ActivationMemory"
PARAMETER_MEMORY_TYPE = "ParameterMemory"
USER_INPUT_TYPE = "UserInput"
NOT_APPLICABLE_TYPE = "NotApplicable"

# monitors
MONITOR_WORKING_MEMORY_TYPE = "MonitorWorkingMemoryType"
MONITOR_ACTIVATION_MEMORY_TYPE = "MonitorActivationMemoryType"
DEFAULT_MAX_QUERY_KEY_WORDS = 1000
DEFAULT_WEIGHT_VECTOR_FOR_RANKING = [0.9, 0.05, 0.05]


# new types
UserID = NewType("UserID", str)
MemCubeID = NewType("CubeID", str)
