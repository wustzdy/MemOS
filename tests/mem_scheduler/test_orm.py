import os
import tempfile
import time

from datetime import datetime, timedelta

import pytest

from memos.mem_scheduler.orm_modules.base_model import BaseDBManager

# Import the classes to test
from memos.mem_scheduler.orm_modules.monitor_models import (
    DBManagerForMemoryMonitorManager,
    DBManagerForQueryMonitorQueue,
)
from memos.mem_scheduler.orm_modules.redis_model import RedisDBManager
from memos.mem_scheduler.schemas.monitor_schemas import (
    MemoryMonitorItem,
    MemoryMonitorManager,
    QueryMonitorItem,
    QueryMonitorQueue,
)


# Test data
TEST_USER_ID = "test_user"
TEST_MEM_CUBE_ID = "test_mem_cube"
TEST_QUEUE_ID = "test_queue"


class TestBaseDBManager:
    """Base class for DBManager tests with common fixtures"""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_scheduler_orm.db")
        yield db_path
        # Cleanup
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
            os.rmdir(temp_dir)
        except (OSError, PermissionError):
            pass  # Ignore cleanup errors (e.g., file locked on Windows)

    @pytest.fixture
    def memory_manager_obj(self):
        """Create a MemoryMonitorManager object for testing"""
        return MemoryMonitorManager(
            user_id=TEST_USER_ID,
            mem_cube_id=TEST_MEM_CUBE_ID,
            items=[
                MemoryMonitorItem(
                    item_id="custom-id-123",
                    memory_text="Full test memory",
                    tree_memory_item=None,
                    tree_memory_item_mapping_key="full_test_key",
                    keywords_score=0.8,
                    sorting_score=0.9,
                    importance_score=0.7,
                    recording_count=3,
                )
            ],
        )

    @pytest.fixture
    def query_queue_obj(self):
        """Create a QueryMonitorQueue object for testing"""
        queue = QueryMonitorQueue()
        queue.put(
            QueryMonitorItem(
                item_id="query1",
                user_id=TEST_USER_ID,
                mem_cube_id=TEST_MEM_CUBE_ID,
                query_text="How are you?",
                timestamp=datetime.now(),
                keywords=["how", "you"],
            )
        )
        return queue

    @pytest.fixture
    def query_monitor_manager(self, temp_db, query_queue_obj):
        """Create DBManagerForQueryMonitorQueue instance with temp DB."""
        engine = BaseDBManager.create_engine_from_db_path(temp_db)
        manager = DBManagerForQueryMonitorQueue(
            engine=engine,
            user_id=TEST_USER_ID,
            mem_cube_id=TEST_MEM_CUBE_ID,
            obj=query_queue_obj,
            lock_timeout=10,
        )

        assert manager.engine is not None
        assert manager.SessionLocal is not None
        assert os.path.exists(temp_db)

        yield manager
        manager.close()

    @pytest.fixture
    def memory_monitor_manager(self, temp_db, memory_manager_obj):
        """Create DBManagerForMemoryMonitorManager instance with temp DB."""
        engine = BaseDBManager.create_engine_from_db_path(temp_db)
        manager = DBManagerForMemoryMonitorManager(
            engine=engine,
            user_id=TEST_USER_ID,
            mem_cube_id=TEST_MEM_CUBE_ID,
            obj=memory_manager_obj,
            lock_timeout=10,
        )

        assert manager.engine is not None
        assert manager.SessionLocal is not None
        assert os.path.exists(temp_db)

        yield manager
        manager.close()

    def test_save_and_load_query_queue(self, query_monitor_manager, query_queue_obj):
        """Test saving and loading QueryMonitorQueue."""
        # Save to database
        query_monitor_manager.save_to_db(query_queue_obj)

        # Load in a new manager
        engine = BaseDBManager.create_engine_from_db_path(query_monitor_manager.engine.url.database)
        new_manager = DBManagerForQueryMonitorQueue(
            engine=engine,
            user_id=TEST_USER_ID,
            mem_cube_id=TEST_MEM_CUBE_ID,
            obj=None,
            lock_timeout=10,
        )
        loaded_queue = new_manager.load_from_db(acquire_lock=True)

        assert loaded_queue is not None
        items = loaded_queue.get_queue_content_without_pop()
        assert len(items) == 1
        assert items[0].item_id == "query1"
        assert items[0].query_text == "How are you?"
        new_manager.close()

    def test_lock_mechanism(self, query_monitor_manager, query_queue_obj):
        """Test lock acquisition and release."""
        # Save current state
        query_monitor_manager.save_to_db(query_queue_obj)

        # Acquire lock
        acquired = query_monitor_manager.acquire_lock(block=True)
        assert acquired

        # Try to acquire again (should fail without blocking)
        assert not query_monitor_manager.acquire_lock(block=False)

        # Release lock
        query_monitor_manager.release_locks(
            user_id=TEST_USER_ID,
            mem_cube_id=TEST_MEM_CUBE_ID,
        )

        # Should be able to acquire again
        assert query_monitor_manager.acquire_lock(block=False)

    def test_lock_timeout(self, query_monitor_manager, query_queue_obj):
        """Test lock timeout mechanism."""
        # Save current state
        query_monitor_manager.save_to_db(query_queue_obj)

        query_monitor_manager.lock_timeout = 1

        # Acquire lock
        assert query_monitor_manager.acquire_lock(block=True)

        # Wait for lock to expire
        time.sleep(1.1)

        # Should be able to acquire again
        assert query_monitor_manager.acquire_lock(block=False)

    def test_sync_with_orm(self, query_monitor_manager, query_queue_obj):
        """Test synchronization between ORM and object."""
        query_queue_obj.put(
            QueryMonitorItem(
                item_id="query2",
                user_id=TEST_USER_ID,
                mem_cube_id=TEST_MEM_CUBE_ID,
                query_text="What's your name?",
                timestamp=datetime.now(),
                keywords=["name"],
            )
        )

        # Save current state
        query_monitor_manager.save_to_db(query_queue_obj)

        # Create sync manager with empty queue
        empty_queue = QueryMonitorQueue(maxsize=10)
        engine = BaseDBManager.create_engine_from_db_path(query_monitor_manager.engine.url.database)
        sync_manager = DBManagerForQueryMonitorQueue(
            engine=engine,
            user_id=TEST_USER_ID,
            mem_cube_id=TEST_MEM_CUBE_ID,
            obj=empty_queue,
            lock_timeout=10,
        )

        # First sync - should create a new record with empty queue
        sync_manager.sync_with_orm(size_limit=None)
        items = sync_manager.obj.get_queue_content_without_pop()
        assert len(items) == 0  # Empty queue since no existing data to merge

        # Now save the empty queue to create a record
        sync_manager.save_to_db(empty_queue)

        # Test that sync_with_orm correctly handles version control
        # The sync should increment version but not merge data when versions are the same
        sync_manager.sync_with_orm(size_limit=None)
        items = sync_manager.obj.get_queue_content_without_pop()
        assert len(items) == 0  # Should remain empty since no merge occurred

        # Verify that the version was incremented
        assert sync_manager.last_version_control == "3"  # Should increment from 2 to 3

        sync_manager.close()

    def test_sync_with_size_limit(self, query_monitor_manager, query_queue_obj):
        """Test synchronization with size limit."""
        now = datetime.now()
        item_size = 1
        for i in range(2, 6):
            item_size += 1
            query_queue_obj.put(
                QueryMonitorItem(
                    item_id=f"query{i}",
                    user_id=TEST_USER_ID,
                    mem_cube_id=TEST_MEM_CUBE_ID,
                    query_text=f"Question {i}",
                    timestamp=now + timedelta(minutes=i),
                    keywords=[f"kw{i}"],
                )
            )

        # First sync - should create a new record (size_limit not applied for new records)
        size_limit = 3
        query_monitor_manager.sync_with_orm(size_limit=size_limit)
        items = query_monitor_manager.obj.get_queue_content_without_pop()
        assert len(items) == item_size  # All items since size_limit not applied for new records

        # Save to create the record
        query_monitor_manager.save_to_db(query_monitor_manager.obj)

        # Test that sync_with_orm correctly handles version control
        # The sync should increment version but not merge data when versions are the same
        query_monitor_manager.sync_with_orm(size_limit=size_limit)
        items = query_monitor_manager.obj.get_queue_content_without_pop()
        assert len(items) == item_size  # Should remain the same since no merge occurred

        # Verify that the version was incremented
        assert query_monitor_manager.last_version_control == "2"

    def test_concurrent_access(self, temp_db, query_queue_obj):
        """Test concurrent access to the same database."""

        # Manager 1
        engine1 = BaseDBManager.create_engine_from_db_path(temp_db)
        manager1 = DBManagerForQueryMonitorQueue(
            engine=engine1,
            user_id=TEST_USER_ID,
            mem_cube_id=TEST_MEM_CUBE_ID,
            obj=query_queue_obj,
            lock_timeout=10,
        )
        manager1.save_to_db(query_queue_obj)

        # Manager 2
        engine2 = BaseDBManager.create_engine_from_db_path(temp_db)
        manager2 = DBManagerForQueryMonitorQueue(
            engine=engine2,
            user_id=TEST_USER_ID,
            mem_cube_id=TEST_MEM_CUBE_ID,
            obj=query_queue_obj,
            lock_timeout=10,
        )

        # Manager1 acquires lock
        assert manager1.acquire_lock(block=True)

        # Manager2 fails to acquire
        assert not manager2.acquire_lock(block=False)

        # Manager1 releases
        manager1.release_locks(user_id=TEST_USER_ID, mem_cube_id=TEST_MEM_CUBE_ID)

        # Manager2 can now acquire
        assert manager2.acquire_lock(block=False)

        manager1.close()
        manager2.close()


class TestRedisDBManager:
    """Test class for RedisDBManager functionality"""

    @pytest.fixture
    def memory_manager_obj(self):
        """Create a MemoryMonitorManager object for testing"""
        return MemoryMonitorManager(
            user_id=TEST_USER_ID,
            mem_cube_id=TEST_MEM_CUBE_ID,
            memories=[
                MemoryMonitorItem(
                    item_id="redis-test-123",
                    memory_text="Redis test memory",
                    tree_memory_item=None,
                    tree_memory_item_mapping_key="redis_test_key",
                    keywords_score=0.8,
                    sorting_score=0.9,
                    importance_score=0.7,
                    recording_count=3,
                )
            ],
        )

    @pytest.fixture
    def mock_redis_client(self):
        """Create a mock Redis client for testing"""
        try:
            from unittest.mock import MagicMock

            # Create a mock Redis client
            mock_client = MagicMock()

            # Mock Redis data storage
            mock_data = {}

            def mock_set(key, value, nx=False, ex=None, **kwargs):
                if nx and key in mock_data:
                    # NX means "only set if not exists"
                    return False  # Redis returns False when NX fails
                mock_data[key] = value
                return True

            def mock_get(key):
                return mock_data.get(key)

            def mock_hset(key, mapping=None, **kwargs):
                if key not in mock_data:
                    mock_data[key] = {}
                if mapping:
                    mock_data[key].update(mapping)
                if kwargs:
                    mock_data[key].update(kwargs)
                return len(mapping) if mapping else len(kwargs)

            def mock_hgetall(key):
                return mock_data.get(key, {})

            def mock_delete(*keys):
                deleted = 0
                for key in keys:
                    if key in mock_data:
                        del mock_data[key]
                        deleted += 1
                return deleted

            def mock_keys(pattern):
                import fnmatch

                return [key for key in mock_data if fnmatch.fnmatch(key, pattern)]

            def mock_ping():
                return True

            def mock_close():
                pass

            # Configure mock methods
            mock_client.set = mock_set
            mock_client.get = mock_get
            mock_client.hset = mock_hset
            mock_client.hgetall = mock_hgetall
            mock_client.delete = mock_delete
            mock_client.keys = mock_keys
            mock_client.ping = mock_ping
            mock_client.close = mock_close

            return mock_client

        except ImportError:
            pytest.skip("Redis package not available for testing")

    @pytest.fixture
    def redis_manager(self, mock_redis_client, memory_manager_obj):
        """Create RedisDBManager instance with mock Redis client"""
        manager = RedisDBManager(
            user_id=TEST_USER_ID,
            mem_cube_id=TEST_MEM_CUBE_ID,
            obj=memory_manager_obj,
            lock_timeout=10,
            redis_client=mock_redis_client,
        )
        yield manager
        manager.close()

    def test_redis_manager_initialization(self, mock_redis_client):
        """Test RedisDBManager initialization"""
        manager = RedisDBManager(
            user_id=TEST_USER_ID, mem_cube_id=TEST_MEM_CUBE_ID, redis_client=mock_redis_client
        )

        assert manager.user_id == TEST_USER_ID
        assert manager.mem_cube_id == TEST_MEM_CUBE_ID
        assert manager.redis_client is mock_redis_client
        assert manager.orm_class.__name__ == "RedisLockableORM"
        assert manager.obj_class == MemoryMonitorManager

        manager.close()

    def test_redis_lockable_orm_save_load(self, mock_redis_client):
        """Test RedisLockableORM save and load operations"""
        from memos.mem_scheduler.orm_modules.redis_model import RedisLockableORM

        orm = RedisLockableORM(
            redis_client=mock_redis_client, user_id=TEST_USER_ID, mem_cube_id=TEST_MEM_CUBE_ID
        )

        # Test save
        orm.serialized_data = '{"test": "data"}'
        orm.version_control = "1"
        orm.lock_acquired = True
        orm.lock_expiry = datetime.now()

        orm.save()

        # Test load
        new_orm = RedisLockableORM(
            redis_client=mock_redis_client, user_id=TEST_USER_ID, mem_cube_id=TEST_MEM_CUBE_ID
        )

        exists = new_orm.load()
        assert exists
        assert new_orm.serialized_data == '{"test": "data"}'
        assert new_orm.version_control == "1"
        # Note: lock_acquired is False after load by design - locks are managed separately
        assert not new_orm.lock_acquired
