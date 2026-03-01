# ADR-002: Database Choice

## Status
Accepted

## Context

GAAP needed persistent storage for:
- Chat sessions and messages
- Memory system (episodic, semantic)
- Configuration and state
- Audit logs

Requirements:
- Easy development setup
- Production scalability
- SQL for structured queries
- JSON support for flexible schemas
- Async support for performance

## Decision

We use **SQLite** for development and **PostgreSQL** for production.

## SQLite (Development/Default)

### Why SQLite

```
┌─────────────────────────────────────────────────────────┐
│                    SQLite Benefits                       │
├─────────────────────────────────────────────────────────┤
│  ✓ Zero configuration                                    │
│  ✓ Single file, easy backup                              │
│  ✓ No separate process                                   │
│  ✓ Excellent for single-node deployments               │
│  ✓ Full SQL support                                      │
│  ✓ JSON1 extension                                       │
└─────────────────────────────────────────────────────────┘
```

### Configuration

```python
# Default path
~/.gaap/gaap.db

# Or via environment
GAAP_DB_PATH=/custom/path/gaap.db
```

### Limitations

| Limitation | Mitigation |
|------------|------------|
| Concurrent writes | WAL mode enabled |
| Network access | Single-node only |
| Scale | Migrate to PostgreSQL |

## PostgreSQL (Production)

### Why PostgreSQL

- **Scalability**: Connection pooling, read replicas
- **Reliability**: ACID transactions, point-in-time recovery
- **Features**: Full-text search, JSONB, vector extensions (pgvector)
- **Ecosystem**: Managed services (RDS, Cloud SQL)

### Configuration

```bash
# Environment variable
DATABASE_URL=postgresql://user:pass@host:5432/gaap
```

### Migration Path

SQLite → PostgreSQL is seamless:

```python
# Same SQLAlchemy models work with both
from gaap.db.models import Session, Message

# SQLite (dev)
sqlite_url = "sqlite+aiosqlite:///./gaap.db"

# PostgreSQL (prod)
postgres_url = "postgresql+asyncpg://user:pass@localhost/gaap"
```

## Schema Design

### Core Tables

```sql
-- Sessions
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'active',
    priority TEXT DEFAULT 'normal',
    tags JSON,
    config JSON,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_cost REAL DEFAULT 0.0
);

-- Messages
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    status TEXT DEFAULT 'completed',
    provider TEXT,
    model TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    latency_ms REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Key Design Decisions

1. **String IDs**: UUIDs for easy generation, no sequence contention
2. **JSON Columns**: Flexible schema for config/metadata
3. **Async SQLAlchemy**: Non-blocking database operations
4. **Soft Deletes**: Archive instead of delete for audit trail

## Alternatives Considered

### MongoDB
- **Pros:** Native JSON, flexible schema
- **Cons:** Less mature async support, transaction complexity
- **Verdict:** Overkill for relational data

### Redis (Primary)
- **Pros:** Extremely fast
- **Cons:** Limited query capabilities, persistence concerns
- **Verdict:** Good for cache, insufficient for primary store

### MySQL
- **Pros:** Widely deployed
- **Cons:** Less advanced JSON support than PostgreSQL
- **Verdict:** PostgreSQL has better JSONB and vector support

## Consequences

### Positive
- Single codebase supports both dev and prod
- Easy local development (no DB setup)
- Clear migration path as you scale
- Full SQL power when needed

### Negative
- SQLite limitations at scale
- Need to test on both databases
- Slight performance overhead of abstraction

## Implementation

```python
# gaap/db/__init__.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Database URL selection
if os.environ.get("DATABASE_URL"):
    DATABASE_URL = os.environ["DATABASE_URL"]
else:
    db_path = os.path.expanduser("~/.gaap/gaap.db")
    DATABASE_URL = f"sqlite+aiosqlite:///{db_path}"

# Create engine
engine = create_async_engine(DATABASE_URL)

# Session factory
async_session = sessionmaker(engine, class_=AsyncSession)

async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session
```

## References

- [SQLite Documentation](https://www.sqlite.org/docs.html)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [SQLAlchemy AsyncIO](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
