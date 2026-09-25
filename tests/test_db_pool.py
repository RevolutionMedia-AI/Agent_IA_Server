"""Pool accounting contract for STT_server.db.get_conn().

Production incident: ~30s after startup on Railway, GET
/twilio-credentials failed in require_auth -> load_sessions_db ->
get_conn -> pool.getconn() with psycopg2 PoolError (pool exhausted),
and GET /tools failed at the same time in db_list_tools. Both are
single-slot sequential users, so the exhaustion came from elsewhere
holding all 10 slots.

These tests pin the acquisition/release contract of the REAL
get_conn() implementation by injecting a FakePool in place of
STT_server.db._pool (no Postgres required):

  P1 success returns the slot
  P2 exception returns the slot
  P3 early return returns the slot
  P4 parallel operations do not leak
  P5 auth lookup + endpoint access (sequential) returns both slots
  P6 repeated failure paths do not shrink the pool
  P7 BaseException (CancelledError) returns the slot AND rolls back
     (before the fix, the slot was returned but the transaction was
     left open - the next borrower inherited idle-in-transaction)
  P8 nested acquisition reaches 2 slots at once; P8b proves the inner
     slot is RELEASED before the HTTP call (the 2-slot window is the
     SELECT only, never the 15s HTTP window)
  P9 PoolError logs safe counters and re-raises
  P10 cur= reuses the caller's cursor without touching the pool
  P11 the no-cur default path still opens its own connection
  P12 PRE-FIX production shape: real get_integration_by_id() with no
     cur=, 5 concurrent refreshes gated inside the real query window.
     10 slots at the query gate, victim gets PoolExhausted, and only
     5 slots during the HTTP window (inner already released)
  P13 POST-FIX same path with cur=outer_cursor: 5 slots at the query
     gate, victim succeeds, peak 6, pool returns to baseline. Red
     pre-fix (no cur= parameter existed)
  P13b static ast guard: no get_integration_by_id() call inside a
     `with get_conn()` block may omit cur=
  P14 a failing rollback() still returns the slot via finally
"""
from __future__ import annotations

import asyncio
import threading

import pytest

import STT_server.db as db_mod


class PoolExhausted(Exception):
    pass


class FakeConn:
    def __init__(self, on_execute=None):
        self.committed = 0
        self.rolled_back = 0
        self.closed = False
        self._cursor = FakeCursor(on_execute)

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed += 1

    def rollback(self):
        self.rolled_back += 1


class FakeCursor:
    def __init__(self, on_execute=None):
        self._on_execute = on_execute

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        if self._on_execute is not None:
            self._on_execute(sql, params)
        return None

    def fetchone(self):
        return None

    def fetchall(self):
        return []


class FakePool:
    """Mimics psycopg2 ThreadedConnectionPool(minconn, maxconn)."""

    def __init__(self, maxconn=10, on_execute=None, on_getconn=None,
                 on_putconn=None):
        self.maxconn = maxconn
        self._lock = threading.Lock()
        self._on_execute = on_execute
        self._on_getconn = on_getconn
        self._on_putconn = on_putconn
        self._free = [
            FakeConn(on_execute=on_execute) for _ in range(maxconn)
        ]
        self._checked_out = set()
        self.peak_in_use = 0
        self.getconn_calls = 0
        self.putconn_calls = 0

    def available(self):
        with self._lock:
            return len(self._free)

    def in_use(self):
        with self._lock:
            return len(self._checked_out)

    def getconn(self):
        with self._lock:
            self.getconn_calls += 1
            if not self._free:
                raise PoolExhausted("connection pool exhausted")
            conn = self._free.pop()
            self._checked_out.add(id(conn))
            self.peak_in_use = max(self.peak_in_use, len(self._checked_out))
            self._conns = getattr(self, "_conns", {})
            self._conns[id(conn)] = conn
        if self._on_getconn is not None:
            self._on_getconn(conn)
        return conn

    def putconn(self, conn):
        with self._lock:
            self.putconn_calls += 1
            self._checked_out.discard(id(conn))
            self._free.append(conn)
        if self._on_putconn is not None:
            self._on_putconn(conn)

    def closeall(self):
        pass


def _install_pool(monkeypatch, fake):
    """Install `fake` as STT_server.db._pool plus a stub psycopg2.

    _init_pool() short-circuits when _pool is not None, so the real
    get_conn() code runs unmodified against the fake. The stub exists
    because get_conn() does a lazy `import psycopg2` that only exists
    in production images.
    """
    import sys
    import types

    monkeypatch.setattr(db_mod, "_pool", fake)
    stub = types.ModuleType("psycopg2")
    pool_ns = types.SimpleNamespace(PoolError=PoolExhausted)
    stub.pool = pool_ns
    monkeypatch.setitem(sys.modules, "psycopg2", stub)
    return fake


@pytest.fixture
def pool(monkeypatch):
    return _install_pool(monkeypatch, FakePool(maxconn=10))


def test_p1_success_returns_slot(pool):
    with db_mod.get_conn() as conn:
        assert conn is not None
    assert pool.available() == 10
    assert pool.in_use() == 0


def test_p2_exception_returns_slot(pool):
    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        with db_mod.get_conn():
            raise Boom()
    assert pool.available() == 10
    assert pool.in_use() == 0


def test_p3_early_return_returns_slot(pool):
    def do_work():
        with db_mod.get_conn():
            return "early"

    assert do_work() == "early"
    assert pool.available() == 10
    assert pool.in_use() == 0


def test_p4_parallel_operations_do_not_leak(pool):
    errors = []

    def worker(n):
        try:
            for _ in range(20):
                with db_mod.get_conn():
                    pass
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert pool.available() == 10
    assert pool.in_use() == 0


def test_p5_auth_then_endpoint_sequential_returns_both(pool):
    """Mirrors a normal request: require_auth (load_sessions, 1 slot,
    released) then the endpoint helper (1 slot, released). Peak
    concurrent usage must be 1, never 2."""
    with db_mod.get_conn():
        pass  # auth: load_sessions_db
    with db_mod.get_conn():
        pass  # endpoint: e.g. db_list_tools
    assert pool.available() == 10
    assert pool.peak_in_use == 1


def test_p6_repeated_failures_do_not_shrink_pool(pool):
    for _ in range(100):
        try:
            with db_mod.get_conn():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
    assert pool.available() == 10
    assert pool.in_use() == 0
    # The pool must still serve new work after 100 failures.
    with db_mod.get_conn():
        pass
    assert pool.available() == 10


def test_p7_cancelled_returns_slot_and_rolls_back(pool):
    """CancelledError is BaseException, not Exception. Before the fix,
    get_conn() skipped rollback on this path (only `except Exception`
    rolled back) and returned the connection with an open transaction
    â€” the next borrower inherited idle-in-transaction."""
    conns = []

    async def use_conn():
        with db_mod.get_conn() as conn:
            conns.append(conn)
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(use_conn())
    assert pool.available() == 10
    assert pool.in_use() == 0
    assert conns and conns[0].rolled_back == 1


def test_p9_exhaustion_warns_and_raises(pool, caplog):
    """When all slots are checked out, get_conn() must fail fast with
    PoolError (never block/queue) and log a safe warning (no DSN).
    caplog pins the log record so a future refactor can't silently
    drop the only production signal for this incident class."""
    import logging

    held = []
    for _ in range(10):
        held.append(pool.getconn())
    with caplog.at_level(logging.WARNING, logger="stt_server.db"):
        with pytest.raises(PoolExhausted):
            with db_mod.get_conn():
                pass
    assert any(
        "DB_POOL" in r.message and "exhausted" in r.message
        for r in caplog.records
    )
    assert not any("postgres" in r.message.lower() for r in caplog.records)
    for c in held:
        pool.putconn(c)
    assert pool.available() == 10


class FakeSelectCursor:
    """Minimal cursor stub for get_integration_by_id(cur=...)."""

    def __init__(self, row):
        self._row = row
        self.queries = []

    def execute(self, query, params=None):
        self.queries.append((query, params))

    def fetchone(self):
        return self._row

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_p10_get_integration_by_id_cur_reuses_cursor_without_pool(monkeypatch):
    """Regression for the nested-acquisition fix: passing cur= must
    run the lookup on the caller's cursor without touching the pool.
    get_conn is stubbed to raise so any pool hop fails loudly."""
    import STT_server.db_integrations as dbi

    monkeypatch.setattr(dbi, "is_postgres", lambda: True)
    monkeypatch.setattr(dbi, "_columns_check_done", True)

    def _boom(*a, **k):
        raise AssertionError("pool touched")

    monkeypatch.setattr(dbi, "get_conn", _boom)
    cur = FakeSelectCursor({"id": "int-1"})
    out = dbi.get_integration_by_id("int-1", cur=cur)
    assert out is not None and out["id"] == "int-1"
    assert len(cur.queries) == 1
    assert "WHERE id = %s" in cur.queries[0][0]
    assert cur.queries[0][1] == ("int-1",)


def test_p11_get_integration_by_id_default_still_opens_own_connection(monkeypatch):
    """Default path (no cur) is unchanged: opens its own connection
    and releases it."""
    import contextlib

    import STT_server.db_integrations as dbi

    monkeypatch.setattr(dbi, "is_postgres", lambda: True)
    monkeypatch.setattr(dbi, "_columns_check_done", True)
    cur = FakeSelectCursor({"id": "int-9"})
    opened = []

    @contextlib.contextmanager
    def fake_get_conn():
        opened.append(True)
        yield FakeConnWithCursor(cur)

    monkeypatch.setattr(dbi, "get_conn", fake_get_conn)
    out = dbi.get_integration_by_id("int-9")
    assert out is not None and out["id"] == "int-9"
    assert opened == [True]
    assert len(cur.queries) == 1


class FakeConnWithCursor(FakeConn):
    def __init__(self, cur):
        super().__init__()
        self._cur = cur

    def cursor(self):
        return self._cur


def test_p8_nested_acquisition_holds_two_slots(pool):
    """Documents the amplification found in the OAuth refresh paths
    (routes/api.py internal credentials + execute endpoints,
    services/dynamics365.py): `with get_conn()` wrapping a helper
    that opens its own `with get_conn()`. Both slots are held only
    for the duration of the inner SELECT â€” the helper returns and
    its `finally` runs putconn BEFORE refresh_access_token() runs.
    Peak is 2, but the 2-slot window is milliseconds, not the 15s
    HTTP timeout. See test_p8b for that lifetime proof."""
    with db_mod.get_conn():
        assert pool.in_use() == 1
        with db_mod.get_conn():
            assert pool.in_use() == 2
        assert pool.in_use() == 1
    assert pool.available() == 10
    assert pool.peak_in_use == 2


class RecordingCursor(FakeCursor):
    """Cursor that records the slot state observed during execute().

    This is what proves the inner connection's lifetime: the refresh
    HTTP call happens AFTER the inner `with get_conn()` has already
    exited, so in_use() is back down to 1 by then."""

    def __init__(self, pool, slot_at_open):
        super().__init__()
        self._pool = pool
        self._slot_at_open = slot_at_open
        self.in_use_during_execute = None

    def execute(self, *a, **k):
        self.in_use_during_execute = self._pool.in_use()
        return None


def test_p8b_inner_slot_released_before_http_refresh(pool):
    """Lifetime proof for the nested refresh path.

    Order asserted: outer acquire -> inner acquire -> SELECT (2 slots)
    -> inner return/putconn (back to 1) -> 'HTTP' (1 slot) ->
    outer putconn (0). So the report must NOT claim 2 connections are
    held for the 15s HTTP window.
    """
    timeline = []

    def do_refresh_like_the_real_path():
        with db_mod.get_conn() as outer:               # outer acquire
            timeline.append(("outer_acquired", pool.in_use()))
            with db_mod.get_conn() as inner:           # inner acquire
                timeline.append(("inner_acquired", pool.in_use()))
                with inner.cursor() as cur:
                    cur.execute("SELECT ... WHERE id = %s", ("x",))
                    timeline.append(("inner_select", pool.in_use()))
            # inner with-block exited here: putconn already ran
            timeline.append(("inner_released", pool.in_use()))
            timeline.append(("http_refresh_start", pool.in_use()))
        timeline.append(("outer_released", pool.in_use()))

    do_refresh_like_the_real_path()
    assert timeline == [
        ("outer_acquired", 1),
        ("inner_acquired", 2),
        ("inner_select", 2),
        ("inner_released", 1),
        ("http_refresh_start", 1),
        ("outer_released", 0),
    ]
    assert pool.peak_in_use == 2
    assert pool.in_use() == 0


# â”€â”€ Production-shape causal repro: 5 concurrent refreshes + 1 unrelated â”€â”€â”€â”€
# The production refresh path (services/dynamics365.py:211-219,
# routes/api.py:4785-4792, routes/api.py:5072-5075) is:
#   with get_conn() as conn:                     <- outer slot
#     with conn.cursor() as cur:
#       acquire_advisory_xact_lock(cur, id)
#       fresh = get_integration_by_id(id)         <- PRE-FIX: opens its OWN
#                                                   slot (db_integrations.py:455)
#       refresh_access_token(...)                 <- HTTP, outer slot only
#
# get_integration_by_id() releases its own slot in the `finally` of its
# `with get_conn()` BEFORE returning, so the 2-slot window is the SELECT
# only; the HTTP window holds 1 slot. These tests drive the REAL
# get_integration_by_id() (no stubbed reader) and gate the REAL query
# window: the pool's cursor hook fires while the query is executing,
# which is strictly inside the nested reader's connection scope.

N_CONCURRENT_REFRESHES = 5
POOL_SIZE = 10
INTEGRATIONS_QUERY = "FROM integrations"


class Gate:
    """A synchronization point that parks every thread that reaches it.

    arrive() blocks until all N participants have reached it, so the
    test can observe the pool at a deterministic instant. Two gates are
    used: one at the nested SELECT (pre-fix: inner slot still held) and
    one at the HTTP window (inner slot already released).
    """

    def __init__(self, n):
        self._n = n
        self._arrived = 0
        self._lock = threading.Lock()
        self.all_parked = threading.Event()
        self.release = threading.Event()

    def arrive(self):
        with self._lock:
            self._arrived += 1
            if self._arrived == self._n:
                self.all_parked.set()
        assert self.all_parked.wait(timeout=15), "not all threads reached the gate"
        self.release.wait(timeout=15)


class Timeline:
    """Ordered log of (event, thread, in_use) for the causal proof."""

    def __init__(self, pool):
        self._pool = pool
        self._lock = threading.Lock()
        self.events = []

    def add(self, event, thread=None):
        with self._lock:
            self.events.append(
                (event, thread or threading.current_thread().name,
                 self._pool.in_use())
            )

    def indices(self, event):
        return [i for i, e in enumerate(self.events) if e[0] == event]


def _refresh_path(integration_id, query_gate, http_gate, timeline, dbi,
                  pass_cur):
    """The real production refresh path, with the OAuth HTTP client
    replaced by a gate.

    Outer get_conn -> cursor -> advisory xact lock -> integration re-read
    via the REAL get_integration_by_id -> 'HTTP' window.

    pass_cur=False reproduces the pre-fix call (no cur=, own connection).
    pass_cur=True reproduces the post-fix call (cur=outer cursor).
    """
    with db_mod.get_conn() as conn:
        timeline.add("outer_acquired")
        with conn.cursor() as cur:
            dbi.acquire_advisory_xact_lock(cur, integration_id)
            timeline.add("advisory_locked")
            if pass_cur:
                dbi.get_integration_by_id(integration_id, cur=cur)
            else:
                dbi.get_integration_by_id(integration_id)
            # the nested reader has returned: pre-fix it already released
            # its own slot, post-fix it never took one.
            timeline.add("reader_returned")
            query_gate.arrive()
            timeline.add("http_entered")
            http_gate.arrive()
            timeline.add("http_done")
        timeline.add("outer_released")


def _make_env(monkeypatch, timeline_box):
    """Installs a query-instrumented pool + postgres-mode db_integrations."""
    import STT_server.db_integrations as dbi

    query_gate = Gate(N_CONCURRENT_REFRESHES)
    http_gate = Gate(N_CONCURRENT_REFRESHES)

    def on_execute(sql, params):
        # fires while the integration re-read query is running, i.e.
        # inside the nested reader's connection scope
        if INTEGRATIONS_QUERY in str(sql):
            timeline_box["tl"].add("query_executed")
            query_gate.arrive()

    fake = FakePool(maxconn=POOL_SIZE, on_execute=on_execute)
    timeline_box["tl"] = Timeline(fake)
    _install_pool(monkeypatch, fake)
    monkeypatch.setattr(dbi, "is_postgres", lambda: True)
    monkeypatch.setattr(dbi, "_columns_check_done", True)
    return fake, dbi, query_gate, http_gate


def _spawn(fake, dbi, query_gate, http_gate, timeline, pass_cur, errors):
    def do_refresh(name):
        try:
            _refresh_path(f"int-{name}", query_gate, http_gate, timeline,
                          dbi, pass_cur)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=do_refresh, args=(f"r{i}",), name=f"r{i}",
                         daemon=True)
        for i in range(N_CONCURRENT_REFRESHES)
    ]
    for t in threads:
        t.start()
    return threads


def _victim_attempt(fake, timeline):
    """The unrelated request that arrives while every refresh is busy."""
    timeline.add("victim_attempted", thread="victim")
    try:
        with db_mod.get_conn():
            pass
    except Exception as exc:  # noqa: BLE001
        return type(exc).__name__
    return "ok"


def test_p12_prefix_nested_reader_exhausts_pool(monkeypatch):
    """PRE-FIX, real production code path.

    5 concurrent refreshes call the REAL get_integration_by_id() with no
    cur=, so each opens its own connection while holding the outer one.
    Each thread is parked inside the nested reader's query window, where
    both of its slots are genuinely checked out: 5 x 2 = 10 = maxconn.
    The 11th (unrelated) acquisition must raise PoolExhausted, and the
    pool must return to baseline once the gates are released.
    """
    timeline_box = {}
    fake, dbi, query_gate, http_gate = _make_env(monkeypatch, timeline_box)
    timeline = timeline_box["tl"]
    errors = []
    threads = _spawn(fake, dbi, query_gate, http_gate, timeline,
                     pass_cur=False, errors=errors)

    assert query_gate.all_parked.wait(timeout=15), "no query gate"
    at_query_gate = fake.in_use()

    victim = _victim_attempt(fake, timeline)

    query_gate.release.set()
    assert http_gate.all_parked.wait(timeout=15), "no http gate"
    at_http_gate = fake.in_use()
    http_gate.release.set()

    for t in threads:
        t.join(timeout=15)

    assert errors == [], f"refreshes must not error: {errors}"
    assert at_query_gate == POOL_SIZE, (
        f"pre-fix: all 10 slots held at the nested query window, got {at_query_gate}"
    )
    assert at_http_gate == N_CONCURRENT_REFRESHES, (
        f"pre-fix: inner slot released before HTTP, so {N_CONCURRENT_REFRESHES} "
        f"slots during HTTP, got {at_http_gate}"
    )
    assert victim == PoolExhausted.__name__, (
        f"pre-fix victim must hit PoolExhausted, got {victim}"
    )
    assert fake.peak_in_use == POOL_SIZE
    assert fake.in_use() == 0, "pool must return to baseline"
    assert fake.available() == POOL_SIZE

    # causal order inside every refresh thread
    for name in [f"r{i}" for i in range(N_CONCURRENT_REFRESHES)]:
        order = [e[0] for e in timeline.events if e[1] == name]
        assert order == [
            "outer_acquired", "advisory_locked", "query_executed",
            "reader_returned", "http_entered", "http_done", "outer_released",
        ], f"{name} unexpected order: {order}"
    # the victim is attempted while all refreshes sit at the query gate
    assert timeline.indices("query_executed") and timeline.indices("victim_attempted")
    assert max(timeline.indices("query_executed")) < timeline.indices("victim_attempted")[0]
    assert min(timeline.indices("reader_returned")) > timeline.indices("victim_attempted")[0]


def test_p13_postfix_cur_reader_never_exhausts_pool(monkeypatch):
    """POST-FIX, same production call path, same logical SELECT window.

    get_integration_by_id(cur=outer_cursor) runs the identical query on
    the caller's cursor, so each parked refresh holds exactly 1 slot:
    5 refreshes = 5 slots, the victim takes the 6th, peak 6.
    """
    timeline_box = {}
    fake, dbi, query_gate, http_gate = _make_env(monkeypatch, timeline_box)
    timeline = timeline_box["tl"]
    errors = []
    threads = _spawn(fake, dbi, query_gate, http_gate, timeline,
                     pass_cur=True, errors=errors)

    assert query_gate.all_parked.wait(timeout=15), "no query gate"
    at_query_gate = fake.in_use()

    victim = _victim_attempt(fake, timeline)

    query_gate.release.set()
    assert http_gate.all_parked.wait(timeout=15), "no http gate"
    at_http_gate = fake.in_use()
    http_gate.release.set()

    for t in threads:
        t.join(timeout=15)

    assert errors == [], f"refreshes must not error: {errors}"
    assert at_query_gate == N_CONCURRENT_REFRESHES, (
        f"post-fix: one slot per refresh at the query window, got {at_query_gate}"
    )
    assert at_http_gate == N_CONCURRENT_REFRESHES, (
        f"post-fix: one slot per refresh during HTTP, got {at_http_gate}"
    )
    assert victim == "ok", f"post-fix victim must succeed, got {victim}"
    assert fake.peak_in_use == N_CONCURRENT_REFRESHES + 1, (
        f"post-fix peak should be 5 refreshes + 1 victim = 6, got {fake.peak_in_use}"
    )
    assert fake.peak_in_use < POOL_SIZE
    assert fake.in_use() == 0, "pool must return to baseline"
    assert fake.available() == POOL_SIZE
    # no nested acquisition happened: 5 refreshes + 1 victim
    assert fake.getconn_calls == N_CONCURRENT_REFRESHES + 1, (
        f"post-fix acquisitions should be 6, saw {fake.getconn_calls}"
    )
    assert fake.putconn_calls == fake.getconn_calls

    for name in [f"r{i}" for i in range(N_CONCURRENT_REFRESHES)]:
        order = [e[0] for e in timeline.events if e[1] == name]
        assert order == [
            "outer_acquired", "advisory_locked", "query_executed",
            "reader_returned", "http_entered", "http_done", "outer_released",
        ], f"{name} unexpected order: {order}"


def test_p14_failing_rollback_still_returns_slot(monkeypatch):
    """The `except BaseException: conn.rollback(); raise` block sits
    BEFORE the `finally: pool.putconn(conn)`. If rollback() itself
    raises, Python still runs the finally block, so the slot is
    returned. It also means a rollback failure REPLACES the original
    exception (kept only as __context__) - documented, not fixed,
    because the pre-fix `except Exception` had the identical property.
    """
    conns = []

    def on_getconn(conn):
        conns.append(conn)

    fake2 = _install_pool(monkeypatch, FakePool(maxconn=POOL_SIZE,
                                                on_getconn=on_getconn))
    conn = None

    class Boom(Exception):
        pass

    class RollbackFails(Boom):
        pass

    def bad_rollback():
        raise RollbackFails("rollback exploded")

    def grab():
        with db_mod.get_conn() as c:
            nonlocal conn
            conn = c
            c.rollback = bad_rollback
            raise Boom("original failure")

    with pytest.raises(RollbackFails):
        grab()

    # the finally ran: the slot came back
    assert fake2.in_use() == 0
    assert fake2.available() == POOL_SIZE
    # and the original exception survives as __context__
    try:
        grab()
    except RollbackFails as exc:
        assert isinstance(exc.__context__, Boom)
    assert conn is not None



def test_p13b_no_nested_get_conn_remains_in_refresh_paths():
    """Static guard: the three production call sites must pass cur=, and
    no call to get_integration_by_id() may sit inside a `with get_conn()`
    block without cur=. Parses the real sources with ast."""
    import ast
    import pathlib

    root = pathlib.Path(db_mod.__file__).resolve().parent
    offenders = []
    checked = 0
    for rel in ("routes/api.py", "services/dynamics365.py", "db_integrations.py"):
        path = root / rel
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # track, per function, whether we're lexically inside a with-block
        # that holds a db connection
        class V(ast.NodeVisitor):
            def __init__(self):
                self.depth = 0

            def visit_With(self, node):
                holds = self._is_conn_with(node)
                self.depth += 1 if holds else 0
                try:
                    self.generic_visit(node)
                finally:
                    self.depth -= 1 if holds else 0

            @staticmethod
            def _is_conn_with(node):
                for item in node.items:
                    call = item.context_expr
                    if isinstance(call, ast.Call):
                        f = call.func
                        name = getattr(f, "id", None) or getattr(f, "attr", None)
                        if name in ("get_conn", "_get_conn"):
                            return True
                return False

            def visit_Call(self, node):
                nonlocal checked
                f = node.func
                name = getattr(f, "id", None) or getattr(f, "attr", None)
                if name in ("get_integration_by_id", "_db_get_integration_by_id",
                            "_reload"):
                    checked += 1
                    if self.depth > 0 and not any(
                        kw.arg == "cur" for kw in node.keywords
                    ):
                        offenders.append(f"{rel}:{node.lineno} nested without cur=")
                self.generic_visit(node)

        V().visit(tree)

    assert checked > 0, "expected to find get_integration_by_id call sites"
    assert offenders == [], f"nested acquisitions without cur=: {offenders}"
