"""Reads the training artifacts and keeps the team's follow-up notes in a small SQLite file."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

STATUSES = ["To call", "Called", "Offer sent"]
TASK_STATUSES = ["Not started", "In progress", "Done"]
OWNERS = ["Customer success", "Sales", "Support", "Product"]


class Store:
    def __init__(self, artifacts_dir: str | Path, db_path: str | Path):
        self.art = Path(artifacts_dir)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.load()

    # ---- artifacts -------------------------------------------------------------------------
    def load(self) -> None:
        mpath = self.art / "metrics.json"
        self.metrics: dict | None = None
        self.segments: dict | None = None
        self.customers = pd.DataFrame()
        if not mpath.exists() or not (self.art / "customers_scored.csv").exists():
            return
        self.metrics = json.loads(mpath.read_text(encoding="utf-8"))
        self.segments = json.loads((self.art / "segments.json").read_text(encoding="utf-8"))
        df = pd.read_csv(self.art / "customers_scored.csv")
        self.id_col = self.metrics["features"].get("id_column") or df.columns[0]
        self.customers = df.set_index(df.columns[0], drop=False)
        self.id_col = df.columns[0]

    @property
    def ready(self) -> bool:
        return self.metrics is not None and not self.customers.empty

    # ---- follow-ups ------------------------------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS followups ("
                "customer_id INTEGER PRIMARY KEY, status TEXT, note TEXT NOT NULL DEFAULT '', "
                "updated_at TEXT NOT NULL)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS driver_tasks ("
                "feature TEXT PRIMARY KEY, owner TEXT NOT NULL, status TEXT NOT NULL)"
            )

    def followups(self) -> dict[int, dict]:
        with self._conn() as c:
            return {r["customer_id"]: dict(r) for r in c.execute("SELECT * FROM followups")}

    def followup(self, cid: int) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM followups WHERE customer_id=?", (cid,)).fetchone()
            return dict(r) if r else None

    def set_status(self, cid: int, status: str) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute(
                "INSERT INTO followups (customer_id, status, note, updated_at) VALUES (?, ?, '', ?) "
                "ON CONFLICT(customer_id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at",
                (cid, status, now),
            )

    def set_note(self, cid: int, note: str) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute(
                "INSERT INTO followups (customer_id, status, note, updated_at) VALUES (?, NULL, ?, ?) "
                "ON CONFLICT(customer_id) DO UPDATE SET note=excluded.note, updated_at=excluded.updated_at",
                (cid, note[:2000], now),
            )

    # ---- driver tasks ----------------------------------------------------------------------
    def tasks(self) -> dict[str, dict]:
        with self._conn() as c:
            return {r["feature"]: dict(r) for r in c.execute("SELECT * FROM driver_tasks")}

    def set_task(self, feature: str, owner: str, status: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO driver_tasks (feature, owner, status) VALUES (?, ?, ?) "
                "ON CONFLICT(feature) DO UPDATE SET owner=excluded.owner, status=excluded.status",
                (feature, owner, status),
            )
