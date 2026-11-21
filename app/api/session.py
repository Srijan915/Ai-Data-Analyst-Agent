"""Thread-safe session management for the FastAPI backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from typing import Any, Dict, List, Optional
import uuid

import pandas as pd

from app.core.agent import DataAnalystAgent
from app.core.dataset_catalog import DatasetCatalog
from app.core.disk_dataset import DiskDataset
from app.models.schemas import CatalogProfile, DatasetMetadata, DatasetProfile, RelationshipSuggestion
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SessionState:
    session_id: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent: Optional[DataAnalystAgent] = None
    catalog: Optional[DatasetCatalog] = None
    disk_dataset: Optional[DiskDataset] = None
    dataframe: Optional[pd.DataFrame] = None
    dataset_profile: Optional[DatasetProfile | CatalogProfile] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    last_result: Optional[Dict[str, Any]] = None

    def release(self) -> None:
        """Safely release underlying agent, catalog, and disk resources."""
        if self.agent is not None:
            try:
                self.agent.release_dataset()
            except Exception as exc:
                logger.warning("Error releasing agent dataset: %s", exc)
            self.agent = None

        if self.catalog is not None:
            try:
                self.catalog.close()
            except Exception as exc:
                logger.warning("Error closing catalog: %s", exc)
            self.catalog = None

        if self.disk_dataset is not None:
            try:
                self.disk_dataset.close()
            except Exception as exc:
                logger.warning("Error closing disk dataset: %s", exc)
            self.disk_dataset = None

        self.dataframe = None
        self.dataset_profile = None
        self.conversation_history = []
        self.last_result = None

    def get_datasets(self) -> List[DatasetMetadata]:
        if self.catalog is not None:
            return list(self.catalog.profile.datasets)
        return []

    def get_relationships(self) -> List[RelationshipSuggestion]:
        if self.catalog is not None:
            return self.catalog.profile.relationships
        return []


class SessionManager:
    """Manages active user sessions with thread safety and lifecycle isolation."""

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.RLock()

    def get_or_create(self, session_id: Optional[str] = None) -> SessionState:
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]

            new_id = session_id or str(uuid.uuid4())
            session = SessionState(session_id=new_id)
            self._sessions[new_id] = session
            return session

    def get(self, session_id: str) -> Optional[SessionState]:
        with self._lock:
            return self._sessions.get(session_id)

    def remove(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                session.release()

    def clear_all(self) -> None:
        with self._lock:
            for session in self._sessions.values():
                session.release()
            self._sessions.clear()


# Global singleton instance for the backend process
session_manager = SessionManager()
