from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator


@dataclass
class TaskChannel:
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    completed: bool = False
    task: asyncio.Task | None = None
    cancellation_event: asyncio.Event = field(default_factory=asyncio.Event)


class TaskEventBroker:
    def __init__(self) -> None:
        self._channels: dict[str, TaskChannel] = {}
        self._lock = asyncio.Lock()

    async def create(self, task_id: str) -> None:
        async with self._lock:
            self._channels[task_id] = TaskChannel()

    async def attach_task(self, task_id: str, task: asyncio.Task) -> None:
        async with self._lock:
            self._channels[task_id].task = task

    async def cancellation_event(self, task_id: str) -> asyncio.Event:
        async with self._lock:
            channel = self._channels.get(task_id)
            if channel is None:
                raise KeyError(task_id)
            return channel.cancellation_event

    async def cancel(self, task_id: str) -> None:
        async with self._lock:
            channel = self._channels.get(task_id)
            if channel is None:
                raise KeyError(task_id)
            channel.cancellation_event.set()

    async def publish(self, task_id: str, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        event = {
            "task_id": task_id, "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(), "payload": payload or {},
        }
        async with self._lock:
            channel = self._channels.setdefault(task_id, TaskChannel())
            channel.events.append(event)
            subscribers = list(channel.subscribers)
        for queue in subscribers:
            await queue.put(event)
        return event

    async def complete(self, task_id: str) -> None:
        async with self._lock:
            channel = self._channels.setdefault(task_id, TaskChannel())
            channel.completed = True
            subscribers = list(channel.subscribers)
        for queue in subscribers:
            await queue.put(None)

    async def stream(self, task_id: str) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            channel = self._channels.get(task_id)
            if channel is None:
                raise KeyError(task_id)
            history = list(channel.events)
            completed = channel.completed
            if not completed:
                channel.subscribers.add(queue)
        for event in history:
            yield event
        if completed:
            return
        try:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event
        finally:
            async with self._lock:
                channel = self._channels.get(task_id)
                if channel:
                    channel.subscribers.discard(queue)
                    if not channel.completed and not channel.subscribers:
                        channel.cancellation_event.set()


task_event_broker = TaskEventBroker()
