from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from typing import Any

from synapse.core.bus import Bus
from synapse.core.coordinator import Coordinator
from synapse.core.memory import SharedMemory
from synapse.dag import Node, NodeExecution
from synapse.protocols.message import Goal, Task, TaskStatus
from synapse.result import AgentResult


class Runtime:
    """Entry point that wires together Synapse core components and agent lifecycle management."""

    def __init__(self) -> None:
        self.memory = SharedMemory()
        self.bus = Bus()
        self.coordinator = Coordinator(self.bus, self.memory)
        self.results: dict[str, AgentResult] = {}
        self.executions: dict[str, NodeExecution] = {}
        self.trace: list[dict[str, Any]] = []
        self._agents: list[Any] = []
        self._agent_index: dict[str, Any] = {}
        self._pending_tasks: dict[str, Task] = {}
        self._context_ref: dict[str, dict[str, AgentResult]] = {"results": self.results}
        self._trace_enabled: bool = False
        self._results_lock = asyncio.Lock()
        self._schedule_lock = asyncio.Lock()

    def add(self, agent: Any) -> Runtime:
        """Register an agent with the runtime, bus, and coordinator."""

        agent._inject(self.bus, self.memory, self.coordinator, self)
        self._agents.append(agent)
        self._agent_index[agent.id] = agent
        self.bus.attach(agent)
        self.coordinator.register_agent(agent)
        return self

    async def start(self) -> None:
        """Start all registered agents concurrently."""

        await asyncio.gather(*(agent.start() for agent in self._agents))

    async def stop(self) -> None:
        """Stop all registered agents concurrently."""

        await asyncio.gather(*(agent.stop() for agent in self._agents))

    async def run(self, until: asyncio.Future[Any] | None = None) -> None:
        """Start agents, wait for completion or indefinitely, and always stop cleanly."""

        await self.start()
        try:
            if until is None:
                await asyncio.Future()
            else:
                await until
        finally:
            await self.stop()

    async def submit_goal(self, goal: Goal) -> list[Task]:
        """Submit a goal to the coordinator and return the created tasks."""

        self._validate_dependencies()
        tasks = await self.coordinator.submit_goal(goal)
        for task in tasks:
            self._pending_tasks[task.id] = task
        await self._schedule_ready_tasks()
        return tasks

    async def run_dag(self, goal: Goal, nodes: list[Node]) -> dict[str, AgentResult]:
        """Execute an explicit DAG for a goal and return the collected agent results."""

        node_map = self._validate_dag(nodes)
        async with self._results_lock:
            self.results.clear()
        self.executions = {node.id: NodeExecution(node_id=node.id) for node in nodes}
        self.trace = []
        self._trace_enabled = True

        successors: dict[str, list[str]] = {node.id: [] for node in nodes}
        remaining_dependencies: dict[str, int] = {}
        for node in nodes:
            remaining_dependencies[node.id] = len(node.depends_on)
            for dependency in node.depends_on:
                successors[dependency].append(node.id)

        await self.start()
        try:
            pending = set(node_map)
            ready = [
                node_map[node_id]
                for node_id, dependency_count in remaining_dependencies.items()
                if dependency_count == 0
            ]

            while pending:
                if not ready:
                    raise ValueError("DAG contains unsatisfied dependencies or a cycle.")

                batch = ready
                ready = []
                results = await asyncio.gather(
                    *(self._run_node(goal, node) for node in batch),
                    return_exceptions=True,
                )

                failures = [
                    result for result in results if isinstance(result, Exception)
                ]
                for node in batch:
                    pending.discard(node.id)
                    if self.executions[node.id].status == "complete":
                        for successor_id in successors[node.id]:
                            remaining_dependencies[successor_id] -= 1
                            if remaining_dependencies[successor_id] == 0 and successor_id in pending:
                                ready.append(node_map[successor_id])

                if failures:
                    raise failures[0]

            return dict(self.results)
        finally:
            self._trace_enabled = False
            await self.stop()

    def progress(self, goal_id: str) -> dict[str, int]:
        """Return progress information for the given goal."""

        return self.coordinator.progress(goal_id)

    def export_trace(self) -> str:
        """Return the DAG execution trace as JSON for visualization or replay tooling."""

        return json.dumps(self.trace)

    @property
    def agents(self) -> list[Any]:
        """Return a copy of the registered agents."""

        return list(self._agents)

    def _context(self) -> dict[str, dict[str, AgentResult]]:
        """Return the execution context passed to agent task handlers."""

        return self._context_ref

    async def _store_result(
        self,
        agent_id: str,
        output: Any,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Store the latest result produced by an agent."""

        async with self._results_lock:
            result = AgentResult(
                agent_id=agent_id,
                output=output,
                metadata={} if metadata is None else metadata,
            )
            self.results[result.agent_id] = result
        await self._schedule_ready_tasks()
        return result

    async def _run_node(self, goal: Goal, node: Node) -> AgentResult:
        """Run a DAG node with retries and execution metadata."""

        agent = self._agent_by_id(node.agent)
        execution = self.executions[node.id]
        last_error: Exception | None = None

        for attempt in range(node.retries + 1):
            task = self._task_for_node(goal, node, agent)
            execution.start_time = datetime.now(timezone.utc)
            execution.end_time = None
            execution.retry_count = attempt
            execution.status = "running"
            task.status = TaskStatus.IN_PROGRESS
            self._trace_event(
                "start",
                node,
                agent.id,
                status=execution.status,
                retry_count=attempt,
            )

            try:
                context = self._context()
                coroutine = agent._invoke_handle_task(task, context)
                output = (
                    await asyncio.wait_for(coroutine, timeout=node.timeout)
                    if node.timeout is not None
                    else await coroutine
                )
                task.complete(output)
                execution.end_time = datetime.now(timezone.utc)
                execution.status = "complete"
                self._trace_event(
                    "end",
                    node,
                    agent.id,
                    status=execution.status,
                    retry_count=attempt,
                )
                return await self._store_result(
                    agent.id,
                    output,
                    {"node_id": node.id, "goal_id": goal.id, "task_id": task.id},
                )
            except Exception as error:
                last_error = error
                task.fail(str(error))
                execution.end_time = datetime.now(timezone.utc)
                await agent.on_error(error)
                if attempt < node.retries:
                    execution.status = "retrying"
                    self._trace_event(
                        "retry",
                        node,
                        agent.id,
                        status=execution.status,
                        retry_count=attempt + 1,
                        error=str(error),
                    )
                else:
                    execution.status = "failed"
                    self._trace_event(
                        "fail",
                        node,
                        agent.id,
                        status=execution.status,
                        retry_count=attempt,
                        error=str(error),
                    )

        if last_error is None:
            last_error = RuntimeError(f"Node '{node.id}' failed without an error.")
        raise last_error

    def _validate_dependencies(self) -> None:
        """Ensure every declared dependency refers to a registered agent."""

        known_agent_ids = {agent.id for agent in self._agents}
        for agent in self._agents:
            for dependency in getattr(agent, "depends_on", []):
                if dependency not in known_agent_ids:
                    raise ValueError(
                        f"Agent '{agent.id}' depends on missing agent '{dependency}'."
                    )

    def _validate_dag(self, nodes: list[Node]) -> dict[str, Node]:
        """Validate node ids, agent targets, dependencies, and cycles for a DAG."""

        node_map: dict[str, Node] = {}
        for node in nodes:
            if node.id in node_map:
                raise ValueError(f"Duplicate node id '{node.id}' in DAG.")
            self._agent_by_id(node.agent)
            node_map[node.id] = node

        for node in nodes:
            for dependency in node.depends_on:
                if dependency not in node_map:
                    raise ValueError(
                        f"Node '{node.id}' depends on missing node '{dependency}'."
                    )

        temporary: set[str] = set()
        permanent: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in permanent:
                return
            if node_id in temporary:
                raise ValueError("DAG contains a cycle.")

            temporary.add(node_id)
            for dependency in node_map[node_id].depends_on:
                visit(dependency)
            temporary.remove(node_id)
            permanent.add(node_id)

        for node_id in node_map:
            visit(node_id)

        return node_map

    def _agent_by_id(self, agent_id: str) -> Any:
        """Return a registered agent by id or raise if it does not exist."""

        agent = self._agent_index.get(agent_id)
        if agent is None:
            raise ValueError(f"Node targets missing agent '{agent_id}'.")
        return agent

    def _task_for_node(self, goal: Goal, node: Node, agent: Any) -> Task:
        """Build a task object for a DAG node and goal."""

        strength = agent.profile.strengths[0] if agent.profile.strengths else agent.profile.name
        return Task(
            description=f"{strength}: {goal.description}",
            goal_id=goal.id,
            assigned_to=agent.id,
        )

    def _trace_event(
        self,
        event: str,
        node: Node,
        agent_id: str,
        **extra: Any,
    ) -> None:
        """Append a lightweight node execution event to the runtime trace."""

        if not self._trace_enabled:
            return
        entry: dict[str, Any] = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node_id": node.id,
            "agent_id": agent_id,
        }
        entry.update(extra)
        self.trace.append(entry)

    async def _schedule_ready_tasks(self) -> None:
        """Assign all pending tasks whose agent dependencies have completed."""

        async with self._schedule_lock:
            while True:
                ready_tasks: list[Task] = []
                for task in self._pending_tasks.values():
                    if task.assigned_to is None:
                        ready_tasks.append(task)
                        continue

                    agent = self.coordinator._agents.get(task.assigned_to)
                    if agent is None:
                        raise ValueError(f"Task '{task.id}' targets missing agent '{task.assigned_to}'.")

                    dependencies = getattr(agent, "depends_on", [])
                    if all(dependency in self.results for dependency in dependencies):
                        ready_tasks.append(task)

                if not ready_tasks:
                    break

                for task in ready_tasks:
                    self._pending_tasks.pop(task.id, None)

                await asyncio.gather(
                    *(self.coordinator.assign_task(task) for task in ready_tasks)
                )
