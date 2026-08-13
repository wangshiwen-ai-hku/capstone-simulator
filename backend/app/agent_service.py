"""Incremental, memory-backed MARS workflow modelling agent."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
import ssl
from threading import RLock
from time import perf_counter
from typing import Any, TypedDict
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from uuid import uuid4

import certifi
from openai import OpenAI
from pydantic import BaseModel, Field

from .config import Settings
from .mars_adapter import validate_scene
from .scene_generator import build_deterministic_scene
from .schedulability import ensure_generated_scene_schedulable
from .schemas import (
    AgentAtomicTaskPlan,
    AgentChatRequest,
    AgentChatResponse,
    AgentSource,
    AgentStructuredInfo,
    BenchmarkScene,
    Difficulty,
    GenerateSceneRequest,
    TaskCategory,
)

logger = logging.getLogger(__name__)

KNOWN_TASK_TYPES = {
    "localization": TaskCategory.localization,
    "object_detection": TaskCategory.object_detection,
    "environment_understanding": TaskCategory.environment_understanding,
    "semantic_segmentation": TaskCategory.semantic_segmentation,
    "local_planning": TaskCategory.local_planning,
    "obstacle_avoidance": TaskCategory.obstacle_avoidance,
    "local_control": TaskCategory.local_control,
    "emergency_stop": TaskCategory.emergency_stop,
    "data_compression": TaskCategory.data_compression,
    "local_llm_7b": TaskCategory.local_llm_7b,
    "local_llm_10b": TaskCategory.local_llm_10b,
    "result_verification": TaskCategory.result_verification,
    "map_fusion": TaskCategory.map_fusion,
}


class AgentState(TypedDict, total=False):
    thread_id: str
    request: AgentChatRequest
    session: "ModellingSession"
    sources: list[AgentSource]
    response: AgentChatResponse


@dataclass
class ModellingSession:
    phase: str = "discovery"
    requirements: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, str]] = field(default_factory=list)
    atomic_tasks: list[AgentAtomicTaskPlan] = field(default_factory=list)
    sources: list[AgentSource] = field(default_factory=list)
    scene: BenchmarkScene | None = None
    planned_by_model: bool = False
    provenance: str = "local_intake"
    effective_model: str | None = None
    diagnostic: str = ""


class DiscoveryPayload(BaseModel):
    summary: str
    questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class SemanticWorkflowPayload(BaseModel):
    summary: str
    pipelines: dict[str, list[str]]
    reasons: dict[str, str] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)


DISCOVERY_SYSTEM_PROMPT = """
You are the discovery node of MARS Agent. Read the current multi-robot scenario
and ask 1-3 specific questions whose answers materially affect task selection,
dependencies, timing, placement, or optimization. Do not generate a workflow.
Return compact JSON: {"summary": string, "questions": [string],
"assumptions": [string]}. Answer in the user's language.
"""


PLAN_SYSTEM_PROMPT = """
You are the semantic planning node of MARS Agent. Do not produce IDs, deadlines,
ports, placement schemas, or a formal DAG; the backend compiler owns those.
Choose only the atomic capabilities needed by each robot and put them in causal
order. Allowed values: {task_types}. Return compact JSON:
{{"summary": string, "pipelines": {{"robot_1": [task_type, ...]}},
"reasons": {{task_type: string}}, "assumptions": [string],
"insights": [string]}}. Include every declared robot exactly once. Use 2-8
capabilities per robot. Answer summary/reasons in the user's language.
"""


class MarsAgentService:
    """LangGraph-guided discovery, planning, review and compilation workflow."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._sessions: dict[str, ModellingSession] = {}
        self._lock = RLock()
        self._graph = self._build_graph()

    def _build_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:
            logger.warning("langgraph unavailable; using sequential phase router")
            return None
        graph = StateGraph(AgentState)
        graph.add_node("turn", self._turn_node)
        graph.add_edge(START, "turn")
        graph.add_edge("turn", END)
        return graph.compile()

    def chat(self, request: AgentChatRequest) -> AgentChatResponse:
        thread_id = request.thread_id or f"agent_{uuid4().hex[:12]}"
        with self._lock:
            if request.action == "restart":
                self._sessions.pop(thread_id, None)
            session = self._sessions.setdefault(thread_id, ModellingSession())
        state: AgentState = {
            "thread_id": thread_id,
            "request": request,
            "session": session,
        }
        if self._graph is None:
            state.update(self._turn_node(state))
        else:
            state = self._graph.invoke(state)
        response = state["response"].model_copy(update={"thread_id": thread_id})
        with self._lock:
            self._sessions[thread_id] = session
        return response

    def _turn_node(self, state: AgentState) -> AgentState:
        request, session = state["request"], state["session"]
        if request.current_scene is not None:
            source_robot = next(
                (
                    node
                    for node in request.current_scene.nodes
                    if node.kind == "robot"
                ),
                None,
            )
            hardware_by_architecture = {
                "jetson-orin-nano": "orin_nano",
                "jetson-orin-nx": "orin_nx",
                "jetson-agx-orin": "orin_agx",
            }
            if source_robot is not None:
                session.requirements["robot_hardware"] = (
                    hardware_by_architecture.get(
                        source_robot.architecture,
                        "orin_nx",
                    )
                )
        if request.action == "restart":
            return {"response": self._discovery_response(request, session, reset=True)}

        if request.action == "confirm":
            if session.phase != "review" or not session.atomic_tasks:
                return {"response": self._plan_response(request, session)}
            return {"response": self._compile_response(request, session)}

        session.messages.append({"role": "user", "content": request.message})
        session.messages = session.messages[-12:]
        self._merge_requirements(session.requirements, request.message)

        if session.phase == "discovery" and len(session.messages) == 1:
            return {"response": self._discovery_response(request, session)}
        return {"response": self._plan_response(request, session)}

    def _discovery_response(
        self,
        request: AgentChatRequest,
        session: ModellingSession,
        *,
        reset: bool = False,
    ) -> AgentChatResponse:
        session.phase = "discovery"
        requirements = session.requirements
        robots = requirements.get("robot_count", 2)
        interval = requirements.get("interval_ms")
        understood = f"我先记录了 {robots} 台机器人"
        if interval:
            understood += f"、任务间隔 {interval / 1000:g} 秒"
        understood += "。这一轮先补齐约束，下一轮再规划原子任务，避免一次性生成完整 workflow。"
        if reset:
            understood = "已清空当前建模记忆。请描述机器人、任务到达方式和优化目标。"
        questions = [
            "每个任务的起点、终点或服务目标是什么？",
            "优先优化按时完成率、总完工时间、能耗，还是三者的加权组合？",
            "是否需要避障、视觉识别或边缘节点卸载？",
        ]
        assumptions = self._assumptions(requirements)
        fallback = False
        provenance = "local_intake"
        effective_model = None
        diagnostic = ""
        cfg = self.settings.apiyi_agent_config(request.model)
        if cfg.get("api_key") and not reset:
            try:
                content, effective_model, call_diagnostic = self._call_api_model(
                    request.model,
                    DISCOVERY_SYSTEM_PROMPT,
                    {"requirements": requirements, "message": request.message},
                    timeout=min(12, self.settings.agent_model_timeout_seconds),
                    operation="discovery",
                )
                try:
                    discovery = DiscoveryPayload.model_validate(
                        self._extract_json_object(content)
                    )
                    understood = discovery.summary
                    questions = discovery.questions[:3] or questions
                    assumptions = discovery.assumptions or assumptions
                    provenance = "api"
                except Exception as exc:
                    recovered = self._recover_questions(content)
                    understood = content.strip()[:500] or understood
                    questions = recovered[:3] or questions
                    provenance = "api_recovered"
                    diagnostic = f"API response was not valid discovery JSON; recovered useful text ({type(exc).__name__})."
                if call_diagnostic:
                    diagnostic = " ".join(filter(None, [diagnostic, call_diagnostic]))
            except Exception as exc:
                fallback = True
                provenance = "local_intake"
                diagnostic = self._diagnostic(exc)
                logger.warning("MARS Agent discovery failed; local intake used: %s", diagnostic)
        session.provenance = provenance
        session.effective_model = effective_model
        session.diagnostic = diagnostic
        return self._response(
            request,
            session,
            message=understood,
            questions=questions,
            insights=["需求澄清完成后，我会先展示原子任务 DAG，确认后才编译 Studio workflow。"],
            assumptions=assumptions,
            fallback=fallback,
            provenance=provenance,
            effective_model=effective_model,
            diagnostic=diagnostic,
            progress=20,
        )

    def _plan_response(
        self,
        request: AgentChatRequest,
        session: ModellingSession,
    ) -> AgentChatResponse:
        sources = self._retrieve(request.message) if request.enable_web_search else []
        session.sources = sources
        fallback = False
        cfg = self.settings.apiyi_agent_config(request.model)
        if cfg.get("api_key"):
            try:
                session.atomic_tasks, plan, provenance, effective_model, diagnostic = self._plan_with_model(
                    request.model,
                    session,
                )
                session.planned_by_model = True
                message = plan.summary
                assumptions = plan.assumptions
                insights = plan.insights
                session.provenance = provenance
                session.effective_model = effective_model
                session.diagnostic = diagnostic
            except Exception as exc:
                diagnostic = self._diagnostic(exc)
                logger.warning("MARS Agent planning failed; local plan used: %s", diagnostic)
                session.atomic_tasks = self._local_plan(session.requirements)
                session.planned_by_model = False
                message = "API 规划不可用；我先生成了可继续编辑的本地原子任务计划。"
                assumptions = self._assumptions(session.requirements)
                insights = ["你可以直接指出要增加、删除或重排的任务，我会在下一轮重新规划。"]
                fallback = True
                session.provenance = "local_fallback"
                session.effective_model = None
                session.diagnostic = diagnostic
        else:
            session.atomic_tasks = self._local_plan(session.requirements)
            session.planned_by_model = False
            message = "已根据当前对话规划原子任务。请检查任务、依赖和到达时间，再确认编译。"
            assumptions = self._assumptions(session.requirements)
            insights = ["当前未配置 APIYI，使用本地可编辑计划。"]
            fallback = True
            session.provenance = "local_fallback"
            session.effective_model = None
            session.diagnostic = "APIYI_KEY is not configured."
        session.phase = "review"
        session.messages.append({"role": "assistant", "content": message})
        return self._response(
            request,
            session,
            message=message,
            questions=["这个原子任务计划是否可以编译？也可以继续输入修改意见。"],
            insights=insights,
            assumptions=assumptions,
            fallback=fallback,
            provenance=session.provenance,
            effective_model=session.effective_model,
            diagnostic=session.diagnostic,
            progress=65,
        )

    def _compile_response(
        self,
        request: AgentChatRequest,
        session: ModellingSession,
    ) -> AgentChatResponse:
        scene = self._compile_scene(session)
        validate_scene(scene)
        session.scene = scene
        session.phase = "ready"
        return self._response(
            request,
            session,
            message=(
                f"已将确认的 {len(session.atomic_tasks)} 个原子任务编译为 Studio workflow。"
                "后端已验证 DAG、依赖、节点放置和端口契约，可以导入。"
            ),
            insights=["后续对话仍保留当前需求和任务计划；输入修改意见即可重新规划。"],
            scene=scene,
            progress=100,
        )

    def _plan_with_model(
        self,
        requested_model: str,
        session: ModellingSession,
    ) -> tuple[
        list[AgentAtomicTaskPlan],
        SemanticWorkflowPayload,
        str,
        str,
        str,
    ]:
        context = {
            "requirements": session.requirements,
            "conversation": session.messages[-8:],
        }
        content, effective_model, call_diagnostic = self._call_api_model(
            requested_model,
            PLAN_SYSTEM_PROMPT.format(
                task_types=", ".join(sorted(KNOWN_TASK_TYPES))
            ),
            context,
            timeout=self.settings.agent_model_timeout_seconds,
            operation="semantic planning",
        )
        provenance = "api"
        diagnostic = call_diagnostic
        try:
            plan = SemanticWorkflowPayload.model_validate(
                self._extract_json_object(content)
            )
        except Exception as exc:
            pipelines = self._recover_pipelines(content, session.requirements)
            if not pipelines:
                raise ValueError(
                    "API returned content, but no supported MARS task types could be recovered"
                ) from exc
            plan = SemanticWorkflowPayload(
                summary=content.strip()[:600],
                pipelines=pipelines,
                assumptions=self._assumptions(session.requirements),
                insights=["Recovered semantic task choices from a non-standard API response."],
            )
            provenance = "api_recovered"
            diagnostic = " ".join(filter(None, [
                diagnostic,
                f"Semantic JSON validation failed ({type(exc).__name__}); supported task types were recovered from text.",
            ]))
        tasks = self._compile_semantic_pipelines(plan, session.requirements)
        return tasks, plan, provenance, effective_model, diagnostic

    def _compile_semantic_pipelines(
        self,
        plan: SemanticWorkflowPayload,
        requirements: dict[str, Any],
    ) -> list[AgentAtomicTaskPlan]:
        robot_count = int(requirements.get("robot_count", 2))
        interval = int(requirements.get("interval_ms", 0))
        tasks: list[AgentAtomicTaskPlan] = []
        for robot_index in range(robot_count):
            robot_id = f"robot_{robot_index + 1}"
            raw_pipeline = plan.pipelines.get(robot_id) or plan.pipelines.get("default") or []
            pipeline = [task_type for task_type in raw_pipeline if task_type in KNOWN_TASK_TYPES]
            pipeline = list(dict.fromkeys(pipeline))[:8]
            if not pipeline:
                raise ValueError(f"API semantic plan omitted supported tasks for {robot_id}")
            previous = ""
            arrival = robot_index * interval
            for stage, task_type in enumerate(pipeline):
                task_id = f"task_{len(tasks) + 1:03d}"
                safety = task_type in {"local_control", "obstacle_avoidance", "emergency_stop"}
                tasks.append(AgentAtomicTaskPlan(
                    id=task_id,
                    name=f"{task_type.replace('_', ' ').title()} / Robot {robot_index + 1}",
                    task_type=task_type,
                    purpose=plan.reasons.get(task_type, f"Required by the API semantic plan for {robot_id}."),
                    source_robot_id=robot_id,
                    dependencies=[previous] if previous else [],
                    arrival_time_ms=arrival,
                    deadline_ms=arrival + 1500 + stage * 500,
                    priority=5 if safety else 3,
                    placement_hint="source robot" if safety else "robot or edge",
                ))
                previous = task_id
        if not tasks or len(tasks) > 24:
            raise ValueError("compiled semantic plan has an invalid atomic task count")
        return tasks

    def _local_plan(self, requirements: dict[str, Any]) -> list[AgentAtomicTaskPlan]:
        robots = int(requirements.get("robot_count", 2))
        interval = int(requirements.get("interval_ms", 0))
        types = requirements.get("task_types") or [
            "localization",
            "object_detection",
            "local_planning",
            "local_control",
        ]
        tasks: list[AgentAtomicTaskPlan] = []
        for robot_index in range(robots):
            previous = ""
            arrival = robot_index * interval
            for task_type in types:
                task_id = f"task_{len(tasks) + 1:03d}"
                tasks.append(AgentAtomicTaskPlan(
                    id=task_id,
                    name=f"{task_type.replace('_', ' ').title()} / Robot {robot_index + 1}",
                    task_type=task_type,
                    purpose=f"Execute {task_type.replace('_', ' ')} for the requested service task.",
                    source_robot_id=f"robot_{robot_index + 1}",
                    dependencies=[previous] if previous else [],
                    arrival_time_ms=arrival,
                    deadline_ms=arrival + 1500 + len(tasks) * 250,
                    priority=5 if task_type in {"local_control", "obstacle_avoidance", "emergency_stop"} else 3,
                    placement_hint="source robot" if task_type in {"local_control", "obstacle_avoidance", "emergency_stop"} else "robot or edge",
                ))
                previous = task_id
        return tasks

    def _compile_scene(self, session: ModellingSession) -> BenchmarkScene:
        requirements = session.requirements
        robot_count = int(requirements.get("robot_count", 2))
        categories = [KNOWN_TASK_TYPES[task.task_type] for task in session.atomic_tasks]
        unique_categories = list(dict.fromkeys(categories))
        base = build_deterministic_scene(GenerateSceneRequest(
            scenario_type="custom",
            custom_scene=str(requirements.get("description", "MARS Agent workflow")),
            robot_count=robot_count,
            edge_count=int(requirements.get("edge_count", 1)),
            task_categories=unique_categories,
            difficulty=Difficulty.medium,
            seed=7,
            use_llm=False,
            robot_hardware=str(
                requirements.get("robot_hardware", "orin_nx")
            ),
        ))
        prototypes = {task.task_type: task for task in base.tasks}
        compiled = []
        for index, plan in enumerate(session.atomic_tasks):
            prototype = prototypes[plan.task_type].model_copy(deep=True)
            prototype.id = plan.id
            prototype.name = plan.name
            prototype.source_robot_id = plan.source_robot_id
            prototype.dependencies = list(plan.dependencies)
            prototype.arrival_time_ms = plan.arrival_time_ms
            prototype.deadline_ms = plan.deadline_ms
            prototype.latency_budget_ms = min(
                prototype.latency_budget_ms,
                max(1.0, plan.deadline_ms - plan.arrival_time_ms),
            )
            prototype.priority = plan.priority
            prototype.stage_index = self._stage(plan, session.atomic_tasks)
            prototype.input_ports = []
            prototype.output_ports = []
            compiled.append(prototype)
        base.id = f"scene_agent_{uuid4().hex[:8]}"
        base.workflow_id = f"workflow_{base.id}"
        base.title = self._title(str(requirements.get("description", "MARS Agent workflow")))
        base.tasks = compiled
        base.data_edges = []
        base.workflow_deadline_ms = max(task.deadline_ms for task in compiled) * 1.1
        base.generation_source = "llm" if session.planned_by_model else "deterministic_fallback"
        base.generation_note = "Compiled from the user-confirmed MARS Agent atomic-task plan"
        ensure_generated_scene_schedulable(base)
        return base

    @staticmethod
    def _stage(task: AgentAtomicTaskPlan, tasks: list[AgentAtomicTaskPlan]) -> int:
        by_id = {item.id: item for item in tasks}
        memo: dict[str, int] = {}
        def level(task_id: str) -> int:
            if task_id in memo:
                return memo[task_id]
            item = by_id[task_id]
            memo[task_id] = 0 if not item.dependencies else 1 + max(level(dep) for dep in item.dependencies)
            return memo[task_id]
        return level(task.id)

    def _call_api_model(
        self,
        requested_model: str,
        system_prompt: str,
        context: dict[str, Any],
        *,
        timeout: int,
        operation: str,
    ) -> tuple[str, str, str]:
        candidates = [requested_model]
        if requested_model == "deepseek-v4-flash":
            candidates.append("gemini-3.1-flash-lite")
        errors: list[str] = []
        per_attempt_timeout = min(timeout, 20) if len(candidates) > 1 else timeout
        for candidate in candidates:
            cfg = self.settings.apiyi_agent_config(candidate)
            if not cfg.get("api_key"):
                continue
            model_name = str(cfg["model"])
            started = perf_counter()
            logger.info(
                "MARS Agent %s started: requested=%s effective=%s timeout=%ss",
                operation,
                requested_model,
                model_name,
                per_attempt_timeout,
            )
            try:
                client = OpenAI(
                    api_key=str(cfg["api_key"]),
                    base_url=str(cfg["base_url"]),
                    timeout=per_attempt_timeout,
                    max_retries=0,
                )
                completion = client.chat.completions.create(
                    model=model_name,
                    temperature=self.settings.llm_temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
                    ],
                )
                content = completion.choices[0].message.content or ""
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("API returned empty content")
                elapsed_ms = (perf_counter() - started) * 1000
                logger.info(
                    "MARS Agent %s completed: effective=%s bytes=%d elapsed_ms=%.1f",
                    operation,
                    model_name,
                    len(content),
                    elapsed_ms,
                )
                fallback_note = ""
                if candidate != requested_model:
                    fallback_note = (
                        f"Requested {requested_model} failed; APIYI fallback model "
                        f"{candidate} completed the turn."
                    )
                return content, candidate, fallback_note
            except Exception as exc:
                error = self._diagnostic(exc)
                errors.append(f"{candidate}: {error}")
                logger.warning(
                    "MARS Agent %s failed: effective=%s error=%s",
                    operation,
                    model_name,
                    error,
                )
        raise RuntimeError("; ".join(errors) or "No configured APIYI model candidate")

    @staticmethod
    def _extract_json_object(content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise
            parsed = json.loads(text[start:end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("API response root must be an object")
        return parsed

    @staticmethod
    def _recover_questions(content: str) -> list[str]:
        questions = []
        for line in re.split(r"[\r\n]+", content):
            cleaned = re.sub(r"^[\s\-*\d.、]+", "", line).strip()
            if cleaned.endswith(("?", "？")) and cleaned not in questions:
                questions.append(cleaned)
        return questions

    @staticmethod
    def _recover_pipelines(
        content: str,
        requirements: dict[str, Any],
    ) -> dict[str, list[str]]:
        lowered = content.lower()
        aliases = {
            "localization": ("localization", "定位"),
            "object_detection": ("object_detection", "object detection", "目标检测", "视觉识别"),
            "environment_understanding": ("environment_understanding", "环境理解"),
            "semantic_segmentation": ("semantic_segmentation", "语义分割"),
            "local_planning": ("local_planning", "local planning", "路径规划", "局部规划"),
            "obstacle_avoidance": ("obstacle_avoidance", "obstacle avoidance", "避障"),
            "local_control": ("local_control", "local control", "运动控制", "本地控制"),
            "emergency_stop": ("emergency_stop", "emergency stop", "紧急停止"),
            "data_compression": ("data_compression", "data compression", "数据压缩"),
            "local_llm_7b": ("local_llm_7b", "llm planning", "语言模型规划"),
            "result_verification": ("result_verification", "result verification", "结果验证"),
            "map_fusion": ("map_fusion", "map fusion", "地图融合"),
        }
        positions: list[tuple[int, str]] = []
        for task_type, names in aliases.items():
            found = [lowered.find(name.lower()) for name in names if lowered.find(name.lower()) >= 0]
            if found:
                positions.append((min(found), task_type))
        ordered = [task_type for _, task_type in sorted(positions)]
        if not ordered:
            ordered = [
                task_type for task_type in requirements.get("task_types", [])
                if task_type in KNOWN_TASK_TYPES
            ]
        robot_count = int(requirements.get("robot_count", 2))
        return {
            f"robot_{index + 1}": list(dict.fromkeys(ordered))
            for index in range(robot_count)
        } if ordered else {}

    @staticmethod
    def _diagnostic(exc: Exception) -> str:
        message = re.sub(r"\s+", " ", str(exc)).strip()
        if not message:
            message = type(exc).__name__
        return f"{type(exc).__name__}: {message[:500]}"

    def _retrieve(self, query: str) -> list[AgentSource]:
        sources = [AgentSource(
            title="MARS workload and placement contract",
            snippet="Studio tasks use dependencies, arrival/deadline times and placement constraints.",
        )]
        if not self.settings.agent_web_search:
            return sources
        url = (
            "https://export.arxiv.org/api/query?search_query=all:"
            f"{quote_plus('multi robot task allocation scheduling ' + query[:180])}"
            "&start=0&max_results=3"
        )
        try:
            context = ssl.create_default_context(cafile=certifi.where())
            request = Request(url, headers={"User-Agent": "MARS-Agent/1.0"})
            with urlopen(
                request,
                timeout=self.settings.agent_search_timeout_seconds,
                context=context,
            ) as response:
                text = response.read().decode("utf-8", errors="replace")
            for entry in re.findall(r"<entry>(.*?)</entry>", text, re.S):
                title = re.sub(r"\s+", " ", self._xml(entry, "title")).strip()
                if title:
                    sources.append(AgentSource(
                        title=title,
                        url=self._xml(entry, "id").strip(),
                        snippet=re.sub(r"\s+", " ", self._xml(entry, "summary")).strip()[:280],
                        kind="web",
                    ))
            logger.info("MARS Agent retrieval completed: web_results=%d", len(sources) - 1)
        except Exception as exc:
            logger.info("Agent retrieval unavailable; continuing without web sources: %s", exc)
        return sources

    def _response(
        self,
        request: AgentChatRequest,
        session: ModellingSession,
        *,
        message: str,
        questions: list[str] | None = None,
        insights: list[str] | None = None,
        assumptions: list[str] | None = None,
        fallback: bool = False,
        scene: BenchmarkScene | None = None,
        provenance: str | None = None,
        effective_model: str | None = None,
        diagnostic: str | None = None,
        progress: int,
    ) -> AgentChatResponse:
        return AgentChatResponse(
            thread_id="pending",
            message=message,
            model=request.model,
            fallback=fallback,
            questions=questions or [],
            insights=insights or [],
            suggested_nodes=list(dict.fromkeys(task.task_type for task in session.atomic_tasks)),
            sources=session.sources,
            structured_info=AgentStructuredInfo(
                task_spec={
                    "robot_count": session.requirements.get("robot_count"),
                    "atomic_task_count": len(session.atomic_tasks),
                },
                workflow_spec={
                    "phase": session.phase,
                    "dependencies": {
                        task.id: task.dependencies for task in session.atomic_tasks
                    },
                },
                assumptions=assumptions or self._assumptions(session.requirements),
            ),
            scene_draft=scene,
            ready_to_import=scene is not None,
            phase=session.phase,
            progress=progress,
            atomic_tasks=session.atomic_tasks,
            provenance=provenance or session.provenance,
            effective_model=(
                effective_model
                if effective_model is not None
                else session.effective_model
            ),
            diagnostic=(diagnostic if diagnostic is not None else session.diagnostic),
        )

    @staticmethod
    def _merge_requirements(requirements: dict[str, Any], message: str) -> None:
        requirements["description"] = (
            f"{requirements.get('description', '')} {message}"
        ).strip()
        requirements["robot_count"] = MarsAgentService._extract_count(message, requirements.get("robot_count", 2))
        interval = MarsAgentService._extract_interval_ms(message)
        if interval is not None:
            requirements["interval_ms"] = interval
        lowered = message.lower()
        types = list(requirements.get("task_types", []))
        mapping = {
            "定位": "localization", "localization": "localization",
            "识别": "object_detection", "视觉": "object_detection", "detection": "object_detection",
            "路径": "local_planning", "规划": "local_planning", "planning": "local_planning",
            "避障": "obstacle_avoidance", "obstacle": "obstacle_avoidance",
            "控制": "local_control", "control": "local_control",
        }
        for keyword, task_type in mapping.items():
            if keyword in lowered and task_type not in types:
                types.append(task_type)
        if any(word in lowered for word in ("取货", "配送", "pickup", "delivery")):
            for task_type in ("localization", "object_detection", "local_planning", "local_control"):
                if task_type not in types:
                    types.append(task_type)
        requirements["task_types"] = types
        requirements["edge_count"] = 0 if any(word in lowered for word in ("无边缘", "no edge")) else 1

    @staticmethod
    def _extract_count(text: str, fallback: int) -> int:
        chinese = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5}
        matches = re.findall(
            r"([1-9]\d*|[一两二三四五])\s*(?:个|台)?(?:(?![一两二三四五1-9，。,.]).){0,10}机器人",
            text,
            re.I,
        )
        if not matches:
            return fallback
        value = matches[-1]
        return min(50, int(value) if value.isdigit() else chinese[value])

    @staticmethod
    def _extract_interval_ms(text: str) -> int | None:
        match = re.search(
            r"(?:间隔|相隔|every)\s*([0-9]+|一|两|二)\s*(分钟|分|秒|minute|second)",
            text,
            re.I,
        )
        if not match:
            return None
        value = {"一": 1, "两": 2, "二": 2}.get(
            match.group(1),
            int(match.group(1)) if match.group(1).isdigit() else 1,
        )
        return value * (60_000 if match.group(2).lower() in {"分钟", "分", "minute"} else 1_000)

    @staticmethod
    def _assumptions(requirements: dict[str, Any]) -> list[str]:
        return [
            f"当前按 {requirements.get('robot_count', 2)} 台机器人建模。",
            "未明确的资源需求在最终编译时采用 MARS workload profile。",
            "只有用户确认后的原子任务计划才会编译成 Studio workflow。",
        ]

    @staticmethod
    def _xml(entry: str, tag: str) -> str:
        match = re.search(fr"<{tag}[^>]*>(.*?)</{tag}>", entry, re.S)
        return match.group(1) if match else ""

    @staticmethod
    def _title(text: str) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        return (compact[:45] + "...") if len(compact) > 45 else compact
