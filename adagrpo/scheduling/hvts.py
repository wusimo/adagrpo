"""Hierarchical Vision Task Segmenter (HVTS).

Uses a VLM (e.g. Qwen2-VL, MiniCPM-V) to perform zero-shot task decomposition
from language instructions and visual observations. Each stage is assigned a
complexity label that determines denoising budget and action horizon during
RL rollouts.

For prototyping, we provide a rule-based fallback that uses keyword matching
on language instructions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch

logger = logging.getLogger("adagrpo")


class StageComplexity(Enum):
    SIMPLE = auto()    # approach, open gripper
    MEDIUM = auto()    # grasp, place on surface
    COMPLEX = auto()   # insertion, fine alignment, tool use


@dataclass
class TaskStage:
    """A single stage in a decomposed manipulation task."""

    name: str
    description: str
    complexity: StageComplexity
    start_step: Optional[int] = None
    end_step: Optional[int] = None


@dataclass
class TaskDecomposition:
    """Full task decomposition into ordered stages."""

    task_instruction: str
    stages: list[TaskStage]

    def get_stage_at_step(self, step: int) -> TaskStage:
        """Return the active stage for a given environment step."""
        for stage in self.stages:
            if stage.start_step is not None and stage.end_step is not None:
                if stage.start_step <= step < stage.end_step:
                    return stage
        return self.stages[-1]


# Keyword → complexity mapping for rule-based fallback
_COMPLEXITY_KEYWORDS = {
    StageComplexity.SIMPLE: [
        "approach", "move to", "move toward", "open gripper", "release",
        "lift", "raise", "lower",
    ],
    StageComplexity.MEDIUM: [
        "grasp", "pick up", "place", "put", "push", "pull", "slide",
        "rotate", "turn",
    ],
    StageComplexity.COMPLEX: [
        "insert", "align", "screw", "thread", "stack precisely",
        "assemble", "connect", "fine-tune",
    ],
}


def _classify_stage_complexity(description: str) -> StageComplexity:
    """Rule-based complexity classification from stage description."""
    desc_lower = description.lower()
    for complexity in [StageComplexity.COMPLEX, StageComplexity.MEDIUM, StageComplexity.SIMPLE]:
        for kw in _COMPLEXITY_KEYWORDS[complexity]:
            if kw in desc_lower:
                return complexity
    return StageComplexity.MEDIUM  # default


class HierarchicalVisionTaskSegmenter:
    """Zero-shot task decomposition using a VLM or rule-based fallback.

    In VLM mode, queries the model with the task instruction and (optionally)
    a keyframe image to produce a list of stages with complexity labels.
    """

    def __init__(
        self,
        use_vlm: bool = False,
        vlm_model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
    ):
        self.use_vlm = use_vlm
        self.vlm_model_name = vlm_model_name
        self.device = device
        self._vlm = None

        if use_vlm:
            self._load_vlm()

    def _load_vlm(self) -> None:
        """Lazy-load the VLM for task decomposition."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("Loading VLM: %s", self.vlm_model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.vlm_model_name, trust_remote_code=True
            )
            self._vlm = AutoModelForCausalLM.from_pretrained(
                self.vlm_model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map=self.device,
            )
            logger.info("VLM loaded successfully.")
        except Exception as e:
            logger.warning("Failed to load VLM (%s), falling back to rule-based.", e)
            self.use_vlm = False

    def decompose(
        self,
        task_instruction: str,
        image: Optional[torch.Tensor] = None,
        max_episode_steps: int = 300,
    ) -> TaskDecomposition:
        """Decompose a task instruction into stages.

        Args:
            task_instruction: natural language task description.
            image: optional [C, H, W] observation image.
            max_episode_steps: total episode length for step allocation.

        Returns:
            TaskDecomposition with ordered stages.
        """
        if self.use_vlm and self._vlm is not None:
            return self._decompose_vlm(task_instruction, image, max_episode_steps)
        return self._decompose_rule_based(task_instruction, max_episode_steps)

    def _decompose_vlm(
        self,
        task_instruction: str,
        image: Optional[torch.Tensor],
        max_episode_steps: int,
    ) -> TaskDecomposition:
        """VLM-based decomposition (requires loaded VLM)."""
        prompt = (
            f"Decompose this robot manipulation task into sequential stages. "
            f"For each stage, provide a name, description, and complexity "
            f"(simple/medium/complex).\n\n"
            f"Task: {task_instruction}\n\n"
            f"Output format (one stage per line):\n"
            f"stage_name | description | complexity\n"
        )

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self._vlm.generate(**inputs, max_new_tokens=256, temperature=0.1)
        response = self._tokenizer.decode(output[0], skip_special_tokens=True)

        stages = self._parse_vlm_response(response, max_episode_steps)
        return TaskDecomposition(task_instruction=task_instruction, stages=stages)

    def _parse_vlm_response(
        self, response: str, max_episode_steps: int
    ) -> list[TaskStage]:
        """Parse VLM text output into TaskStage objects."""
        stages = []
        lines = response.strip().split("\n")
        for line in lines:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                name = parts[0]
                desc = parts[1]
                complexity_str = parts[2].lower()
                if "complex" in complexity_str:
                    complexity = StageComplexity.COMPLEX
                elif "simple" in complexity_str:
                    complexity = StageComplexity.SIMPLE
                else:
                    complexity = StageComplexity.MEDIUM
                stages.append(TaskStage(name=name, description=desc, complexity=complexity))

        if not stages:
            stages = [TaskStage(
                name="full_task",
                description="Execute the full task",
                complexity=StageComplexity.MEDIUM,
            )]

        # Allocate steps evenly
        self._allocate_steps(stages, max_episode_steps)
        return stages

    def _decompose_rule_based(
        self,
        task_instruction: str,
        max_episode_steps: int,
    ) -> TaskDecomposition:
        """Rule-based decomposition using common manipulation primitives."""
        instruction_lower = task_instruction.lower()

        # Common manipulation task patterns
        stages = []

        if "pick" in instruction_lower and "place" in instruction_lower:
            stages = [
                TaskStage("approach", "Move gripper to object", StageComplexity.SIMPLE),
                TaskStage("grasp", "Close gripper on object", StageComplexity.MEDIUM),
                TaskStage("transport", "Move object to target", StageComplexity.SIMPLE),
                TaskStage("place", "Place object at target location", StageComplexity.MEDIUM),
            ]
        elif "push" in instruction_lower:
            stages = [
                TaskStage("approach", "Move to pushing position", StageComplexity.SIMPLE),
                TaskStage("push", "Push object to target", StageComplexity.MEDIUM),
            ]
        elif "open" in instruction_lower or "close" in instruction_lower:
            stages = [
                TaskStage("approach", "Move to handle", StageComplexity.SIMPLE),
                TaskStage("grasp", "Grasp handle", StageComplexity.MEDIUM),
                TaskStage("manipulate", "Open/close object", StageComplexity.MEDIUM),
            ]
        elif "stack" in instruction_lower:
            stages = [
                TaskStage("approach", "Move to first object", StageComplexity.SIMPLE),
                TaskStage("grasp", "Grasp object", StageComplexity.MEDIUM),
                TaskStage("align", "Align above target", StageComplexity.COMPLEX),
                TaskStage("stack", "Place precisely on top", StageComplexity.COMPLEX),
            ]
        elif "insert" in instruction_lower:
            stages = [
                TaskStage("approach", "Move to peg/object", StageComplexity.SIMPLE),
                TaskStage("grasp", "Grasp object", StageComplexity.MEDIUM),
                TaskStage("align", "Align with hole/slot", StageComplexity.COMPLEX),
                TaskStage("insert", "Insert object", StageComplexity.COMPLEX),
            ]
        else:
            # Generic fallback: approach + execute
            stages = [
                TaskStage("approach", "Move to task-relevant position", StageComplexity.SIMPLE),
                TaskStage("execute", "Perform the task", StageComplexity.MEDIUM),
            ]

        self._allocate_steps(stages, max_episode_steps)
        return TaskDecomposition(task_instruction=task_instruction, stages=stages)

    @staticmethod
    def _allocate_steps(stages: list[TaskStage], max_steps: int) -> None:
        """Allocate environment steps to stages proportional to complexity."""
        complexity_weights = {
            StageComplexity.SIMPLE: 1.0,
            StageComplexity.MEDIUM: 2.0,
            StageComplexity.COMPLEX: 3.0,
        }
        total_weight = sum(complexity_weights[s.complexity] for s in stages)
        current_step = 0
        for i, stage in enumerate(stages):
            w = complexity_weights[stage.complexity]
            n_steps = int(max_steps * w / total_weight)
            if i == len(stages) - 1:
                n_steps = max_steps - current_step
            stage.start_step = current_step
            stage.end_step = current_step + n_steps
            current_step += n_steps
