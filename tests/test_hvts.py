"""Unit tests for HVTS task decomposition and budget allocation."""

import pytest

from adagrpo.scheduling.hvts import (
    HierarchicalVisionTaskSegmenter,
    StageComplexity,
    TaskDecomposition,
)
from adagrpo.scheduling.budget_allocator import BudgetAllocator, RolloutBudget


class TestHVTS:
    @pytest.fixture
    def hvts(self):
        return HierarchicalVisionTaskSegmenter(use_vlm=False)

    def test_pick_and_place_decomposition(self, hvts):
        decomp = hvts.decompose("pick up the red cube and place it on the plate")
        assert len(decomp.stages) >= 2
        stage_names = [s.name for s in decomp.stages]
        assert "approach" in stage_names
        assert "grasp" in stage_names

    def test_push_decomposition(self, hvts):
        decomp = hvts.decompose("push the block to the target")
        assert len(decomp.stages) >= 2

    def test_insert_decomposition(self, hvts):
        decomp = hvts.decompose("insert the peg into the hole")
        assert len(decomp.stages) >= 3
        complexities = [s.complexity for s in decomp.stages]
        assert StageComplexity.COMPLEX in complexities

    def test_step_allocation(self, hvts):
        decomp = hvts.decompose("pick up and place", max_episode_steps=300)
        total_steps = sum(
            (s.end_step or 0) - (s.start_step or 0)
            for s in decomp.stages
        )
        assert total_steps == 300

    def test_get_stage_at_step(self, hvts):
        decomp = hvts.decompose("pick up and place", max_episode_steps=100)
        stage_0 = decomp.get_stage_at_step(0)
        assert stage_0 is not None
        stage_last = decomp.get_stage_at_step(99)
        assert stage_last is not None

    def test_generic_fallback(self, hvts):
        decomp = hvts.decompose("do something unusual with the widget")
        assert len(decomp.stages) >= 1


class TestBudgetAllocator:
    @pytest.fixture
    def allocator(self):
        return BudgetAllocator()

    def test_simple_gets_fewer_steps(self, allocator):
        from adagrpo.scheduling.hvts import TaskStage
        simple = TaskStage("test", "test", StageComplexity.SIMPLE)
        complex_ = TaskStage("test", "test", StageComplexity.COMPLEX)

        b_simple = allocator.get_budget(simple)
        b_complex = allocator.get_budget(complex_)

        assert b_simple.num_denoise_steps < b_complex.num_denoise_steps

    def test_respects_min_max(self):
        allocator = BudgetAllocator(min_denoise_steps=3, max_denoise_steps=8)
        from adagrpo.scheduling.hvts import TaskStage

        for complexity in StageComplexity:
            stage = TaskStage("test", "test", complexity)
            budget = allocator.get_budget(stage)
            assert budget.num_denoise_steps >= 3
            assert budget.num_denoise_steps <= 8

    def test_compute_savings(self, allocator):
        from adagrpo.scheduling.hvts import TaskStage
        stages = [
            TaskStage("approach", "move", StageComplexity.SIMPLE, 0, 100),
            TaskStage("grasp", "grasp", StageComplexity.MEDIUM, 100, 200),
            TaskStage("place", "place", StageComplexity.MEDIUM, 200, 300),
        ]
        savings = allocator.compute_savings(stages)
        assert savings["total_savings_ratio"] > 1.0  # Should save compute
