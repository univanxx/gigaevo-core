from __future__ import annotations

from typing import Any

from loguru import logger

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.programs.core_types import VoidInput
from gigaevo.programs.metrics.context import VALIDITY_KEY
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import DEFAULT_CACHE  # NO_CACHE
from gigaevo.programs.stages.common import Box
from gigaevo.programs.stages.stage_registry import StageRegistry


@StageRegistry.register(
    description="Provide evolution-aware context: generation seed and ancestor metrics chain."
)
class EvolutionAwareContextStage(Stage):
    """Provides deterministic seed and ancestor metrics for fair per-generation evaluation.

    Output schema (Box[Any].data):
        {
            "seed": int | None,
            "ancestor_metrics": [
                {"fitness": float, "tokens_count": float, "is_valid": float, ...},
                ...  # ordered from parent → grandparent → ...
            ],
        }
    """

    InputsModel = VoidInput
    OutputModel = Box[Any]
    cache_handler = DEFAULT_CACHE  # NO_CACHE

    def __init__(
        self,
        *,
        storage: ProgramStorage,
        max_ancestor_depth: int = 5,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.storage = storage
        self.max_ancestor_depth = max_ancestor_depth

    async def compute(self, program: Program) -> Box[Any]:
        iteration = program.get_metadata("iteration")
        seed = iteration if iteration is not None else None

        ancestor_metrics: list[dict[str, float]] = []
        current = program
        for depth in range(self.max_ancestor_depth):
            if not current.lineage.parents:
                break
            parents = await self.storage.mget(current.lineage.parents)
            if not parents:
                break
            valid_parents = [
                p for p in parents
                if p.metrics and p.metrics.get(VALIDITY_KEY, 0) > 0
            ]
            if not valid_parents:
                break
            best_parent = max(
                valid_parents,
                key=lambda p: p.metrics.get("fitness", 0.0),
            )
            ancestor_metrics.append(dict(best_parent.metrics))
            current = best_parent
            logger.debug(
                "[EvolutionAwareContext] depth={} ancestor={} fitness={}",
                depth,
                best_parent.id,
                best_parent.metrics.get("fitness"),
            )

        logger.info(
            "[EvolutionAwareContext] program={} seed={} ancestor_chain_len={}",
            program.id,
            seed,
            len(ancestor_metrics),
        )

        context = {
            "seed": seed,
            "ancestor_metrics": ancestor_metrics,
        }
        return Box[Any](data=context)
