from __future__ import annotations

import unittest

from mars.domain import TaskClass
from mars.synthetic_workloads import (
    ExecutionTarget,
    SyntheticSampler,
    SyntheticWorkloadCatalog,
    UnsupportedTargetError,
    load_default_synthetic_workloads,
)


def custom_workload() -> dict:
    profile = {
        "latency_ms": {"p50": 10, "p95": 15, "p99": 20},
        "resources": {"cpu_cores": 1, "gpu_units": 0, "memory_mb": 128},
        "input_size_mb": {"min": 0.1, "typical": 0.2, "max": 0.4},
        "output_size_mb": {"min": 0.01, "typical": 0.02, "max": 0.04},
        "energy_j": {"min": 0.1, "typical": 0.2, "max": 0.4},
        "failure_rate": 0.01,
        "accuracy": {"min": 0.8, "typical": 0.9, "max": 0.95},
        "max_concurrency": 3,
    }
    return {
        "task_type": "custom_inspector",
        "display_name": "Custom inspector",
        "task_class": "realtime_offloadable",
        "description": "Test fixture",
        "model_variant": "custom-fixture",
        "accelerator_demand_tops": 3.5,
        "inputs": [{"name": "sample", "semantic_type": "sample"}],
        "outputs": [{"name": "result", "semantic_type": "inspection_result"}],
        "profiles": {"orin": dict(profile), "edge": dict(profile)},
    }


class SyntheticCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = load_default_synthetic_workloads()

    def test_default_catalog_covers_all_three_task_classes_and_named_modules(self):
        self.assertEqual(
            {workload.task_class for workload in self.catalog},
            set(TaskClass),
        )
        required = {
            "obstacle_avoidance",
            "emergency_stop",
            "local_control",
            "localization",
            "environment_understanding",
            "object_detection",
            "semantic_segmentation",
            "local_planning",
            "local_llm_7b",
            "local_llm_10b",
            "map_fusion",
        }
        self.assertTrue(required.issubset({item.task_type for item in self.catalog}))

    def test_every_workload_has_complete_orin_and_edge_profiles(self):
        for workload in self.catalog:
            self.assertGreaterEqual(workload.accelerator_demand_tops, 0)
            for target in ExecutionTarget:
                profile = workload.profile_for(target)
                self.assertGreater(profile.latency.p99_ms, 0)
                self.assertGreater(profile.resources.memory_mb, 0)
                self.assertGreaterEqual(profile.input_size_mb.typical, 0)
                self.assertGreaterEqual(profile.output_size_mb.typical, 0)
                self.assertGreaterEqual(profile.energy_j.typical, 0)
                self.assertGreaterEqual(profile.failure_rate, 0)
                self.assertLessEqual(profile.accuracy.maximum, 1)
                self.assertGreaterEqual(profile.max_concurrency, 1)

    def test_local_safety_is_explicitly_unsupported_at_edge(self):
        for workload in self.catalog.by_class(TaskClass.LOCAL_SAFETY):
            self.assertFalse(workload.profile_for(ExecutionTarget.EDGE).supported)
            with self.assertRaises(UnsupportedTargetError):
                SyntheticSampler(self.catalog).sample(workload.task_type, ExecutionTarget.EDGE)

    def test_deterministic_mode_returns_typical_values_and_never_injects_failure(self):
        workload = self.catalog.get("object_detection")
        profile = workload.profile_for(ExecutionTarget.ORIN)
        sampler = SyntheticSampler(self.catalog, seed=41, deterministic=True)
        first = sampler.sample(workload.task_type, ExecutionTarget.ORIN)
        second = sampler.sample(workload.task_type, ExecutionTarget.ORIN)
        self.assertEqual(first, second)
        self.assertEqual(first.latency_ms, profile.latency.p50_ms)
        self.assertEqual(first.input_size_mb, profile.input_size_mb.typical)
        self.assertEqual(first.output_size_mb, profile.output_size_mb.typical)
        self.assertEqual(first.energy_j, profile.energy_j.typical)
        self.assertEqual(first.accuracy, profile.accuracy.typical)
        self.assertFalse(first.failed)

    def test_seeded_sampling_is_reproducible_and_bounded(self):
        left = SyntheticSampler(self.catalog, seed=177)
        right = SyntheticSampler(self.catalog, seed=177)
        left_samples = [left.sample("map_fusion", "edge") for _ in range(8)]
        right_samples = [right.sample("map_fusion", "edge") for _ in range(8)]
        self.assertEqual(left_samples, right_samples)
        profile = self.catalog.get("map_fusion").profile_for("edge")
        for sample in left_samples:
            self.assertGreater(sample.latency_ms, 0)
            self.assertGreaterEqual(sample.input_size_mb, profile.input_size_mb.minimum)
            self.assertLessEqual(sample.input_size_mb, profile.input_size_mb.maximum)
            self.assertGreaterEqual(sample.accuracy, profile.accuracy.minimum)
            self.assertLessEqual(sample.accuracy, profile.accuracy.maximum)

    def test_custom_workload_can_be_registered_from_one_dictionary(self):
        catalog = SyntheticWorkloadCatalog()
        workload = catalog.register_dict(custom_workload())
        self.assertIs(catalog.get("custom_inspector"), workload)
        component = catalog.component("custom_inspector", "orin", seed=9, deterministic=True)
        self.assertEqual(component.max_concurrency, 3)
        self.assertTrue(component.can_accept(2))
        self.assertFalse(component.can_accept(3))
        self.assertEqual(component.execute_sample().task_type, "custom_inspector")
        with self.assertRaisesRegex(ValueError, "already registered"):
            catalog.register_dict(custom_workload())

    def test_workload_can_create_scheduler_task_spec(self):
        workload = self.catalog.get("local_llm_7b")
        spec = workload.to_task_spec("edge", allow_local_fallback=False)
        profile = workload.profile_for("edge")
        self.assertEqual(spec.task_type, "local_llm_7b")
        self.assertIs(spec.task_class, TaskClass.EDGE_HEAVY)
        self.assertEqual(spec.input_size_mb, profile.input_size_mb.typical)
        self.assertEqual(spec.output_size_mb, profile.output_size_mb.typical)
        self.assertEqual(spec.gpu_demand, workload.accelerator_demand_tops)
        self.assertFalse(spec.allow_local_fallback)
        self.assertEqual(
            [(port.name, port.message_type) for port in spec.input_ports],
            [("context", "text_context")],
        )
        self.assertEqual(
            [(port.name, port.message_type) for port in spec.output_ports],
            [("response", "structured_text")],
        )


if __name__ == "__main__":
    unittest.main()
