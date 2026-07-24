import mars.domain as domain
import mars.models as legacy_models
from mars.domain.artifact import ArtifactRef
from mars.domain.execution import Assignment
from mars.domain.task import TaskSpec
from mars.domain.topology import NodeSpec
from mars.domain.transfer import TransferEstimate
from mars.domain.workflow import WorkflowSpec


def test_legacy_models_module_reexports_domain_objects() -> None:
    assert legacy_models.__all__ == domain.__all__
    for name in domain.__all__:
        assert getattr(legacy_models, name) is getattr(domain, name)


def test_domain_objects_are_owned_by_focused_modules() -> None:
    assert ArtifactRef.__module__ == "mars.domain.artifact"
    assert Assignment.__module__ == "mars.domain.execution"
    assert TaskSpec.__module__ == "mars.domain.task"
    assert NodeSpec.__module__ == "mars.domain.topology"
    assert TransferEstimate.__module__ == "mars.domain.transfer"
    assert WorkflowSpec.__module__ == "mars.domain.workflow"
