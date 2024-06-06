
import os
from typing import Any, Dict, List, Optional

from .artifact_data import ArtifactInfo
from .attribute_type import LoggingItemKind
from .parameter_data import ParameterInfo
from .metric_data import MetricInfo
from ..provenance.context import Context
from ..utils import funcs

class Prov4MLData:
    """
    Holds the provenance data for metrics, parameters, and artifacts in a machine learning experiment.

    Attributes:
        metrics (Dict[str, MetricInfo]): A dictionary of metrics.
        parameters (Dict[str, ParameterInfo]): A dictionary of parameters.
        artifacts (Dict[str, ArtifactInfo]): A dictionary of artifacts.
        experiment_name (str): The name of the experiment.
    """
    def __init__(self) -> None:
        self.metrics: Dict[(str, Context), MetricInfo] = {}
        self.parameters: Dict[str, ParameterInfo] = {}
        self.artifacts: Dict[(str, Context), ArtifactInfo] = {}

        self.PROV_SAVE_PATH = "prov_save_path"
        self.EXPERIMENT_NAME = "test_experiment"
        self.EXPERIMENT_DIR = "test_experiment_dir"
        self.ARTIFACTS_DIR = "artifact_dir"
        self.USER_NAMESPACE = "user_namespace"
        self.RUN_ID = 0

        self.global_rank = None
        self.is_collecting = False

    def init(self, experiment_name, prov_save_path=None, user_namespace=None, collect_all_processes=False): 
        
        self.global_rank = os.getenv("SLURM_PROCID", None)
        self.EXPERIMENT_NAME = experiment_name + f"_GR{self.global_rank}" if self.global_rank else experiment_name
        self.is_collecting = self.global_rank is None or self.global_rank == 0 or collect_all_processes
        
        if not self.is_collecting: return

        if prov_save_path: 
            self.PROV_SAVE_PATH = prov_save_path
        if user_namespace:
            self.USER_NAMESPACE = user_namespace

        # look at PROV dir how many experiments are there with the same name
        if not os.path.exists(self.PROV_SAVE_PATH):
            os.makedirs(self.PROV_SAVE_PATH, exist_ok=True)
        prev_exps = os.listdir(self.PROV_SAVE_PATH) if self.PROV_SAVE_PATH else []
        run_id = len([exp for exp in prev_exps if funcs.prov4ml_experiment_matches(experiment_name, exp)]) 

        self.EXPERIMENT_DIR = os.path.join(self.PROV_SAVE_PATH, experiment_name + f"_{run_id}")
        self.RUN_ID = run_id
        self.ARTIFACTS_DIR = os.path.join(self.EXPERIMENT_DIR, "artifacts")


    def add_metric(self, metric: str, value: Any, step: int, context: Optional[Any] = None, source:LoggingItemKind=None) -> None:
        """
        Adds a metric to the metrics dictionary.

        Parameters:
            metric (str): The name of the metric.
            value (Any): The value of the metric.
            step (int): The step number for the metric.
            context (Optional[Any]): The context of the metric. Defaults to None.
        """
        if not self.is_collecting: return

        if (metric, context) not in self.metrics:
            self.metrics[(metric, context)] = MetricInfo(metric, context, source=source)
        self.metrics[(metric, context)].add_metric(value, step)

    def add_parameter(self, parameter: str, value: Any) -> None:
        """
        Adds a parameter to the parameters dictionary.

        Parameters:
            parameter (str): The name of the parameter.
            value (Any): The value of the parameter.
        """
        if not self.is_collecting: return

        self.parameters[parameter] = ParameterInfo(parameter, value)

    def add_artifact(
        self, 
        artifact_name: str, 
        value: Any = None, 
        step: Optional[int] = None, 
        context: Optional[Any] = None, 
        timestamp: Optional[int] = None
    ) -> None:
        """
        Adds an artifact to the artifacts dictionary.

        Parameters:
            artifact_name (str): The name of the artifact.
            value (Any): The value of the artifact. Defaults to None.
            step (Optional[int]): The step number for the artifact. Defaults to None.
            context (Optional[Any]): The context of the artifact. Defaults to None.
            timestamp (Optional[int]): The timestamp of the artifact. Defaults to None.
        """
        if not self.is_collecting: return

        self.artifacts[(artifact_name, context)] = ArtifactInfo(artifact_name, value, step, context=context, timestamp=timestamp)

    def get_artifacts(self) -> List[ArtifactInfo]:
        """
        Returns a list of all artifacts.

        Returns:
            List[ArtifactInfo]: A list of artifact information objects.
        """
        if not self.is_collecting: return

        return list(self.artifacts.values())
    
    def get_model_versions(self) -> List[ArtifactInfo]:
        """
        Returns a list of all model version artifacts.

        Returns:
            List[ArtifactInfo]: A list of model version artifact information objects.
        """
        if not self.is_collecting: return

        return [artifact for artifact in self.artifacts.values() if artifact.is_model_version]
    
    def get_final_model(self) -> Optional[ArtifactInfo]:
        """
        Returns the most recent model version artifact.

        Returns:
            Optional[ArtifactInfo]: The most recent model version artifact information object, or None if no model versions exist.
        """
        if not self.is_collecting: return

        model_versions = self.get_model_versions()
        if model_versions:
            return model_versions[-1]
        return None
