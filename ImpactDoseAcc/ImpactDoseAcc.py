import logging
from pathlib import Path
from typing import Iterable, Optional
import numpy as np
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleLogic, ScriptedLoadableModuleWidget
logger = logging.getLogger(__name__)

from qt import QTabWidget,QWidget

SERVICE = "ImpactDoseAcc"


try:
    # Normal package import within Slicer
    from .widgets.prescription_widget import PrescriptionDoseEstimationWidget
    from .widgets.accumulation_widget import DoseAccumulationWidget
    from .widgets.metrics_widget import MetricsEvaluationWidget
    from .widgets.dvh_widget import DVHWidget
except Exception:
    # Fallback when executed without package context: import modules by file path
    import importlib.util
    base_dir = Path(__file__).resolve().parent

    def _load_from_file(module_name: str, file_path: Path):
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        return module

    pres_path = base_dir / "widgets" / "prescription_widget.py"
    acc_path = base_dir / "widgets" / "accumulation_widget.py"
    metrics_path = base_dir / "widgets" / "metrics_widget.py"
    dvh_path = base_dir / "widgets" / "dvh_widget.py"

    pres_mod = _load_from_file("impactdoseacc_prescription_widget", pres_path)
    acc_mod = _load_from_file("impactdoseacc_accumulation_widget", acc_path)
    metrics_mod = _load_from_file("impactdoseacc_metrics_widget", metrics_path)
    dvh_mod = _load_from_file("impactdoseacc_dvh_widget", dvh_path)

    PrescriptionDoseEstimationWidget = getattr(pres_mod, "PrescriptionDoseEstimationWidget")
    DoseAccumulationWidget = getattr(acc_mod, "DoseAccumulationWidget")
    MetricsEvaluationWidget = getattr(metrics_mod, "MetricsEvaluationWidget")
    DVHWidget = getattr(dvh_mod, "DVHWidget")



def compute_statistics_from_arrays(arrays: Iterable[np.ndarray], *, dtype=np.float32, mask: Optional[np.ndarray] = None):
    """Compute mean, std (population), min and max over a sequence of arrays using an
    incremental Welford-style algorithm to avoid stacking all arrays in memory.

    Parameters
    - arrays: iterable of numpy arrays (all must have same shape)
    - dtype: numpy dtype to use for outputs (default float32)
    - mask: optional boolean array with the same shape as arrays to restrict computation to ROI

    Returns a dict with keys: mean, std, min, max (all numpy arrays with dtype) and n_samples.
    """
    arrays = list(arrays or [])
    if not arrays:
        return {"mean": None, "std": None, "min": None, "max": None, "n_samples": 0}

    # Validate shapes
    first = np.asarray(arrays[0])
    shape = first.shape
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != shape:
            raise ValueError("mask shape must match input arrays")

    mean = None
    M2 = None
    mn = None
    mx = None
    n = 0

    for a in arrays:
        arr = np.asarray(a)
        if arr.shape != shape:
            raise ValueError("All arrays must have the same shape")
        if mask is not None:
            arr = arr[mask]
        arr = arr.astype(dtype, copy=False)

        if mean is None:
            # Initialize
            mean = np.array(arr, dtype=dtype, copy=True)
            M2 = np.zeros_like(mean, dtype=np.float64)
            mn = np.array(arr, dtype=dtype, copy=True)
            mx = np.array(arr, dtype=dtype, copy=True)
            n = 1
            continue

        n += 1
        # delta correction uses float64 internally for stability
        delta = arr - mean
        mean = mean + delta / n
        M2 = M2 + (delta * (arr - mean))
        mn = np.minimum(mn, arr)
        mx = np.maximum(mx, arr)

    if n == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "n_samples": 0}

    # Population variance (ddof=0) to match previous behaviour (np.std default)
    var = (M2 / n) if n > 0 else np.zeros_like(M2)
    std = np.sqrt(var)

    return {
        "mean": np.asarray(mean, dtype=dtype),
        "std": np.asarray(std, dtype=dtype),
        "min": np.asarray(mn, dtype=dtype),
        "max": np.asarray(mx, dtype=dtype),
        "n_samples": n,
    }


class ImpactDoseAcc(ScriptedLoadableModule):
    """3D Slicer module for dose accumulation with uncertainty quantification."""

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("IMPACT-DoseAcc")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Radiotherapy")]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Cédric Hémon (University of Rennes, France)",
            "Valentin Boussot (University of Rennes, France)",
            "Jean-Louis Dillenseger (University of Rennes, France)",
        ]
        self.parent.helpText = _(
            """
            <p>
            <b>IMPACT-DoseAcc: End-to-end Dose Accumulation with Uncertainty Quantification</b>
            </p>
            <p>
            This module enables:<br>
            &bull; Estimation of prescribed dose from multiple synthetic CTs and DVFs<br>
            &bull; Dose accumulation across fractions with uncertainty quantification<br>
            &bull; Classical and uncertainty-aware metrics
            </p>
            """
        )

        self.parent.acknowledgementText = _(
            """
            <p>
            This module was originally developed by Valentin Boussot and Cédric Hémon (University of Rennes, France).
            </p>
            """
        )

class ImpactDoseAccLogic(ScriptedLoadableModuleLogic):
    """Computation logic for dose deformation and statistics."""

    def __init__(self):
        super().__init__()
        self._work_dir = Path(slicer.app.temporaryPath) / "ImpactDoseAcc"
        self._work_dir.mkdir(parents=True, exist_ok=True)

    def deform_dose_by_dvf(self, dose_volume_node, dvf_transform_node, reference_volume_node=None):
        if reference_volume_node is None:
            reference_volume_node = dose_volume_node
        warped_volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", dose_volume_node.GetName() + "_warped")
        params = {
            "inputVolume": dose_volume_node.GetID(),
            "referenceVolume": reference_volume_node.GetID(),
            "outputVolume": warped_volume_node.GetID(),
            "interpolationType": "linear",
            "transformationFile": dvf_transform_node.GetID(),
        }
        slicer.cli.runSync(slicer.modules.resamplescalarvectordwivolume, None, params)
        return warped_volume_node

    def safe_remove_node(self, node) -> None:
        """Safely remove a node from the current scene if present."""
        try:
            if node is not None and getattr(node, "GetScene", lambda: None)() == slicer.mrmlScene:
                slicer.mrmlScene.RemoveNode(node)
        except Exception:
            logger.exception("safe_remove_node failed")
            
    def compute_dose_statistics(self, dose_volumes: list, *, dtype=np.float32, mask=None):
        """
        Compute per-voxel statistics (mean, std, min, max) using an incremental Welford algorithm
        implemented in ImpactDoseAcc.utils.compute_statistics_from_arrays. Accepts either a list of
        numpy arrays or a list of MRML volume nodes. dtype defaults to float32 for memory efficiency.
        """
        try:
            # If the input already contains numpy arrays, forward directly to the util.
            if dose_volumes and isinstance(dose_volumes[0], (np.ndarray,)):
                return {**compute_statistics_from_arrays(dose_volumes, dtype=dtype, mask=mask),
                    "n_samples": len(dose_volumes)}

            arrays = []
            for n in dose_volumes or []:
                try:
                    arr = slicer.util.arrayFromVolume(n)
                except Exception as e:
                    logger.exception("Failed to extract array from volume node")
                    raise
                if arr is None:
                    raise RuntimeError("Failed to read voxel array from volume node")
                arrays.append(arr.astype(dtype, copy=False))

            stats = compute_statistics_from_arrays(arrays, dtype=dtype, mask=mask)
            stats["n_samples"] = len(arrays)
            return stats
        except Exception:
            logger.exception("compute_dose_statistics failed")
            raise
    def save_dose_to_dicom(self, dose_volumes, output_path: str, series_number: int = 1, metadata: dict = None) -> str:
        logger.warning("save_dose_to_dicom is not implemented; implement with pydicom/dcmqi")
        raise NotImplementedError("Not implemented")

    def load_dose_from_dicom(self, dicom_path: str):
        logger.warning("load_dose_from_dicom is not implemented; implement with pydicom/dcmqi")
        raise NotImplementedError("Not implemented")


class ImpactDoseAccWidget(ScriptedLoadableModuleWidget):
    """Main UI widget; loads prescription, accumulation, and metrics widgets."""

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.logic = ImpactDoseAccLogic()

    def setup(self) -> None:
        super().setup()
        self.tabWidget = QTabWidget()
        self.tabWidget.setMaximumWidth(800)
        self.tab_prescription = PrescriptionDoseEstimationWidget(self.logic)
        self.tab_accumulation = DoseAccumulationWidget(self.logic)
        self.tab_metrics = MetricsEvaluationWidget(self.logic)
        self.tab_dvh = DVHWidget(self.logic)
        self.tabWidget.addTab(self.tab_prescription, _("Delivered Dose"))
        self.tabWidget.addTab(self.tab_accumulation, _("Dose Accumulation"))
        self.tabWidget.addTab(self.tab_metrics, _("Metrics/QA"))
        self.tabWidget.addTab(self.tab_dvh, _("DVH"))
        self.layout.addWidget(self.tabWidget)

    def cleanup(self) -> None:
        pass

    def enter(self) -> None:
        pass

    def exit(self) -> None:
        pass

