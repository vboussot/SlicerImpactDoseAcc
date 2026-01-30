import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import slicer
from qt import QTabWidget, QWidget
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)
from widgets.accumulation_widget import DoseAccumulationWidget
from widgets.dvh_widget import DVHWidget
from widgets.metrics_widget import MetricsEvaluationWidget
from widgets.prescription_widget import PrescriptionDoseEstimationWidget

logger = logging.getLogger(__name__)


def compute_statistics_from_arrays(arrays: Iterable[np.ndarray], *, dtype=np.float32, mask: np.ndarray | None = None):
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
    m2 = None
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
            m2 = np.zeros_like(mean, dtype=np.float64)
            mn = np.array(arr, dtype=dtype, copy=True)
            mx = np.array(arr, dtype=dtype, copy=True)
            n = 1
            continue

        n += 1
        # delta correction uses float64 internally for stability
        delta = arr - mean
        mean = mean + delta / n
        m2 = m2 + (delta * (arr - mean))
        mn = np.minimum(mn, arr)
        mx = np.maximum(mx, arr)

    if n == 0 or m2 is None:
        return {"mean": None, "std": None, "min": None, "max": None, "n_samples": 0}

    # Population variance (ddof=0) to match previous behaviour (np.std default)
    var = (m2 / n) if n > 0 else np.zeros_like(m2)
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
        warped_volume_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode", dose_volume_node.GetName() + "_warped"
        )
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
                return {
                    **compute_statistics_from_arrays(dose_volumes, dtype=dtype, mask=mask),
                    "n_samples": len(dose_volumes),
                }

            arrays = []
            for n in dose_volumes or []:
                try:
                    arr = slicer.util.arrayFromVolume(n)
                except Exception:
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

    def save_dose_to_dicom(
        self, dose_volumes, output_path: str, series_number: int = 1, metadata: dict | None = None
    ) -> str:
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
