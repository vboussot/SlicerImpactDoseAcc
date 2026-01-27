import logging
from pathlib import Path

import numpy as np
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)

logger = logging.getLogger(__name__)

from qt import (
    QTabWidget,
    QWidget,
    QLabel,
    QMessageBox,
    QFileDialog,
    QVBoxLayout,
    QSizePolicy,
)

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

    def compute_dose_statistics(self, dose_volumes: list):
        dose_arrays = [slicer.util.arrayFromVolume(n) for n in dose_volumes]
        dose_stack = np.array(dose_arrays)
        return {
            "mean": np.mean(dose_stack, axis=0),
            "std": np.std(dose_stack, axis=0),
            "min": np.min(dose_stack, axis=0),
            "max": np.max(dose_stack, axis=0),
            "n_samples": len(dose_arrays),
        }

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

