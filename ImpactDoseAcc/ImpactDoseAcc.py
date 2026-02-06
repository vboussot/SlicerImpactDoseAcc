import logging
import sys
from pathlib import Path

from qt import QTabWidget, QWidget
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleWidget
from widgets.accumulation_widget import DoseAccumulationWidget
from widgets.dvh_widget import DVHWidget
from widgets.metrics_widget import MetricsEvaluationWidget
from widgets.prescription_widget import PrescriptionDoseEstimationWidget

logger = logging.getLogger(__name__)


class _PluginPathFilter(logging.Filter):
    def __init__(self, root_path: Path):
        super().__init__()
        self._root = str(root_path).replace("\\", "/")

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            path = (record.pathname or "").replace("\\", "/")
            return path.startswith(self._root)
        except Exception:
            return False


class _ImpactDoseAccHandler(logging.StreamHandler):
    """Marker handler used to avoid installing duplicates."""


def configure_plugin_logging(level: int = logging.INFO) -> None:
    """Attach a local handler that only emits logs from this plugin's files."""
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, _ImpactDoseAccHandler):
            return
    handler = _ImpactDoseAccHandler(sys.stdout)
    handler.setLevel(level)
    handler.addFilter(_PluginPathFilter(Path(__file__).resolve().parent))
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    root.addHandler(handler)


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


class ImpactDoseAccWidget(ScriptedLoadableModuleWidget):
    """Main UI widget; loads prescription, accumulation, and metrics widgets."""

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)

    def setup(self) -> None:
        super().setup()
        configure_plugin_logging()
        self.tabWidget = QTabWidget()
        self.tabWidget.setMaximumWidth(800)
        self.tab_prescription = PrescriptionDoseEstimationWidget()
        self.tab_accumulation = DoseAccumulationWidget()
        self.tab_metrics = MetricsEvaluationWidget()
        self.tab_dvh = DVHWidget()
        self.tabWidget.addTab(self.tab_prescription, _("Delivered Dose"))
        self.tabWidget.addTab(self.tab_accumulation, _("Dose Accumulation"))
        self.tabWidget.addTab(self.tab_metrics, _("Metrics/QA"))
        self.tabWidget.addTab(self.tab_dvh, _("DVH"))
        self.layout.addWidget(self.tabWidget)
