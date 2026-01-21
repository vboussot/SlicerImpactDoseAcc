"""
Widget package for IMPACT-DoseAcc: separate UI widget modules.
"""

from .prescription_widget import PrescriptionDoseEstimationWidget
from .accumulation_widget import DoseAccumulationWidget
from .metrics_widget import MetricsEvaluationWidget

__all__ = [
    "PrescriptionDoseEstimationWidget",
    "DoseAccumulationWidget",
    "MetricsEvaluationWidget",
]
