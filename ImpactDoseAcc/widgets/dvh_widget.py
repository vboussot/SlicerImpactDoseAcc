import logging
from uuid import uuid4

import numpy as np
import slicer
import vtk
from qt import (
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QWidget,
    QLineEdit,
    QMessageBox,
)

logger = logging.getLogger(__name__)


class DVHWidget(QWidget):
    """UI widget for Phase 4: DVH.

    Placeholder/skeleton only.

    Goal (later): compute DVH curves per segment for a dose volume (and optionally uncertainty).
    Output format (later): MRML Table(s) with columns: dose_bin, volume_cc / volume_% per segment.

    This file is intentionally not wired into the main module UI yet.
    """

    def __init__(self, logic):
        super().__init__()
        self.logic = logic
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()

        info = QLabel(
            "Phase 4: DVH\n\n"
            "Not implemented yet.\n"
            "Planned: DVH per structure + export."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        inputs_group = QGroupBox("Inputs (planned)")
        inputs_layout = QVBoxLayout()

        dose_row = QHBoxLayout()
        dose_row.addWidget(QLabel("Dose:"))
        self.dose_selector = slicer.qMRMLNodeComboBox()
        self.dose_selector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.dose_selector.noneEnabled = True
        self.dose_selector.addEnabled = False
        self.dose_selector.removeEnabled = False
        self.dose_selector.setMRMLScene(slicer.mrmlScene)
        dose_row.addWidget(self.dose_selector, 1)
        inputs_layout.addLayout(dose_row)

        seg_row = QHBoxLayout()
        seg_row.addWidget(QLabel("Segmentation:"))
        self.seg_selector = slicer.qMRMLNodeComboBox()
        self.seg_selector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.seg_selector.noneEnabled = True
        self.seg_selector.addEnabled = False
        self.seg_selector.removeEnabled = False
        self.seg_selector.setMRMLScene(slicer.mrmlScene)
        seg_row.addWidget(self.seg_selector, 1)
        inputs_layout.addLayout(seg_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output base name:"))
        self.output_name_edit = QLineEdit()
        self.output_name_edit.setText(f"dvh_{uuid4().hex[:6]}")
        out_row.addWidget(self.output_name_edit, 1)
        inputs_layout.addLayout(out_row)

        inputs_group.setLayout(inputs_layout)
        layout.addWidget(inputs_group)

        self.run_btn = QPushButton("Compute DVH (TODO)")
        self.run_btn.clicked.connect(self._on_compute_dvh)
        layout.addWidget(self.run_btn)

        layout.addStretch()
        self.setLayout(layout)

    def _on_compute_dvh(self) -> None:
        QMessageBox.information(
            self,
            "Phase 4: DVH",
            "DVH computation is not implemented yet.\n\n"
            "When we start Phase 4, we will:"
            "\n- choose binning (Gy / cGy, bin width)"
            "\n- compute DVH per segment"
            "\n- export as TableNode(s)"
            "\n- (optional) add uncertainty-aware DVH"
        )


def compute_dvh_placeholder(dose_array: np.ndarray, mask_array: np.ndarray, bins_gy: np.ndarray):
    """Placeholder DVH core (not used yet).

    Intended later behavior:
    - dose_array: float dose in Gy on same grid as mask_array
    - mask_array: boolean mask for one ROI
    - bins_gy: 1D bin edges in Gy

    Return cumulative DVH as volume fraction per bin.
    """
    if dose_array is None or mask_array is None or bins_gy is None:
        raise ValueError("Missing inputs")
    raise NotImplementedError("DVH core not implemented yet")
