import logging
import threading
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
    QCheckBox,
    QComboBox,
    QScrollArea,
    QLineEdit,
    QMessageBox,
    QDoubleSpinBox,
    QProgressBar,
    QTimer,
)

logger = logging.getLogger(__name__)

class MetricsEvaluationWidget(QWidget):
    """UI widget for Phase 3: Metrics & Evaluation."""

    def __init__(self, logic):
        super().__init__()
        self.logic = logic
        self._ref_dose_node_by_index = {}
        self._out_dose_node_by_index = {}
        self._unc_node_by_index = {}
        self._segment_checkbox_by_id = {}
        self._active_job = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()

        inputs_group = QGroupBox("1. Inputs")
        inputs_layout = QVBoxLayout()

        header = QHBoxLayout()
        header.addWidget(QLabel("Select inputs for Phase 3 metrics:"))
        header.addStretch()
        refresh_btn = QPushButton("⟳")
        refresh_btn.setMaximumWidth(40)
        refresh_btn.setToolTip("Refresh lists")
        refresh_btn.clicked.connect(self._refresh_lists)
        header.addWidget(refresh_btn)
        inputs_layout.addLayout(header)

        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Reference dose:"))
        self.ref_dose_combo = QComboBox()
        ref_row.addWidget(self.ref_dose_combo, 1)
        inputs_layout.addLayout(ref_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output dose:"))
        self.out_dose_combo = QComboBox()
        out_row.addWidget(self.out_dose_combo, 1)
        inputs_layout.addLayout(out_row)

        unc_row = QHBoxLayout()
        unc_row.addWidget(QLabel("Output uncertainty (optional):"))
        self.unc_combo = QComboBox()
        unc_row.addWidget(self.unc_combo, 1)
        inputs_layout.addLayout(unc_row)

        seg_row = QHBoxLayout()
        seg_row.addWidget(QLabel("Segmentation (structures):"))
        self.seg_selector = slicer.qMRMLNodeComboBox()
        self.seg_selector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.seg_selector.selectNodeUponCreation = False
        self.seg_selector.addEnabled = False
        self.seg_selector.removeEnabled = False
        self.seg_selector.noneEnabled = True
        self.seg_selector.showHidden = False
        self.seg_selector.setMRMLScene(slicer.mrmlScene)
        seg_row.addWidget(self.seg_selector, 1)
        inputs_layout.addLayout(seg_row)

        # Segment selection (enabled once segmentation is chosen)
        segments_group = QGroupBox("Segments to include")
        segments_group_layout = QVBoxLayout()
        self._segments_scroll = QScrollArea()
        self._segments_scroll.setWidgetResizable(True)
        self._segments_scroll_content = QWidget()
        self._segments_scroll_layout = QVBoxLayout()
        self._segments_scroll_content.setLayout(self._segments_scroll_layout)
        self._segments_scroll.setWidget(self._segments_scroll_content)
        segments_group_layout.addWidget(self._segments_scroll)
        segments_group.setLayout(segments_group_layout)
        inputs_layout.addWidget(segments_group)
        self._segments_group = segments_group

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output table name:"))
        self.output_name_edit = QLineEdit()
        self.output_name_edit.setText(self._generate_default_output_name())
        out_row.addWidget(self.output_name_edit, 1)
        inputs_layout.addLayout(out_row)

        inputs_group.setLayout(inputs_layout)
        layout.addWidget(inputs_group)

        metrics_group = QGroupBox("2. Metrics list")
        metrics_layout = QVBoxLayout()
        self.cb_dose_mean = QCheckBox("Output dose mean")
        self.cb_dose_minmax_3sigma = QCheckBox("Output dose min/max (mean ± 3*unc)")
        self.cb_err_mae = QCheckBox("MAE (|Output - Reference|)")
        self.cb_unc_mean = QCheckBox("Mean dose uncertainty")
        self.cb_gamma_pr = QCheckBox("Gamma pass rate (%)")

        for cb in (
            self.cb_dose_mean,
            self.cb_dose_minmax_3sigma,
            self.cb_err_mae,
            self.cb_unc_mean,
            self.cb_gamma_pr,
        ):
            cb.setChecked(True)
            metrics_layout.addWidget(cb)

        # Gamma can be expensive: keep it off by default.
        try:
            self.cb_gamma_pr.setChecked(False)
        except Exception:
            pass

        # Gamma parameters (shown when gamma metric is enabled)
        gamma_params = QWidget()
        gamma_params_layout = QVBoxLayout()
        gamma_params_layout.setContentsMargins(20, 0, 0, 0)

        row1 = QHBoxLayout()

        w_label_dose_diff = QLabel("Dose diff (%):")
        w_spin_dose_diff = QDoubleSpinBox()
        w_spin_dose_diff.setDecimals(1)
        w_spin_dose_diff.setSingleStep(0.5)
        w_spin_dose_diff.setRange(0.0, 5.0)
        w_spin_dose_diff.setValue(2.0)
        w_spin_dose_diff.setToolTip("Dose difference threshold as percentage of reference dose global max")

        # Keep references (PythonQt can GC locals even when added to layouts)
        self.gamma_dose_percent_edit = w_spin_dose_diff
        self._gamma_label_dose_diff = w_label_dose_diff

        row1.addWidget(w_label_dose_diff)
        row1.addWidget(w_spin_dose_diff)
        
        row1.addSpacing(12)

        w_label_dta = QLabel("DTA (mm):")
        w_spin_dta = QDoubleSpinBox()
        w_spin_dta.setDecimals(1)
        w_spin_dta.setSingleStep(0.5)
        w_spin_dta.setRange(0.0, 5.0)
        w_spin_dta.setValue(2.0)
        w_spin_dta.setToolTip("Distance to agreement threshold in millimeters")

        self.gamma_dist_mm_edit = w_spin_dta
        self._gamma_label_dta = w_label_dta
        
        row1.addWidget(w_label_dta)
        row1.addWidget(w_spin_dta)

        row1.addStretch(1)

        row2 = QHBoxLayout()

        w_label_cutoff = QLabel("Low-dose cutoff (% of ref global max):")
        w_spin_cutoff = QDoubleSpinBox()
        w_spin_cutoff.setDecimals(0)
        w_spin_cutoff.setSingleStep(10)
        w_spin_cutoff.setRange(0, 50)
        w_spin_cutoff.setValue(30)
        w_spin_cutoff.setToolTip("Lower dose cutoff as percentage of reference dose global max")

        self.gamma_low_cutoff_edit = w_spin_cutoff
        self._gamma_label_cutoff = w_label_cutoff
        
        row2.addWidget(w_label_cutoff)
        row2.addWidget(w_spin_cutoff)

        row2.addSpacing(12)
        row2.addWidget(QLabel("Mode:"))
        self.gamma_mode_combo = QComboBox()
        self.gamma_mode_combo.addItems(["Global", "Local"])
        self.gamma_mode_combo.setMaximumWidth(100)
        row2.addWidget(self.gamma_mode_combo)
        row2.addStretch(1)
        gamma_params_layout.addLayout(row1)
        gamma_params_layout.addLayout(row2)

        gamma_params.setLayout(gamma_params_layout)
        metrics_layout.addWidget(gamma_params)
        self._gamma_params_widget = gamma_params

        # Keep layout references to avoid row disappearing issues
        self._gamma_params_layout = gamma_params_layout
        self._gamma_row1_layout = row1
        self._gamma_row2_layout = row2

        def _sync_gamma_params_visibility():
            try:
                self._gamma_params_widget.setVisible(bool(self.cb_gamma_pr.isChecked()))
            except Exception:
                pass

        try:
            self.cb_gamma_pr.toggled.connect(_sync_gamma_params_visibility)
        except Exception:
            pass
        _sync_gamma_params_visibility()

        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)

        self.run_btn = QPushButton("Compute metrics")
        self.run_btn.clicked.connect(self._on_compute_metrics)
        layout.addWidget(self.run_btn)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addStretch()
        self.setLayout(layout)

        try:
            self.seg_selector.currentNodeChanged.connect(self._on_segmentation_changed)
        except Exception:
            pass

        self._refresh_lists()
        self._on_segmentation_changed(self.seg_selector.currentNode())

    def _clear_layout(self, layout) -> None:
        if layout is None:
            return
        try:
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget() if item is not None else None
                if w is not None:
                    w.setParent(None)
        except Exception:
            pass

    def _on_segmentation_changed(self, seg_node) -> None:
        self._segment_checkbox_by_id = {}
        self._clear_layout(self._segments_scroll_layout)

        if seg_node is None:
            try:
                self._segments_group.setEnabled(False)
            except Exception:
                pass
            return

        try:
            seg = seg_node.GetSegmentation()
        except Exception:
            seg = None

        if seg is None:
            try:
                self._segments_group.setEnabled(False)
            except Exception:
                pass
            return

        try:
            self._segments_group.setEnabled(True)
        except Exception:
            pass

        try:
            n = seg.GetNumberOfSegments()
        except Exception:
            n = 0

        for i in range(n):
            try:
                seg_id = seg.GetNthSegmentID(i)
                seg_obj = seg.GetSegment(seg_id)
                seg_name = seg_obj.GetName() if seg_obj is not None else seg_id
            except Exception:
                continue
            cb = QCheckBox(str(seg_name))
            cb.setChecked(True)
            self._segment_checkbox_by_id[seg_id] = cb
            self._segments_scroll_layout.addWidget(cb)

        # spacer
        self._segments_scroll_layout.addStretch()

    def _safe_node_name(self, node) -> str:
        if node is None or not hasattr(node, "GetName"):
            return ""
        try:
            return node.GetName() or ""
        except Exception:
            return ""

    def _is_name_match(self, node, needle: str) -> bool:
        try:
            return str(needle).lower() in self._safe_node_name(node).lower()
        except Exception:
            return False

    def _combo_current_index(self, combo) -> int:
        """PythonQt peut exposer currentIndex comme propriété int ou méthode callable."""
        if combo is None:
            return 0
        idx_attr = getattr(combo, "currentIndex", 0)
        try:
            return int(idx_attr() if callable(idx_attr) else idx_attr)
        except Exception:
            return 0

    def _refresh_lists(self) -> None:
        """Refresh filtered lists for reference/output dose and uncertainty."""
        if slicer.mrmlScene is None:
            return

        # Keep current selections by node ID if possible.
        ref_prev = None
        out_prev = None
        unc_prev = None
        try:
            ref_prev = self._ref_dose_node_by_index.get(self._combo_current_index(self.ref_dose_combo), None)
        except Exception:
            ref_prev = None
        try:
            out_prev = self._out_dose_node_by_index.get(self._combo_current_index(self.out_dose_combo), None)
        except Exception:
            out_prev = None
        try:
            unc_prev = self._unc_node_by_index.get(self._combo_current_index(self.unc_combo), None)
        except Exception:
            unc_prev = None

        def node_id(n):
            try:
                return n.GetID() if n is not None and hasattr(n, "GetID") else None
            except Exception:
                return None

        ref_prev_id = node_id(ref_prev)
        out_prev_id = node_id(out_prev)
        unc_prev_id = node_id(unc_prev)

        self.ref_dose_combo.blockSignals(True)
        self.out_dose_combo.blockSignals(True)
        self.unc_combo.blockSignals(True)

        self.ref_dose_combo.clear()
        self.out_dose_combo.clear()
        self.unc_combo.clear()
        self._ref_dose_node_by_index = {0: None}
        self._out_dose_node_by_index = {0: None}
        self._unc_node_by_index = {0: None}

        self.ref_dose_combo.addItem("[Select reference dose]")
        self.out_dose_combo.addItem("[Select output dose]")
        self.unc_combo.addItem("[None]")

        # Filter volumes by name.
        try:
            volumes = list(slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"))
        except Exception:
            volumes = []

        # Reference dose: any volume with "dose" in the name.
        dose_nodes_ref = [n for n in volumes if n is not None and self._is_name_match(n, "dose")]
        # Output dose: exclude uncertainty volumes that often contain "dose" in their name.
        dose_nodes_out = [
            n
            for n in volumes
            if n is not None and self._is_name_match(n, "dose") and not self._is_name_match(n, "uncertainty")
        ]
        unc_nodes = [n for n in volumes if n is not None and self._is_name_match(n, "uncertainty")]

        dose_nodes_ref.sort(key=lambda n: self._safe_node_name(n).lower())
        dose_nodes_out.sort(key=lambda n: self._safe_node_name(n).lower())
        unc_nodes.sort(key=lambda n: self._safe_node_name(n).lower())

        ref_match_index = 0
        out_match_index = 0
        unc_match_index = 0

        idx = 1
        for n in dose_nodes_ref:
            self.ref_dose_combo.addItem(self._safe_node_name(n))
            self._ref_dose_node_by_index[idx] = n
            if ref_prev_id and node_id(n) == ref_prev_id:
                ref_match_index = idx
            idx += 1

        idx = 1
        for n in dose_nodes_out:
            self.out_dose_combo.addItem(self._safe_node_name(n))
            self._out_dose_node_by_index[idx] = n
            if out_prev_id and node_id(n) == out_prev_id:
                out_match_index = idx
            idx += 1

        idx = 1
        for n in unc_nodes:
            self.unc_combo.addItem(self._safe_node_name(n))
            self._unc_node_by_index[idx] = n
            if unc_prev_id and node_id(n) == unc_prev_id:
                unc_match_index = idx
            idx += 1

        try:
            if ref_match_index:
                self.ref_dose_combo.setCurrentIndex(ref_match_index)
        except Exception:
            pass
        try:
            if out_match_index:
                self.out_dose_combo.setCurrentIndex(out_match_index)
        except Exception:
            pass
        try:
            if unc_match_index:
                self.unc_combo.setCurrentIndex(unc_match_index)
        except Exception:
            pass

        self.ref_dose_combo.blockSignals(False)
        self.out_dose_combo.blockSignals(False)
        self.unc_combo.blockSignals(False)

    def _selected_ref_dose_node(self):
        try:
            return self._ref_dose_node_by_index.get(self._combo_current_index(self.ref_dose_combo), None)
        except Exception:
            return None

    def _selected_out_dose_node(self):
        try:
            return self._out_dose_node_by_index.get(self._combo_current_index(self.out_dose_combo), None)
        except Exception:
            return None

    def _selected_unc_node(self):
        try:
            return self._unc_node_by_index.get(self._combo_current_index(self.unc_combo), None)
        except Exception:
            return None

    def _needs_resample_to_reference(self, input_node, reference_node) -> bool:
        if input_node is None or reference_node is None:
            return False
        in_arr = slicer.util.arrayFromVolume(input_node)
        ref_arr = slicer.util.arrayFromVolume(reference_node)
        if tuple(getattr(in_arr, "shape", ())) != tuple(getattr(ref_arr, "shape", ())):
            return True
        m_in = vtk.vtkMatrix4x4()
        m_ref = vtk.vtkMatrix4x4()
        input_node.GetIJKToRASMatrix(m_in)
        reference_node.GetIJKToRASMatrix(m_ref)
        for r in range(4):
            for c in range(4):
                if abs(m_in.GetElement(r, c) - m_ref.GetElement(r, c)) > 1e-6:
                    return True
        return False

    def _create_temp_volume_from_array(self, reference_node, array, name_prefix: str):
        volumes_logic = slicer.modules.volumes.logic()
        if volumes_logic is None:
            return None
        node = volumes_logic.CloneVolume(reference_node, f"{name_prefix}_{uuid4().hex[:6]}")
        try:
            node.SetHideFromEditors(1)
            node.SetSelectable(0)
            node.SetSaveWithScene(0)
        except Exception:
            pass
        ref_arr = slicer.util.arrayFromVolume(reference_node)
        slicer.util.updateVolumeFromArray(node, np.asarray(array).astype(ref_arr.dtype, copy=False))
        slicer.util.arrayFromVolumeModified(node)
        return node

    def _generate_default_output_name(self) -> str:
        return f"metrics_{uuid4().hex[:6]}"

    def _line_edit_text(self, line_edit) -> str:
        if line_edit is None:
            return ""
        text_attr = getattr(line_edit, "text", "")
        try:
            val = text_attr() if callable(text_attr) else text_attr
            return "" if val is None else str(val)
        except Exception:
            return ""

    def _float_from_line_edit(self, line_edit, default: float) -> float:
        # Supports both QLineEdit-like widgets and QDoubleSpinBox.
        if line_edit is None:
            return float(default)

        try:
            return float(line_edit.value())
        except Exception:
            pass

        text = self._line_edit_text(line_edit).strip()
        if text == "":
            return float(default)
        try:
            return float(text)
        except Exception:
            return float(default)

    def _gamma_mode_is_local(self) -> bool:
        try:
            return str(self.gamma_mode_combo.currentText).lower().startswith("local")
        except Exception:
            return str(self.gamma_mode_combo.currentText()).lower().startswith("local")

    def _create_or_get_table_node(self, name: str):
        if slicer.mrmlScene is None:
            return None
        try:
            node = slicer.mrmlScene.GetFirstNodeByName(name)
        except Exception:
            node = None
        if node is not None and hasattr(node, "IsA") and node.IsA("vtkMRMLTableNode"):
            return node
        try:
            return slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", name)
        except Exception:
            return None

    def _set_status(self, text: str) -> None:
        try:
            self.status_label.setText(text or "")
        except Exception:
            pass

    def _set_progress(self, value, visible: bool = True) -> None:
        try:
            if not hasattr(self, "progress_bar") or self.progress_bar is None:
                return
            if value is None:
                # Indeterminate
                self.progress_bar.setRange(0, 0)
            else:
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(int(max(0, min(100, value))))
            self.progress_bar.setVisible(bool(visible))
        except Exception:
            pass

    def _set_ui_busy(self, busy: bool) -> None:
        try:
            self.run_btn.setEnabled(not bool(busy))
        except Exception:
            pass
        try:
            self.ref_dose_combo.setEnabled(not bool(busy))
            self.out_dose_combo.setEnabled(not bool(busy))
            self.unc_combo.setEnabled(not bool(busy))
        except Exception:
            pass
        try:
            self.seg_selector.setEnabled(not bool(busy))
        except Exception:
            pass
        try:
            self.output_name_edit.setEnabled(not bool(busy))
        except Exception:
            pass

    def _export_segment_mask(self, segmentation_node, segment_id: str, reference_volume_node):
        """Export one segment to a temporary labelmap in reference volume geometry and return a boolean mask."""
        if slicer.mrmlScene is None:
            return None

        # Prefer Slicer utility API when available (avoids native crashes seen with some ExportSegmentsToLabelmapNode calls).
        try:
            fn = getattr(slicer.util, "arrayFromSegmentBinaryLabelmap", None)
            if callable(fn):
                arr = fn(segmentation_node, segment_id, reference_volume_node)
                if arr is None:
                    return None
                return (np.asarray(arr) > 0)
        except Exception:
            # Fall back to labelmap export path below.
            pass

        labelmap = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", f"tmp_seg_{uuid4().hex[:6]}"
        )
        try:
            labelmap.SetHideFromEditors(1)
            labelmap.SetSelectable(0)
            labelmap.SetSaveWithScene(0)
        except Exception:
            pass

        try:
            seg_logic = slicer.modules.segmentations.logic()
            # Use vtkStringArray for maximum compatibility (some builds crash on Python list inputs).
            seg_ids = vtk.vtkStringArray()
            seg_ids.InsertNextValue(str(segment_id))
            seg_logic.ExportSegmentsToLabelmapNode(segmentation_node, seg_ids, labelmap, reference_volume_node)
            arr = slicer.util.arrayFromVolume(labelmap)
            mask = np.asarray(arr) > 0
        except Exception:
            # Fallback: use ExportVisibleSegmentsToLabelmapNode by toggling visibility.
            mask = None
            try:
                disp = segmentation_node.GetDisplayNode()
                if disp is None:
                    segmentation_node.CreateDefaultDisplayNodes()
                    disp = segmentation_node.GetDisplayNode()
                prev_vis = {}
                seg = segmentation_node.GetSegmentation()
                n = seg.GetNumberOfSegments() if seg is not None else 0
                for i in range(n):
                    sid = seg.GetNthSegmentID(i)
                    try:
                        prev_vis[sid] = bool(disp.GetSegmentVisibility(sid))
                        disp.SetSegmentVisibility(sid, sid == segment_id)
                    except Exception:
                        pass
                seg_logic = slicer.modules.segmentations.logic()
                seg_logic.ExportVisibleSegmentsToLabelmapNode(segmentation_node, labelmap, reference_volume_node)
                arr = slicer.util.arrayFromVolume(labelmap)
                mask = np.asarray(arr) > 0
                # restore
                for sid, vis in prev_vis.items():
                    try:
                        disp.SetSegmentVisibility(sid, bool(vis))
                    except Exception:
                        pass
            except Exception:
                mask = None
        finally:
            try:
                if labelmap is not None and labelmap.GetScene() == slicer.mrmlScene:
                    slicer.mrmlScene.RemoveNode(labelmap)
            except Exception:
                pass

        return mask

    def _segments_bbox(self, segmentation_node, segment_ids, reference_volume_node, margin_mm: float):
        """Compute a (z0,z1,y0,y1,x0,x1) bbox (end-exclusive) around selected segments.

        This is used to crop gamma computation to a small ROI to avoid OOM/crashes when
        low-dose cutoff is small (large evaluation region).
        """
        if slicer.mrmlScene is None:
            return None
        if segmentation_node is None or reference_volume_node is None:
            return None
        if not segment_ids:
            return None

        # Convert margin (mm) to voxels for each axis (array is z,y,x).
        try:
            sx, sy, sz = (float(v) for v in reference_volume_node.GetSpacing())
        except Exception:
            sx, sy, sz = (1.0, 1.0, 1.0)
        try:
            mx = int(np.ceil(float(margin_mm) / max(sx, 1e-6))) + 2
            my = int(np.ceil(float(margin_mm) / max(sy, 1e-6))) + 2
            mz = int(np.ceil(float(margin_mm) / max(sz, 1e-6))) + 2
        except Exception:
            mx = my = mz = 2

        # Prefer per-segment utility when available.
        try:
            fn = getattr(slicer.util, "arrayFromSegmentBinaryLabelmap", None)
            if callable(fn):
                zmin = ymin = xmin = None
                zmax = ymax = xmax = None
                nz = ny = nx = None
                for sid in list(segment_ids):
                    try:
                        arr = fn(segmentation_node, sid, reference_volume_node)
                    except Exception:
                        arr = None
                    if arr is None:
                        continue
                    mask = np.asarray(arr) > 0
                    if nz is None:
                        nz, ny, nx = mask.shape
                    if not np.any(mask):
                        continue
                    zz, yy, xx = np.where(mask)
                    zmin = int(zz.min()) if zmin is None else min(zmin, int(zz.min()))
                    zmax = int(zz.max()) if zmax is None else max(zmax, int(zz.max()))
                    ymin = int(yy.min()) if ymin is None else min(ymin, int(yy.min()))
                    ymax = int(yy.max()) if ymax is None else max(ymax, int(yy.max()))
                    xmin = int(xx.min()) if xmin is None else min(xmin, int(xx.min()))
                    xmax = int(xx.max()) if xmax is None else max(xmax, int(xx.max()))

                if nz is None or zmin is None:
                    return None

                z0 = max(0, zmin - mz)
                z1 = min(int(nz), zmax + mz + 1)
                y0 = max(0, ymin - my)
                y1 = min(int(ny), ymax + my + 1)
                x0 = max(0, xmin - mx)
                x1 = min(int(nx), xmax + mx + 1)
                return (z0, z1, y0, y1, x0, x1)
        except Exception:
            pass

        # Fallback: export all selected segments into one labelmap.
        labelmap = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", f"tmp_bbox_{uuid4().hex[:6]}"
        )
        try:
            labelmap.SetHideFromEditors(1)
            labelmap.SetSelectable(0)
            labelmap.SetSaveWithScene(0)
        except Exception:
            pass

        arr = None
        try:
            seg_logic = slicer.modules.segmentations.logic()
            seg_ids = vtk.vtkStringArray()
            for sid in list(segment_ids):
                seg_ids.InsertNextValue(str(sid))
            seg_logic.ExportSegmentsToLabelmapNode(segmentation_node, seg_ids, labelmap, reference_volume_node)
            arr = slicer.util.arrayFromVolume(labelmap)
        except Exception:
            arr = None
        finally:
            try:
                if labelmap is not None and labelmap.GetScene() == slicer.mrmlScene:
                    slicer.mrmlScene.RemoveNode(labelmap)
            except Exception:
                pass

        if arr is None:
            return None

        try:
            nz, ny, nx = arr.shape
            mask = np.asarray(arr) > 0
            if not np.any(mask):
                return None
            zz, yy, xx = np.where(mask)
            z0 = max(0, int(zz.min()) - mz)
            z1 = min(nz, int(zz.max()) + mz + 1)
            y0 = max(0, int(yy.min()) - my)
            y1 = min(ny, int(yy.max()) + my + 1)
            x0 = max(0, int(xx.min()) - mx)
            x1 = min(nx, int(xx.max()) + mx + 1)
            return (z0, z1, y0, y1, x0, x1)
        except Exception:
            return None

    def _run_in_thread(self, fn, on_done, on_error, poll_ms: int = 100) -> None:
        state = {"done": False, "result": None, "error": None}

        def _wrapper():
            try:
                state["result"] = fn()
            except Exception as exc:
                state["error"] = exc
            finally:
                state["done"] = True

        t = threading.Thread(target=_wrapper)
        t.daemon = True
        t.start()

        def _poll():
            if self._active_job is None:
                return
            if not state["done"]:
                try:
                    QTimer.singleShot(int(poll_ms), _poll)
                except Exception:
                    pass
                return
            if state["error"] is not None:
                on_error(state["error"])
            else:
                on_done(state["result"])

        _poll()

    def _run_cli_async(self, cli_module, params: dict, on_done, on_error) -> None:
        """Run a Slicer CLI without blocking the UI."""
        try:
            cli_node = slicer.cli.run(cli_module, None, params, wait_for_completion=False)
        except Exception as exc:
            on_error(exc)
            return

        holder = {"tag": None, "handled": False}

        def _cleanup():
            try:
                if holder["tag"] is not None:
                    cli_node.RemoveObserver(holder["tag"])
            except Exception:
                pass
            try:
                if cli_node is not None and cli_node.GetScene() == slicer.mrmlScene:
                    slicer.mrmlScene.RemoveNode(cli_node)
            except Exception:
                pass

        def _finish(ok: bool, err: Exception = None):
            # Avoid removing MRML nodes inside the VTK ModifiedEvent callstack (can crash Slicer).
            if holder.get("handled", False):
                return
            holder["handled"] = True

            def _do_finish():
                _cleanup()
                if ok:
                    on_done()
                else:
                    on_error(err or RuntimeError("CLI failed"))

            try:
                QTimer.singleShot(0, _do_finish)
            except Exception:
                _do_finish()

        def _status_tuple():
            try:
                return (cli_node.GetStatus(), str(cli_node.GetStatusString()))
            except Exception:
                try:
                    return (None, str(cli_node.GetStatusString()))
                except Exception:
                    return (None, "")

        def _on_modified(caller, event):
            if self._active_job is None:
                _finish(True)
                return

            status, status_str = _status_tuple()
            try:
                completed = hasattr(cli_node, "Completed") and status == cli_node.Completed
                failed = hasattr(cli_node, "Failed") and status == cli_node.Failed
                cancelled = hasattr(cli_node, "Cancelled") and status == cli_node.Cancelled
            except Exception:
                completed = failed = cancelled = False

            if (status_str or "").lower() in ("completed", "completed with errors"):
                completed = True
            if (status_str or "").lower() in ("failed",):
                failed = True
            if (status_str or "").lower() in ("cancelled", "canceled"):
                cancelled = True

            if not (completed or failed or cancelled):
                return

            if failed:
                msg = None
                try:
                    msg = str(cli_node.GetErrorText())
                except Exception:
                    msg = None
                _finish(False, RuntimeError(msg or "CLI failed"))
                return

            if cancelled:
                _finish(False, RuntimeError("CLI cancelled"))
                return

            _finish(True)

        try:
            holder["tag"] = cli_node.AddObserver(vtk.vtkCommand.ModifiedEvent, _on_modified)
        except Exception as exc:
            _cleanup()
            on_error(exc)

    def _finish_job(self, ok: bool, message: str = "") -> None:
        self._active_job = None
        self._set_ui_busy(False)
        if ok:
            self._set_progress(100, visible=False)
            self._set_status(message or "Done.")
        else:
            self._set_progress(0, visible=False)
            self._set_status("")
            if message:
                try:
                    QMessageBox.warning(self, "Compute Error", str(message))
                except Exception:
                    pass

    def _on_compute_metrics(self) -> None:
        # Avoid blocking the UI: run as an async job.
        if self._active_job is not None:
            try:
                QMessageBox.information(self, "Busy", "A metrics computation is already running.")
            except Exception:
                pass
            return

        if slicer.mrmlScene is None:
            QMessageBox.warning(self, "No Scene", "MRML scene is not available.")
            return

        ref_dose_node = self._selected_ref_dose_node()
        out_dose_node = self._selected_out_dose_node()
        seg_node = self.seg_selector.currentNode() if hasattr(self, "seg_selector") else None
        if ref_dose_node is None or out_dose_node is None or seg_node is None:
            QMessageBox.warning(self, "Missing Inputs", "Select a reference dose, an output dose, and a segmentation.")
            return

        # Selected segments
        try:
            selected_seg_ids = [sid for sid, cb in self._segment_checkbox_by_id.items() if cb is not None and cb.isChecked()]
        except Exception:
            selected_seg_ids = []
        if not selected_seg_ids:
            QMessageBox.warning(self, "Missing Inputs", "Select at least one segment to compute metrics.")
            return

        unc_node = self._selected_unc_node()

        if self.cb_dose_minmax_3sigma.isChecked() and unc_node is None:
            QMessageBox.warning(
                self,
                "Missing Inputs",
                "To compute dose min/max as mean ± 3*unc, please select an output uncertainty volume (optional input).",
            )
            return

        out_name = self._line_edit_text(self.output_name_edit).strip() or self._generate_default_output_name()
        table_node = self._create_or_get_table_node(out_name)
        if table_node is None:
            QMessageBox.warning(self, "Output Error", "Could not create output table node.")
            return

        # Start async job state
        self._active_job = {
            "out_name": out_name,
            "table_node": table_node,
            "ref_dose_node": ref_dose_node,
            "out_dose_node": out_dose_node,
            "unc_node": unc_node,
            "seg_node": seg_node,
            "selected_seg_ids": list(selected_seg_ids),
            "temp_nodes": [],
            "ref_eval_node": ref_dose_node,
            "unc_eval_node": unc_node,
            "per_seg": {},
            "gamma_arr": None,
            "stage": "start",
        }

        self._set_ui_busy(True)
        self._set_status("Preparing…")
        self._set_progress(0, visible=True)

        def _cleanup_temp_nodes():
            job = self._active_job
            if job is None:
                return
            for tn in job.get("temp_nodes", []):
                try:
                    if tn is not None and slicer.mrmlScene is not None and tn.GetScene() == slicer.mrmlScene:
                        slicer.mrmlScene.RemoveNode(tn)
                except Exception:
                    pass

        def _fail(msg: str):
            _cleanup_temp_nodes()
            self._finish_job(False, msg)

        def _step_after_resample():
            job = self._active_job
            if job is None:
                return

            self._set_status("Loading arrays…")
            self._set_progress(25, visible=True)

            try:
                out_arr = slicer.util.arrayFromVolume(job["out_dose_node"]).astype(np.float64, copy=True)
                ref_arr = slicer.util.arrayFromVolume(job["ref_eval_node"]).astype(np.float64, copy=True)
                unc_arr = None
                if job.get("unc_eval_node") is not None:
                    unc_arr = slicer.util.arrayFromVolume(job["unc_eval_node"]).astype(np.float64, copy=True)
            except Exception as exc:
                logger.exception("Failed to read volume arrays")
                _fail(f"Failed to read arrays: {exc}")
                return

            job["_out_arr"] = out_arr
            job["_ref_arr"] = ref_arr
            job["_unc_arr"] = unc_arr

            # Pre-compute gamma ROI bbox on UI thread (uses MRML).
            if self.cb_gamma_pr.isChecked():
                # Preflight PyMedPhys import on UI thread. Installing packages from a worker thread can crash Slicer.
                try:
                    import pymedphys  # type: ignore  # noqa: F401
                except Exception:
                    _fail(
                        "PyMedPhys is not installed (needed for Gamma). "
                        "Install it in Slicer Python (e.g. `slicer.util.pip_install('pymedphys')`) "
                        "or disable Gamma_PR.%"
                    )
                    return
                try:
                    distance_mm_threshold = float(self._float_from_line_edit(self.gamma_dist_mm_edit, 3.0))
                except Exception:
                    distance_mm_threshold = 3.0
                bbox = None
                try:
                    bbox = self._segments_bbox(job["seg_node"], job.get("selected_seg_ids", []), job["out_dose_node"], margin_mm=float(distance_mm_threshold))
                except Exception:
                    bbox = None
                job["_gamma_bbox"] = bbox

            if self.cb_gamma_pr.isChecked():
                self._set_status("Computing gamma (PyMedPhys)…")

                # Gamma duration is hard to predict; advance progress slowly to provide feedback.
                job["_gamma_prog"] = 25

                def _tick_gamma_progress():
                    j = self._active_job
                    if j is None:
                        return
                    if j.get("gamma_arr") is not None:
                        return
                    try:
                        p = int(j.get("_gamma_prog", 25))
                    except Exception:
                        p = 25
                    p = min(39, p + 1)
                    j["_gamma_prog"] = p
                    self._set_progress(p, visible=True)
                    try:
                        QTimer.singleShot(200, _tick_gamma_progress)
                    except Exception:
                        pass

                _tick_gamma_progress()

                def _gamma_fn():
                    import pymedphys  # type: ignore

                    try:
                        sx, sy, sz = (float(v) for v in job["ref_eval_node"].GetSpacing())
                    except Exception:
                        sx, sy, sz = (1.0, 1.0, 1.0)
                    nz, ny, nx = job["_ref_arr"].shape
                    z = np.arange(nz, dtype=np.float64) * sz
                    y = np.arange(ny, dtype=np.float64) * sy
                    x = np.arange(nx, dtype=np.float64) * sx
                    axes = (z, y, x)

                    dose_percent_threshold = float(self._float_from_line_edit(self.gamma_dose_percent_edit, 3.0))
                    distance_mm_threshold = float(self._float_from_line_edit(self.gamma_dist_mm_edit, 3.0))
                    lower_percent_dose_cutoff = float(self._float_from_line_edit(self.gamma_low_cutoff_edit, 10.0))
                    local_gamma = bool(self._gamma_mode_is_local())

                    bbox = job.get("_gamma_bbox", None)
                    if bbox is not None:
                        z0, z1, y0, y1, x0, x1 = bbox
                        ref_crop = job["_ref_arr"][z0:z1, y0:y1, x0:x1]
                        out_crop = job["_out_arr"][z0:z1, y0:y1, x0:x1]
                        zc = axes[0][z0:z1]
                        yc = axes[1][y0:y1]
                        xc = axes[2][x0:x1]
                        axes_crop = (zc, yc, xc)
                        return pymedphys.gamma(
                            axes_crop,
                            ref_crop,
                            axes_crop,
                            out_crop,
                            dose_percent_threshold=dose_percent_threshold,
                            distance_mm_threshold=distance_mm_threshold,
                            lower_percent_dose_cutoff=lower_percent_dose_cutoff,
                            interp_fraction=3,
                            max_gamma=1.5,
                            local_gamma=local_gamma,
                        )

                    return pymedphys.gamma(
                        axes,
                        job["_ref_arr"],
                        axes,
                        job["_out_arr"],
                        dose_percent_threshold=dose_percent_threshold,
                        distance_mm_threshold=distance_mm_threshold,
                        lower_percent_dose_cutoff=lower_percent_dose_cutoff,
                        interp_fraction=3,
                        max_gamma=1.5,
                        local_gamma=local_gamma,
                    )

                def _gamma_done(gamma_arr):
                    job2 = self._active_job
                    if job2 is None:
                        return
                    job2["gamma_arr"] = gamma_arr
                    self._set_progress(40, visible=True)
                    _start_per_segment_loop()

                def _gamma_err(exc):
                    logger.exception("Gamma computation failed")
                    _fail(f"Gamma failed: {exc}")

                self._run_in_thread(_gamma_fn, _gamma_done, _gamma_err, poll_ms=150)
            else:
                _start_per_segment_loop()

        def _start_per_segment_loop():
            job = self._active_job
            if job is None:
                return

            seg_ids = job.get("selected_seg_ids", [])
            total = max(1, len(seg_ids))
            job["_seg_index"] = 0

            self._set_status("Computing per-segment metrics…")
            self._set_progress(40, visible=True)

            def _one_segment():
                job2 = self._active_job
                if job2 is None:
                    return
                i = int(job2.get("_seg_index", 0))
                if i >= len(seg_ids):
                    _build_table_and_finish()
                    return

                seg_id = seg_ids[i]
                mask = self._export_segment_mask(job2["seg_node"], seg_id, job2["out_dose_node"])
                count = 0
                try:
                    count = int(np.count_nonzero(mask)) if mask is not None else 0
                except Exception:
                    count = 0

                seg = None
                try:
                    seg = job2["seg_node"].GetSegmentation()
                except Exception:
                    seg = None
                seg_name = str(seg_id)
                try:
                    seg_obj = seg.GetSegment(seg_id) if seg is not None else None
                    seg_name = seg_obj.GetName() if seg_obj is not None else str(seg_id)
                except Exception:
                    seg_name = str(seg_id)

                if mask is None or count <= 0:
                    job2["per_seg"][seg_id] = {
                        "name": seg_name,
                        "mean": np.nan,
                        "std": np.nan,
                        "mae": np.nan,
                        "unc_mean": np.nan,
                        "gamma_pr": np.nan,
                    }
                else:
                    out_vals = job2["_out_arr"][mask]
                    mean_v = float(np.mean(out_vals))
                    std_v = float(np.std(out_vals))

                    mae = np.nan
                    if self.cb_err_mae.isChecked():
                        mae = float(np.mean(np.abs(job2["_out_arr"][mask] - job2["_ref_arr"][mask])))

                    unc_mean = np.nan
                    if job2.get("_unc_arr") is not None:
                        unc_mean = float(np.mean(job2["_unc_arr"][mask]))

                    gamma_pr = np.nan
                    if job2.get("gamma_arr") is not None:
                        try:
                            g = job2["gamma_arr"]
                            bbox = job2.get("_gamma_bbox", None)
                            if bbox is not None:
                                z0, z1, y0, y1, x0, x1 = bbox
                                m = mask[z0:z1, y0:y1, x0:x1]
                            else:
                                m = mask
                            valid = m & np.isfinite(g)
                            denom = int(np.count_nonzero(valid))
                            if denom > 0:
                                passed = int(np.count_nonzero(valid & (g <= 1.0)))
                                gamma_pr = 100.0 * passed / float(denom)
                        except Exception:
                            gamma_pr = np.nan

                    job2["per_seg"][seg_id] = {
                        "name": seg_name,
                        "mean": mean_v,
                        "std": std_v,
                        "mae": mae,
                        "unc_mean": unc_mean,
                        "gamma_pr": gamma_pr,
                    }

                job2["_seg_index"] = i + 1
                # progress 40->90 during segments
                prog = 40 + int(50 * (job2["_seg_index"] / float(total)))
                self._set_progress(prog, visible=True)
                self._set_status(f"Computing per-segment metrics… ({job2['_seg_index']}/{len(seg_ids)})")

                try:
                    QTimer.singleShot(0, _one_segment)
                except Exception:
                    _one_segment()

            try:
                QTimer.singleShot(0, _one_segment)
            except Exception:
                _one_segment()

        def _build_table_and_finish():
            job = self._active_job
            if job is None:
                return
            self._set_status("Building table…")
            self._set_progress(95, visible=True)

            per_seg = job.get("per_seg", {})
            table_node = job["table_node"]
            table = table_node.GetTable()
            table.Initialize()

            def _fmt_float(value) -> str:
                try:
                    v = float(value)
                except Exception:
                    return ""
                return "" if not np.isfinite(v) else f"{v:.6g}"

            def add_col(name: str):
                col = vtk.vtkStringArray()
                col.SetName(name)
                table.AddColumn(col)

            name_col = vtk.vtkStringArray()
            name_col.SetName("Segment")
            table.AddColumn(name_col)

            if self.cb_dose_mean.isChecked():
                add_col("Dose_mean")
            if self.cb_dose_minmax_3sigma.isChecked():
                add_col("Dose_min")
                add_col("Dose_max")
            if self.cb_err_mae.isChecked():
                add_col("Err_MAE")
            if self.cb_unc_mean.isChecked():
                add_col("Unc_mean")
            if self.cb_gamma_pr.isChecked():
                add_col("Gamma_PR_%")

            rows = []
            for seg_id in job.get("selected_seg_ids", []):
                d = per_seg.get(seg_id, {"name": seg_id})
                seg_name = str(d.get("name", seg_id))

                values = []
                if self.cb_dose_mean.isChecked():
                    values.append(_fmt_float(d.get("mean", np.nan)))
                if self.cb_dose_minmax_3sigma.isChecked():
                    mean_v = float(d.get("mean", np.nan))
                    unc_v = float(d.get("unc_mean", np.nan))
                    if np.isfinite(mean_v) and np.isfinite(unc_v):
                        values.append(_fmt_float(max(0.0, mean_v - 3.0 * unc_v)))
                        values.append(_fmt_float(mean_v + 3.0 * unc_v))
                    else:
                        values.append("")
                        values.append("")
                if self.cb_err_mae.isChecked():
                    values.append(_fmt_float(d.get("mae", np.nan)))
                if self.cb_unc_mean.isChecked():
                    values.append(_fmt_float(d.get("unc_mean", np.nan)))
                if self.cb_gamma_pr.isChecked():
                    values.append(_fmt_float(d.get("gamma_pr", np.nan)))

                if any(v != "" for v in values):
                    rows.append((seg_name, values))

            table.SetNumberOfRows(len(rows))
            for r, (seg_name, values) in enumerate(rows):
                table.SetValue(r, 0, seg_name)
                for i, v in enumerate(values, start=1):
                    table.SetValue(r, i, v)

            table_node.Modified()

            _cleanup_temp_nodes()
            try:
                slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpTableView)
            except Exception:
                pass
            self._set_progress(100, visible=False)
            self._finish_job(True, f"Done. Output table: {job['out_name']}")

        # Resample reference/uncertainty to output dose geometry if needed (async)
        self._set_status("Resampling inputs (if needed)…")
        self._set_progress(10, visible=True)

        def _start_resample_ref():
            job = self._active_job
            if job is None:
                return
            needs = False
            try:
                needs = self._needs_resample_to_reference(job["ref_dose_node"], job["out_dose_node"])
            except Exception:
                needs = False
            if not needs:
                _start_resample_unc()
                return

            out_name_rs = f"{self._safe_node_name(job['ref_dose_node'])}_resampled_{uuid4().hex[:6]}"
            out_node_rs = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", out_name_rs)
            try:
                out_node_rs.SetHideFromEditors(1)
                out_node_rs.SetSelectable(0)
                out_node_rs.SetSaveWithScene(0)
            except Exception:
                pass

            job["ref_eval_node"] = out_node_rs
            job["temp_nodes"].append(out_node_rs)
            params = {
                "inputVolume": job["ref_dose_node"].GetID(),
                "referenceVolume": job["out_dose_node"].GetID(),
                "outputVolume": out_node_rs.GetID(),
                "interpolationType": "linear",
            }

            def _done():
                self._set_progress(15, visible=True)
                _start_resample_unc()

            def _err(exc):
                logger.exception("Reference resample failed")
                _fail(f"Reference resample failed: {exc}")

            self._run_cli_async(slicer.modules.resamplescalarvectordwivolume, params, _done, _err)

        def _start_resample_unc():
            job = self._active_job
            if job is None:
                return
            if job.get("unc_node") is None:
                _step_after_resample()
                return
            needs = False
            try:
                needs = self._needs_resample_to_reference(job["unc_node"], job["out_dose_node"])
            except Exception:
                needs = False
            if not needs:
                _step_after_resample()
                return

            out_name_rs = f"{self._safe_node_name(job['unc_node'])}_resampled_{uuid4().hex[:6]}"
            out_node_rs = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", out_name_rs)
            try:
                out_node_rs.SetHideFromEditors(1)
                out_node_rs.SetSelectable(0)
                out_node_rs.SetSaveWithScene(0)
            except Exception:
                pass

            job["unc_eval_node"] = out_node_rs
            job["temp_nodes"].append(out_node_rs)
            params = {
                "inputVolume": job["unc_node"].GetID(),
                "referenceVolume": job["out_dose_node"].GetID(),
                "outputVolume": out_node_rs.GetID(),
                "interpolationType": "linear",
            }

            def _done():
                self._set_progress(20, visible=True)
                _step_after_resample()

            def _err(exc):
                logger.exception("Uncertainty resample failed")
                _fail(f"Uncertainty resample failed: {exc}")

            self._run_cli_async(slicer.modules.resamplescalarvectordwivolume, params, _done, _err)

        try:
            QTimer.singleShot(0, _start_resample_ref)
        except Exception:
            _start_resample_ref()
        return
