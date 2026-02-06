import logging
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

import numpy as np
import slicer
import vtk
from qt import QCheckBox, QMessageBox, QTimer, QVBoxLayout
from widgets.base_widget import BaseImpactWidget

os.environ["NUMBA_THREADING_LAYER"] = "omp"

logger = logging.getLogger(__name__)


class _ThreadState(TypedDict):
    done: bool
    result: Any
    error: Exception | None


class MetricsEvaluationWidget(BaseImpactWidget):
    """UI widget for Phase 3: Metrics & Evaluation."""

    def __init__(self, logic):
        super().__init__(logic)
        self._ref_dose_node_by_index = {}
        self._out_dose_node_by_index = {}
        self._unc_node_by_index = {}
        self._segment_checkbox_by_id = {}
        self._active_job = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        ui_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "../Resources/UI/MetricsWidget.ui"))
        ui_widget = slicer.util.loadUI(ui_path)
        self.ui = slicer.util.childWidgetVariables(ui_widget)
        self._root_widget = ui_widget

        # Bind widgets
        self.ref_dose_combo = self._w("ref_dose_combo")
        self.out_dose_combo = self._w("out_dose_combo")
        # Output uncertainty: checkbox enabled only when an
        # uncertainty volume exists in the same folder as the output dose
        self.cb_unc_out = self._w("cb_unc_out")
        self._unc_node_by_index = {}  # kept for compatibility but not used when using folder-based uncertainty
        self.seg_selector = self._w("seg_selector")
        self.output_name_edit = self._w("output_name_edit")
        self.cb_dose_mean = self._w("cb_dose_mean")
        self.cb_dose_minmax_3sigma = self._w("cb_dose_minmax_3sigma")
        self.cb_err_mae = self._w("cb_err_mae")
        self.cb_unc_mean = self._w("cb_unc_mean")
        self.cb_gamma_pr = self._w("cb_gamma_pr")
        self.gamma_dose_percent_edit = self._w("gamma_dose_percent_edit")
        self.gamma_dist_mm_edit = self._w("gamma_dist_mm_edit")
        self.gamma_low_cutoff_edit = self._w("gamma_low_cutoff_edit")
        self.gamma_mode_combo = self._w("gamma_mode_combo")
        self.run_btn = self._w("run_btn")
        self.status_label = self._w("status_label")
        self.progress_bar = self._w("progress_bar")
        self._segments_group = self._w("segments_group")
        self._segments_scroll = self._w("segments_scroll")
        self._segments_scroll_content = self._w("segments_scroll_content")
        self._segments_scroll_layout = self._layout("segments_scroll_content")
        self._gamma_params_widget = self._w("gamma_params_widget")
        self._gamma_label_dose_diff = self._w("gamma_label_dose_diff")
        self._gamma_label_dta = self._w("gamma_label_dta")
        self._gamma_label_cutoff = self._w("gamma_label_cutoff")

        # Configure segmentation selector
        if self.seg_selector is not None:
            self.seg_selector.nodeTypes = ["vtkMRMLSegmentationNode"]
            self.seg_selector.selectNodeUponCreation = False
            self.seg_selector.addEnabled = False
            self.seg_selector.removeEnabled = False
            self.seg_selector.noneEnabled = True
            self.seg_selector.showHidden = False
            self.seg_selector.setMRMLScene(slicer.mrmlScene)

        # Buttons & signals
        self._btn("refresh_btn", self._refresh_lists)
        if self.run_btn is not None:
            self.run_btn.clicked.connect(self._on_compute_metrics)

        if self.seg_selector is not None:
            self.seg_selector.currentNodeChanged.connect(self._on_segmentation_changed)
        # Wire output dose selection changes to update uncertainty checkbox eligibility
        if self.out_dose_combo is not None:
            self.out_dose_combo.currentIndexChanged.connect(self._on_out_dose_changed)

        # Checkbox defaults (gamma off by default)
        for cb in (
            self.cb_dose_mean,
            self.cb_dose_minmax_3sigma,
            self.cb_err_mae,
            self.cb_unc_mean,
        ):
            if cb is not None:
                cb.setChecked(True)
        if self.cb_gamma_pr is not None:
            self.cb_gamma_pr.setChecked(False)

        def _sync_gamma_params_visibility():
            if self._gamma_params_widget is not None and self.cb_gamma_pr is not None:
                self._gamma_params_widget.setVisible(bool(self.cb_gamma_pr.isChecked()))

        if self.cb_gamma_pr is not None:
            self.cb_gamma_pr.toggled.connect(_sync_gamma_params_visibility)
        _sync_gamma_params_visibility()

        if self.output_name_edit is not None:
            self.output_name_edit.setText(self._generate_default_output_name(prefix="metrics"))

        if self.progress_bar is not None:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(False)

        layout = QVBoxLayout(self)
        layout.addWidget(ui_widget)
        self.setLayout(layout)

        self._refresh_lists()
        self._on_segmentation_changed(self.seg_selector.currentNode() if self.seg_selector is not None else None)

    def _clear_layout(self, layout) -> None:
        if layout is None:
            return
        try:
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget() if item is not None else None
                if w is not None:
                    try:
                        w.setParent(None)
                        w.deleteLater()
                    except Exception:
                        logger.exception("Failed to remove widget from layout")
        except Exception:
            logger.exception("_clear_layout failed")

    # Preheat removed — do not perform JIT warm-up automatically from the module.

    def _on_segmentation_changed(self, seg_node) -> None:
        self._segment_checkbox_by_id = {}
        self._clear_layout(self._segments_scroll_layout)

        if seg_node is None:
            self._segments_group.setEnabled(False)
        try:
            seg = seg_node.GetSegmentation()
        except Exception:
            seg = None

        if seg is None:
            self._segments_group.setEnabled(False)
            return

        self._segments_group.setEnabled(True)

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

    def _refresh_lists(self) -> None:
        """Refresh filtered lists for reference/output dose and uncertainty."""
        if slicer.mrmlScene is None:
            return

        # Keep current selections by node ID if possible.
        ref_prev = None
        out_prev = None
        try:
            ref_prev = self._ref_dose_node_by_index.get(self._combo_current_index(self.ref_dose_combo), None)
        except Exception:
            ref_prev = None
        try:
            out_prev = self._out_dose_node_by_index.get(self._combo_current_index(self.out_dose_combo), None)
        except Exception:
            out_prev = None

        def node_id(n):
            try:
                return n.GetID() if n is not None and hasattr(n, "GetID") else None
            except Exception:
                return None

        ref_prev_id = node_id(ref_prev)
        out_prev_id = node_id(out_prev)

        self.ref_dose_combo.blockSignals(True)
        self.out_dose_combo.blockSignals(True)

        self.ref_dose_combo.clear()
        self.out_dose_combo.clear()

        self._ref_dose_node_by_index = {0: None}
        self._out_dose_node_by_index = {0: None}
        self._unc_node_by_index = {0: None}

        self.ref_dose_combo.addItem("[Select reference dose]")
        self.out_dose_combo.addItem("[Select output dose]")

        # Filter volumes by name.
        try:
            volumes = list(slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"))
        except Exception:
            volumes = []

        # Reference dose: dose volumes, excluding uncertainty volumes.
        dose_nodes_ref = [
            n
            for n in volumes
            if n is not None and self._is_name_match(n, "dose") and not self._is_name_match(n, "uncertainty")
        ]
        # Output dose: exclude uncertainty volumes that often contain "dose" in their name.
        dose_nodes_out = [
            n
            for n in volumes
            if n is not None and self._is_name_match(n, "dose") and not self._is_name_match(n, "uncertainty")
        ]

        dose_nodes_ref.sort(key=lambda n: self._safe_node_name(n).lower())
        dose_nodes_out.sort(key=lambda n: self._safe_node_name(n).lower())

        ref_match_index = 0
        out_match_index = 0

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

        self._on_out_dose_changed()
        self.ref_dose_combo.blockSignals(False)
        self.out_dose_combo.blockSignals(False)

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
        """Legacy accessor: when using folder-based uncertainty the checkbox controls selection.

        Return: uncertainty node if the "use uncertainty" checkbox is checked and an uncertainty
        volume exists in the same SH folder as the selected output dose. Otherwise return None.
        """
        try:
            if getattr(self, "cb_unc_out", None) is not None and bool(self.cb_unc_out.isChecked()):
                return self._find_uncertainty_in_same_folder(self._selected_out_dose_node())
            return None
        except Exception:
            return None

    def _on_out_dose_changed(self) -> None:
        """Update uncertainty checkbox eligibility when output dose changes."""
        out_node = self._selected_out_dose_node()
        unc = self._find_uncertainty_in_same_folder(out_node) if out_node is not None else None
        if getattr(self, "cb_unc_out", None) is not None:
            self.cb_unc_out.setEnabled(bool(unc is not None))
            if unc is None:
                self.cb_unc_out.setChecked(False)

    def _needs_resample_to_reference(self, input_node, reference_node) -> bool:
        if input_node is None or reference_node is None:
            return False
        # Compare dimensions without loading voxel arrays.
        try:
            in_img = input_node.GetImageData()
            ref_img = reference_node.GetImageData()
            in_dims = in_img.GetDimensions() if in_img is not None else None
            ref_dims = ref_img.GetDimensions() if ref_img is not None else None
            if in_dims is None or ref_dims is None or in_dims != ref_dims:
                return True
        except Exception:
            pass

        # Compare IJK->RAS matrices.
        m_in = vtk.vtkMatrix4x4()
        m_ref = vtk.vtkMatrix4x4()
        try:
            input_node.GetIJKToRASMatrix(m_in)
            reference_node.GetIJKToRASMatrix(m_ref)
            for r in range(4):
                for c in range(4):
                    if abs(m_in.GetElement(r, c) - m_ref.GetElement(r, c)) > 1e-6:
                        return True
        except Exception:
            pass
        return False

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
        # currentText may be exposed as a property or a callable; normalize before checking
        if self.gamma_mode_combo is None:
            return False
        try:
            val = self.gamma_mode_combo.currentText
            txt = val() if callable(val) else val
            return str(txt).lower().startswith("local")
        except Exception:
            return False

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

    def _ensure_node_in_sh_folder(self, node, folder_item_id):
        if slicer.mrmlScene is None or node is None or not folder_item_id:
            return
        sh_node = self._get_sh_node()
        if sh_node is None:
            return
        try:
            item_id = sh_node.GetItemByDataNode(node)
        except Exception:
            item_id = 0
        if item_id == 0:
            try:
                item_id = sh_node.CreateItem(folder_item_id, node)
            except Exception:
                item_id = 0
        if item_id:
            try:
                sh_node.SetItemParent(item_id, folder_item_id)
            except Exception:
                pass

    def _get_or_create_child_folder_item(self, reference_node, folder_name: str):
        if slicer.mrmlScene is None or reference_node is None:
            return None
        sh_node = self._get_sh_node()
        if sh_node is None:
            return None
        try:
            ref_item_id = sh_node.GetItemByDataNode(reference_node)
        except Exception:
            ref_item_id = 0
        if not ref_item_id:
            return None

        children = vtk.vtkIdList()
        try:
            sh_node.GetItemChildren(ref_item_id, children, False)
        except Exception:
            return None
        for i in range(children.GetNumberOfIds()):
            child_id = children.GetId(i)
            try:
                if sh_node.GetItemName(child_id) == folder_name and sh_node.GetItemDataNode(child_id) is None:
                    return child_id
            except Exception:
                continue
        try:
            return sh_node.CreateFolderItem(ref_item_id, folder_name)
        except Exception:
            return None

    def _get_output_dir_from_ref(self, reference_node, child_folder: str):
        if reference_node is None:
            return None
        try:
            storage = reference_node.GetStorageNode()
        except Exception:
            storage = None
        if storage is None:
            return None
        try:
            ref_path = storage.GetFileName()
        except Exception:
            ref_path = None
        if not ref_path:
            return None
        try:
            base_dir = os.path.dirname(ref_path)
            out_dir = os.path.join(base_dir, child_folder)
            os.makedirs(out_dir, exist_ok=True)
            return out_dir
        except Exception:
            return None

    def _set_ui_busy(self, busy: bool) -> None:
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Stop" if bool(busy) else "Compute metrics")

        self.ref_dose_combo.setEnabled(not bool(busy))
        self.out_dose_combo.setEnabled(not bool(busy))
        if getattr(self, "cb_unc_out", None) is not None:
            self.cb_unc_out.setEnabled(not bool(busy))

        self.seg_selector.setEnabled(not bool(busy))
        self.output_name_edit.setEnabled(not bool(busy))

    def _run_in_thread(self, fn, on_done, on_error, poll_ms: int = 100) -> None:
        state: _ThreadState = {"done": False, "result": None, "error": None}

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

    def _run_cli_async(self, cli_module, params: dict, on_done, on_error):
        """Delegate to base implementation which centralizes CLI handling."""
        return super()._run_cli_async(cli_module, params, on_done, on_error)

    def _cancel_active_job(self) -> None:
        job = self._active_job
        if job is None:
            return
        try:
            job["cancelled"] = True
        except Exception:
            pass
        self._set_status("Cancelling…")
        self._set_progress(None, visible=True)

        cli_node = None
        try:
            cli_node = job.get("_cli_node")
        except Exception:
            cli_node = None
        if cli_node is not None:
            try:
                cancel_fn = getattr(slicer.cli, "cancel", None)
                if callable(cancel_fn):
                    cancel_fn(cli_node)
                else:
                    cli_node.Cancel()
            except Exception:
                pass

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
            self._cancel_active_job()
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
            selected_seg_ids = [
                sid for sid, cb in self._segment_checkbox_by_id.items() if cb is not None and cb.isChecked()
            ]
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

        out_name = self._line_edit_text(self.output_name_edit).strip() or self._generate_default_output_name(
            prefix="metrics"
        )
        table_node = self._create_or_get_table_node(out_name)
        if table_node is None:
            QMessageBox.warning(self, "Output Error", "Could not create output table node.")
            return

        folder_item_id = self._get_or_create_child_folder_item(ref_dose_node, "metrics")
        if folder_item_id:
            self._ensure_node_in_sh_folder(table_node, folder_item_id)

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
            "cancelled": False,
            "_cli_node": None,
            "_metrics_folder_item_id": folder_item_id,
            "_metrics_output_dir": self._get_output_dir_from_ref(ref_dose_node, "metrics"),
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
            job = self._active_job
            if job is not None and bool(job.get("cancelled", False)):
                _cleanup_temp_nodes()
                self._finish_job(True, "Cancelled.")
                return
            _cleanup_temp_nodes()
            self._finish_job(False, msg)

        def _is_cancelled() -> bool:
            j = self._active_job
            if j is None:
                return True
            return bool(j.get("cancelled", False))

        def _cancel_finish():
            _cleanup_temp_nodes()
            self._finish_job(True, "Cancelled.")

        def _step_after_resample():
            job = self._active_job
            if job is None:
                return
            if _is_cancelled():
                _cancel_finish()
                return

            self._set_status("Loading arrays…")
            self._set_progress(25, visible=True)

            def _load_arrays():
                # Avoid forcing float64 + copies: this can double memory and slow things down.
                # Gamma/metrics computations work fine with float32 in practice.
                out_arr = np.asarray(slicer.util.arrayFromVolume(job["out_dose_node"]), dtype=np.float32)
                ref_arr = np.asarray(slicer.util.arrayFromVolume(job["ref_eval_node"]), dtype=np.float32)
                unc_arr = None
                if job.get("unc_eval_node") is not None:
                    unc_arr = np.asarray(slicer.util.arrayFromVolume(job["unc_eval_node"]), dtype=np.float32)
                return out_arr, ref_arr, unc_arr

            def _after_arrays_loaded(result):
                if _is_cancelled():
                    _cancel_finish()
                    return
                try:
                    out_arr, ref_arr, unc_arr = result
                except Exception as exc:
                    _fail(f"Failed to read arrays: {exc}")
                    return

                job["_out_arr"] = out_arr
                job["_ref_arr"] = ref_arr
                job["_unc_arr"] = unc_arr

                distance_mm_threshold = float(self._float_from_line_edit(self.gamma_dist_mm_edit, 2.0))

                # Gamma params/import preflight only when Gamma is enabled.
                if self.cb_gamma_pr.isChecked():
                    dose_percent_threshold = float(self._float_from_line_edit(self.gamma_dose_percent_edit, 2.0))
                    lower_percent_dose_cutoff = float(self._float_from_line_edit(self.gamma_low_cutoff_edit, 30.0))
                    local_gamma = bool(self._gamma_mode_is_local())

                    job["_gamma_params"] = {
                        "dose_percent_threshold": dose_percent_threshold,
                        "distance_mm_threshold": distance_mm_threshold,
                        "lower_percent_dose_cutoff": lower_percent_dose_cutoff,
                        "local_gamma": local_gamma,
                        "interp_fraction": 3,
                    }

                    try:
                        spacing = tuple(job["ref_eval_node"].GetSpacing())
                        job["_gamma_spacing"] = spacing
                    except Exception:
                        job["_gamma_spacing"] = None
                    m = vtk.vtkMatrix4x4()
                    job["ref_eval_node"].GetIJKToRASMatrix(m)
                    # Direction in RAS from IJKToRAS (columns scaled by spacing)
                    sx, sy, sz = job.get("_gamma_spacing") or (1.0, 1.0, 1.0)
                    dir_ras = [
                        [m.GetElement(0, 0) / sx, m.GetElement(0, 1) / sy, m.GetElement(0, 2) / sz],
                        [m.GetElement(1, 0) / sx, m.GetElement(1, 1) / sy, m.GetElement(1, 2) / sz],
                        [m.GetElement(2, 0) / sx, m.GetElement(2, 1) / sy, m.GetElement(2, 2) / sz],
                    ]
                    # Convert RAS -> LPS for SimpleITK
                    dir_lps = [
                        [-dir_ras[0][0], -dir_ras[0][1], -dir_ras[0][2]],
                        [-dir_ras[1][0], -dir_ras[1][1], -dir_ras[1][2]],
                        [dir_ras[2][0], dir_ras[2][1], dir_ras[2][2]],
                    ]
                    job["_gamma_direction_lps"] = tuple([v for row in dir_lps for v in row])
                    origin_ras = (m.GetElement(0, 3), m.GetElement(1, 3), m.GetElement(2, 3))
                    job["_gamma_origin_lps"] = (-origin_ras[0], -origin_ras[1], origin_ras[2])

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
                        # Use plastimatch CLI to compute gamma and return a numpy array.
                        distance_mm_threshold = float(job.get("_gamma_params", {}).get("distance_mm_threshold", 2.0))

                        dose_percent_threshold = float(job.get("_gamma_params", {}).get("dose_percent_threshold", 2.0))

                        plastimatch_bin = shutil.which("plastimatch")
                        if not plastimatch_bin:
                            raise RuntimeError("plastimatch not found in PATH; install plastimatch or add to PATH")

                        # Use a non-hidden path visible to snap-packaged plastimatch (snap may block dot-directories)
                        base_dir = Path.home() / "SlicerImpactDoseAcc_tmp" / "plastimatch"
                        try:
                            base_dir.mkdir(parents=True, exist_ok=True)
                        except Exception:
                            pass
                        td = tempfile.mkdtemp(prefix="impact_gamma_", dir=str(base_dir))
                        tdpath = Path(td)
                        ref_path = tdpath / "ref.nii.gz"
                        out_path = tdpath / "out.nii.gz"
                        gamma_path = tdpath / "gamma.nii.gz"

                        try:
                            # Write with SimpleITK only (avoid MRML operations from worker thread)
                            try:
                                import SimpleITK as sitk  # noqa: N813
                            except Exception as exc:
                                raise RuntimeError(f"SimpleITK is required for plastimatch gamma export: {exc}")

                            # SimpleITK expects arrays in (z,y,x)
                            img_ref = sitk.GetImageFromArray(np.asarray(job["_ref_arr"]).astype(np.float32))
                            img_out = sitk.GetImageFromArray(np.asarray(job["_out_arr"]).astype(np.float32))

                            # Set spacing / origin captured on UI thread (if available)
                            spacing = job.get("_gamma_spacing")
                            if spacing:
                                # SimpleITK spacing is (x,y,z)
                                img_ref.SetSpacing(spacing)
                                img_out.SetSpacing(spacing)
                            origin_lps = job.get("_gamma_origin_lps")
                            if origin_lps:
                                img_ref.SetOrigin(origin_lps)
                                img_out.SetOrigin(origin_lps)
                            direction_lps = job.get("_gamma_direction_lps")
                            if direction_lps:
                                img_ref.SetDirection(direction_lps)
                                img_out.SetDirection(direction_lps)

                            sitk.WriteImage(img_ref, str(ref_path), False)
                            sitk.WriteImage(img_out, str(out_path), False)
                            logger.debug("Wrote temp volumes for plastimatch")

                            try:
                                if not ref_path.exists() or not out_path.exists():
                                    raise RuntimeError("NIfTI files were not created")
                                if ref_path.stat().st_size == 0 or out_path.stat().st_size == 0:
                                    raise RuntimeError("NIfTI files are empty")
                            except Exception as exc:
                                raise RuntimeError(f"Failed to write NIfTI files for plastimatch: {exc}")

                            # plastimatch expects positional image arguments and separate tolerance options
                            dose_tol = float(dose_percent_threshold) / 100.0
                            dta_tol = float(distance_mm_threshold)
                            analysis_threshold = (
                                float(job.get("_gamma_params", {}).get("lower_percent_dose_cutoff", 30.0)) / 100.0
                            )
                            local_gamma = bool(job.get("_gamma_params", {}).get("local_gamma", False))
                            cmd = [
                                plastimatch_bin,
                                "gamma",
                                "--dose-tolerance",
                                str(dose_tol),
                                "--dta-tolerance",
                                str(dta_tol),
                                "--analysis-threshold",
                                str(analysis_threshold),
                                "--gamma-max",
                                "1.5",
                                "--interp-search",
                                "--output",
                                str(gamma_path),
                                str(ref_path),
                                str(out_path),
                            ]
                            if local_gamma:
                                cmd.insert(2, "--local-gamma")

                            logger.debug("Running plastimatch gamma")
                            try:
                                p = subprocess.run(
                                    cmd,
                                    check=True,
                                    capture_output=True,
                                    text=True,
                                    env=os.environ,
                                )
                                logger.debug("plastimatch stdout: %s", p.stdout)
                            except subprocess.CalledProcessError as exc:
                                stdout = getattr(exc, "stdout", None)
                                stderr = getattr(exc, "stderr", None)
                                logger.error(
                                    "plastimatch failed: returncode=%s stdout=%s stderr=%s",
                                    getattr(exc, "returncode", None),
                                    stdout,
                                    stderr,
                                )
                                raise RuntimeError(
                                    f"plastimatch gamma failed: returncode={getattr(exc,'returncode',None)} "
                                    f"stdout={stdout} stderr={stderr}"
                                )

                            # Load resulting gamma map via SimpleITK (avoid MRML access in worker thread)
                            import SimpleITK as sitk  # noqa: N813

                            img = sitk.ReadImage(str(gamma_path))
                            # Write gamma image into metrics folder near reference dose (if available)
                            try:
                                out_dir = job.get("_metrics_output_dir")
                                out_name = job.get("out_name") or "metrics"
                                if out_dir:
                                    out_gamma = os.path.join(out_dir, f"gamma_{out_name}.nii.gz")
                                    sitk.WriteImage(img, out_gamma, True)
                                    job["_gamma_file_path"] = out_gamma
                                    job["_gamma_node_name"] = f"gamma_{out_name}"
                            except Exception:
                                pass

                            arr = sitk.GetArrayFromImage(img)
                            gamma_arr = np.asarray(arr).astype(np.float32)
                            return gamma_arr

                        finally:
                            # Cleanup temp files
                            try:
                                import shutil as _sh

                                _sh.rmtree(td, ignore_errors=True)
                            except Exception:
                                pass

                    def _gamma_done(gamma_arr):
                        job2 = self._active_job
                        if job2 is None:
                            return
                        if _is_cancelled():
                            _cancel_finish()
                            return
                        job2["gamma_arr"] = gamma_arr
                        # Import gamma image into Slicer and place under same child folder as metrics table
                        try:
                            gamma_path = job2.get("_gamma_file_path")
                            if gamma_path:
                                gamma_node = slicer.util.loadVolume(str(gamma_path))
                                if isinstance(gamma_node, (list, tuple)):
                                    gamma_node = gamma_node[0] if gamma_node else None
                                if gamma_node is not None:
                                    try:
                                        node_name = job2.get("_gamma_node_name")
                                        if node_name:
                                            gamma_node.SetName(node_name)
                                    except Exception:
                                        pass
                                    folder_item_id = job2.get("_metrics_folder_item_id")
                                    if folder_item_id:
                                        self._ensure_node_in_sh_folder(gamma_node, folder_item_id)
                        except Exception:
                            logger.exception("Failed to import gamma image into Slicer")
                        self._set_progress(40, visible=True)
                        _start_per_segment_loop()

                    def _gamma_err(exc):
                        if _is_cancelled():
                            _cancel_finish()
                            return
                        _fail(f"Gamma failed: {exc}")

                    self._run_in_thread(_gamma_fn, _gamma_done, _gamma_err, poll_ms=150)
                else:
                    _start_per_segment_loop()

            def _arrays_error(exc):
                _fail(f"Failed to read arrays: {exc}")

            # Load arrays off the UI thread to keep the interface responsive on large volumes.
            self._run_in_thread(_load_arrays, _after_arrays_loaded, _arrays_error, poll_ms=150)
            return

        def _start_per_segment_loop():
            job = self._active_job
            if job is None:
                return
            if _is_cancelled():
                _cancel_finish()
                return

            seg_ids = list(job.get("selected_seg_ids", []))
            total = max(1, len(seg_ids))
            job["_seg_index"] = 0

            self._set_status("Computing per-segment metrics…")
            self._set_progress(40, visible=True)

            def _one_segment():
                job2 = self._active_job
                if job2 is None:
                    return
                if _is_cancelled():
                    _cancel_finish()
                    return
                i = int(job2.get("_seg_index", 0))
                if i >= len(seg_ids):
                    _build_table_and_finish()
                    return

                seg_id = seg_ids[i]
                mask = self.export_segment_mask(job2["seg_node"], seg_id, job2["out_dose_node"])
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
                        ref_vals = job2["_ref_arr"][mask]
                        mae = float(np.mean(np.abs(out_vals - ref_vals)))

                    unc_mean = np.nan
                    if job2.get("_unc_arr") is not None:
                        unc_mean = float(np.mean(job2["_unc_arr"][mask]))

                    gamma_pr = np.nan
                    if job2.get("gamma_arr") is not None:
                        try:
                            g = job2["gamma_arr"]
                            # Ensure mask and gamma shapes match; avoid accidental broadcasting
                            try:
                                if mask.shape != g.shape:
                                    logger.debug(
                                        "mask.shape=%s gamma.shape=%s -- shapes mismatch (will attempt alignment)",
                                        getattr(mask, "shape", None),
                                        getattr(g, "shape", None),
                                    )
                            except Exception:
                                pass
                            valid = mask & np.isfinite(g)
                            passed = int(np.count_nonzero(valid & (g <= 1.0)))
                            denom = int(np.count_nonzero(valid))
                            logger.debug(
                                "Segment %s: valid=%d passed=%d denom=%d",
                                seg_id,
                                int(np.count_nonzero(valid)),
                                passed,
                                denom,
                            )
                            # log max abs diff between ref/out in this segment to help diagnose trivial pass
                            try:
                                maxdiff = float(np.nanmax(np.abs(job2["_ref_arr"][mask] - job2["_out_arr"][mask])))
                                logger.debug("Segment %s max abs diff ref/out = %g", seg_id, maxdiff)
                            except Exception:
                                pass
                            if denom > 0:
                                gamma_pr = 100.0 * passed / float(denom)
                        except Exception:
                            logger.exception("Failed computing per-segment gamma pass rate")
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
            if _is_cancelled():
                _cancel_finish()
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
            if _is_cancelled():
                _cancel_finish()
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
                _fail(f"Reference resample failed: {exc}")

            job["_cli_node"] = self._run_cli_async(slicer.modules.resamplescalarvectordwivolume, params, _done, _err)

        def _start_resample_unc():
            job = self._active_job
            if job is None:
                return
            if _is_cancelled():
                _cancel_finish()
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
                _fail(f"Uncertainty resample failed: {exc}")

            job["_cli_node"] = self._run_cli_async(slicer.modules.resamplescalarvectordwivolume, params, _done, _err)

        try:
            QTimer.singleShot(0, _start_resample_ref)
        except Exception:
            _start_resample_ref()
        return
