import logging
import os
import threading
from functools import partial
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

    def __init__(self):
        super().__init__()
        self._ref_dose_node_by_index = {}
        self._out_dose_node_by_index = {}
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
        self._segments_scroll_layout = self._layout("segments_scroll_content")
        self._gamma_params_widget = self._w("gamma_params_widget")

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

        while layout.count():
            item = layout.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                try:
                    w.setParent(None)
                    w.deleteLater()
                except Exception:
                    logger.exception("Failed to remove widget from layout")

    def _on_segmentation_changed(self, seg_node) -> None:
        self._segment_checkbox_by_id = {}
        self._clear_layout(self._segments_scroll_layout)

        if seg_node is None:
            self._segments_group.setEnabled(False)
            return
        seg = seg_node.GetSegmentation()

        if seg is None:
            self._segments_group.setEnabled(False)
            return

        self._segments_group.setEnabled(True)
        n = seg.GetNumberOfSegments()

        for i in range(n):
            seg_id = seg.GetNthSegmentID(i)
            seg_obj = seg.GetSegment(seg_id)
            seg_name = seg_obj.GetName() if seg_obj is not None else seg_id
            cb = QCheckBox(str(seg_name))
            cb.setChecked(True)
            self._segment_checkbox_by_id[seg_id] = cb
            self._segments_scroll_layout.addWidget(cb)

        self._segments_scroll_layout.addStretch()

    def _refresh_lists(self) -> None:
        """Refresh filtered lists for reference/output dose and uncertainty."""
        if slicer.mrmlScene is None:
            return

        # Keep current selections by node ID if possible.
        ref_prev = None
        out_prev = None
        ref_prev = self._ref_dose_node_by_index.get(self._combo_current_index(self.ref_dose_combo), None)
        out_prev = self._out_dose_node_by_index.get(self._combo_current_index(self.out_dose_combo), None)

        def node_id(n):
            return n.GetID() if n is not None and hasattr(n, "GetID") else None

        ref_prev_id = node_id(ref_prev)
        out_prev_id = node_id(out_prev)

        self.ref_dose_combo.blockSignals(True)
        self.out_dose_combo.blockSignals(True)

        self.ref_dose_combo.clear()
        self.out_dose_combo.clear()

        self._ref_dose_node_by_index = {0: None}
        self._out_dose_node_by_index = {0: None}

        self.ref_dose_combo.addItem("[Select reference dose]")
        self.out_dose_combo.addItem("[Select output dose]")

        # Filter volumes by name (shared for ref/out).
        volumes = list(slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"))

        dose_entries = []
        for n in volumes:
            if n is None:
                continue
            if not self._is_name_match(n, "dose") or self._is_name_match(n, "uncertainty"):
                continue
            name = self._safe_node_name(n)
            dose_entries.append((name.lower(), name, n))
        dose_entries.sort(key=lambda t: t[0])

        ref_match_index = 0
        out_match_index = 0

        idx = 1
        for _, name, n in dose_entries:
            self.ref_dose_combo.addItem(name)
            self._ref_dose_node_by_index[idx] = n
            if ref_prev_id and node_id(n) == ref_prev_id:
                ref_match_index = idx
            idx += 1

        idx = 1
        for _, name, n in dose_entries:
            self.out_dose_combo.addItem(name)
            self._out_dose_node_by_index[idx] = n
            if out_prev_id and node_id(n) == out_prev_id:
                out_match_index = idx
            idx += 1

        if ref_match_index:
            self.ref_dose_combo.setCurrentIndex(ref_match_index)
        if out_match_index:
            self.out_dose_combo.setCurrentIndex(out_match_index)

        self._on_out_dose_changed()
        self.ref_dose_combo.blockSignals(False)
        self.out_dose_combo.blockSignals(False)

    def _selected_ref_dose_node(self):
        return self._ref_dose_node_by_index.get(self._combo_current_index(self.ref_dose_combo), None)

    def _selected_out_dose_node(self):
        return self._out_dose_node_by_index.get(self._combo_current_index(self.out_dose_combo), None)

    def _selected_unc_node(self):
        if getattr(self, "cb_unc_out", None) is not None and bool(self.cb_unc_out.isChecked()):
            return self._find_uncertainty_in_same_folder(self._selected_out_dose_node())
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

        in_img = input_node.GetImageData()
        ref_img = reference_node.GetImageData()
        in_dims = in_img.GetDimensions() if in_img is not None else None
        ref_dims = ref_img.GetDimensions() if ref_img is not None else None
        if in_dims is None or ref_dims is None or in_dims != ref_dims:
            return True

        # Compare IJK->RAS matrices.
        m_in = vtk.vtkMatrix4x4()
        m_ref = vtk.vtkMatrix4x4()
        input_node.GetIJKToRASMatrix(m_in)
        reference_node.GetIJKToRASMatrix(m_ref)
        for r in range(4):
            for c in range(4):
                if abs(m_in.GetElement(r, c) - m_ref.GetElement(r, c)) > 1e-6:
                    return True
        return False

    def _float_from_line_edit(self, line_edit, default: float) -> float:
        if line_edit is None:
            return float(default)

        text = self._line_edit_text(line_edit).strip()
        if text == "":
            return float(default)
        return float(text)

    def _gamma_mode_is_local(self) -> bool:
        # currentText may be exposed as a property or a callable; normalize before checking
        if self.gamma_mode_combo is None:
            return False

        val = self.gamma_mode_combo.currentText
        txt = val() if callable(val) else val
        return str(txt).lower().startswith("local")

    def _create_or_get_table_node(self, name: str):
        if slicer.mrmlScene is None:
            return None
        node = slicer.mrmlScene.GetFirstNodeByName(name)
        if node is not None and hasattr(node, "IsA") and node.IsA("vtkMRMLTableNode"):
            return node
        return slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", name)

    def _ensure_node_in_sh_folder(self, node, folder_item_id):
        if slicer.mrmlScene is None or node is None or not folder_item_id:
            return
        sh_node = self._get_sh_node()
        if sh_node is None:
            return
        item_id = sh_node.GetItemByDataNode(node)
        if item_id == 0:
            item_id = sh_node.CreateItem(folder_item_id, node)

        if item_id:
            sh_node.SetItemParent(item_id, folder_item_id)

    def _get_or_create_child_folder_item(self, reference_node, folder_name: str):
        if slicer.mrmlScene is None or reference_node is None:
            return None
        sh_node = self._get_sh_node()
        if sh_node is None:
            return None
        ref_item_id = sh_node.GetItemByDataNode(reference_node)

        if not ref_item_id:
            return None

        children = vtk.vtkIdList()
        sh_node.GetItemChildren(ref_item_id, children, False)

        for i in range(children.GetNumberOfIds()):
            child_id = children.GetId(i)
            if sh_node.GetItemName(child_id) == folder_name and sh_node.GetItemDataNode(child_id) is None:
                return child_id

        try:
            return sh_node.CreateFolderItem(ref_item_id, folder_name)
        except Exception:
            return None

    def _get_output_dir_from_ref(self, reference_node, child_folder: str):
        if reference_node is None:
            return None
        storage = reference_node.GetStorageNode()

        if storage is None:
            return None
        ref_path = storage.GetFileName()

        if not ref_path:
            return None

        base_dir = os.path.dirname(ref_path)
        out_dir = os.path.join(base_dir, child_folder)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

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
            cancel_fn = getattr(slicer.cli, "cancel", None)
            if callable(cancel_fn):
                cancel_fn(cli_node)
            else:
                cli_node.Cancel()

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
                QMessageBox.warning(self, "Compute Error", str(message))

    def _metrics_cleanup_temp_nodes(self) -> None:
        job = self._active_job
        if job is None:
            return
        for tn in job.get("temp_nodes", []):
            if tn is not None and slicer.mrmlScene is not None and tn.GetScene() == slicer.mrmlScene:
                slicer.mrmlScene.RemoveNode(tn)

    def _metrics_fail(self, msg: str) -> None:
        job = self._active_job
        if job is not None and bool(job.get("cancelled", False)):
            self._metrics_cleanup_temp_nodes()
            self._finish_job(True, "Cancelled.")
            return
        self._metrics_cleanup_temp_nodes()
        self._finish_job(False, msg)

    def _metrics_is_cancelled(self) -> bool:
        j = self._active_job
        if j is None:
            return True
        return bool(j.get("cancelled", False))

    def _metrics_cancel_finish(self) -> None:
        self._metrics_cleanup_temp_nodes()
        self._finish_job(True, "Cancelled.")

    def _metrics_resample_done(self, progress_value: int, next_step) -> None:
        self._set_progress(progress_value, visible=True)
        if callable(next_step):
            next_step()

    def _metrics_resample_err(self, label: str, exc) -> None:
        self._metrics_fail(f"{label} resample failed: {exc}")

    def _metrics_start_resample(self, input_key: str, eval_key: str, label: str, on_done, progress_value: int) -> None:
        job = self._active_job
        if job is None:
            return
        if self._metrics_is_cancelled():
            self._metrics_cancel_finish()
            return
        src_node = job.get(input_key)
        if src_node is None:
            if callable(on_done):
                on_done()
            return
        needs = self._needs_resample_to_reference(src_node, job["out_dose_node"])
        if not needs:
            if callable(on_done):
                on_done()
            return

        out_name_rs = f"{self._safe_node_name(src_node)}_resampled_{uuid4().hex[:6]}"
        out_node_rs = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", out_name_rs)
        out_node_rs.SetHideFromEditors(1)
        out_node_rs.SetSelectable(0)
        out_node_rs.SetSaveWithScene(0)

        job[eval_key] = out_node_rs
        job["temp_nodes"].append(out_node_rs)
        params = {
            "inputVolume": src_node.GetID(),
            "referenceVolume": job["out_dose_node"].GetID(),
            "outputVolume": out_node_rs.GetID(),
            "interpolationType": "linear",
        }

        job["_cli_node"] = self._run_cli_async(
            slicer.modules.resamplescalarvectordwivolume,
            params,
            partial(self._metrics_resample_done, progress_value, on_done),
            partial(self._metrics_resample_err, label),
        )

    def _metrics_start_resample_ref(self) -> None:
        self._metrics_start_resample(
            "ref_dose_node", "ref_eval_node", "Reference", self._metrics_start_resample_unc, 15
        )

    def _metrics_start_resample_unc(self) -> None:
        self._metrics_start_resample("unc_node", "unc_eval_node", "Uncertainty", self._metrics_step_after_resample, 20)

    def _metrics_start_resample_flow(self) -> None:
        self._set_status("Resampling inputs (if needed)…")
        self._set_progress(10, visible=True)
        try:
            QTimer.singleShot(0, self._metrics_start_resample_ref)
        except Exception:
            self._metrics_start_resample_ref()

    def _metrics_load_arrays(self):
        job = self._active_job
        if job is None:
            raise RuntimeError("No active job")
        # Avoid forcing float64 + copies: this can double memory and slow things down.
        # Gamma/metrics computations work fine with float32 in practice.
        out_arr = np.asarray(slicer.util.arrayFromVolume(job["out_dose_node"]), dtype=np.float32, copy=False)
        ref_arr = np.asarray(slicer.util.arrayFromVolume(job["ref_eval_node"]), dtype=np.float32, copy=False)
        unc_arr = None
        if job.get("unc_eval_node") is not None:
            unc_arr = np.asarray(slicer.util.arrayFromVolume(job["unc_eval_node"]), dtype=np.float32, copy=False)
        return out_arr, ref_arr, unc_arr

    def _metrics_arrays_error(self, exc) -> None:
        self._metrics_fail(f"Failed to read arrays: {exc}")

    def _metrics_after_arrays_loaded(self, result) -> None:
        if self._metrics_is_cancelled():
            self._metrics_cancel_finish()
            return
        job = self._active_job
        if job is None:
            return
        try:
            out_arr, ref_arr, unc_arr = result
        except Exception as exc:
            self._metrics_fail(f"Failed to read arrays: {exc}")
            return

        job["_out_arr"] = out_arr
        job["_ref_arr"] = ref_arr
        job["_unc_arr"] = unc_arr

        # Gamma params/import preflight only when Gamma is enabled.
        if self.cb_gamma_pr.isChecked():
            distance_mm_threshold = float(self._float_from_line_edit(self.gamma_dist_mm_edit, 2.0))
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

            spacing = tuple(job["ref_eval_node"].GetSpacing())
            job["_gamma_spacing"] = spacing
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

            # Gamma duration is hard to predict; switch to indeterminate progress.
            self._set_progress(None, visible=True)
            try:
                slicer.app.processEvents()
            except Exception:
                pass

            self._set_status("Computing gamma…")
            try:
                gamma_arr = self._metrics_gamma_fn()
            except Exception as exc:
                self._metrics_gamma_err(exc)
                return
            self._metrics_gamma_done(gamma_arr)
            return

        self._metrics_start_per_segment_loop()

    def _metrics_step_after_resample(self) -> None:
        job = self._active_job
        if job is None:
            return
        if self._metrics_is_cancelled():
            self._metrics_cancel_finish()
            return

        self._set_status("Loading arrays…")
        self._set_progress(25, visible=True)
        # Load arrays off the UI thread to keep the interface responsive on large volumes.
        self._run_in_thread(
            self._metrics_load_arrays, self._metrics_after_arrays_loaded, self._metrics_arrays_error, 150
        )

    def _metrics_tick_gamma_progress(self) -> None:
        j = self._active_job
        if j is None:
            return
        if j.get("gamma_arr") is not None:
            return
        p = int(j.get("_gamma_prog", 25))
        p = min(39, p + 1)
        j["_gamma_prog"] = p
        self._set_progress(p, visible=True)
        try:
            QTimer.singleShot(200, self._metrics_tick_gamma_progress)
        except Exception:
            pass

    def _metrics_gamma_fn(self):
        job = self._active_job
        if job is None:
            raise RuntimeError("No active job")
        ref_node = job.get("ref_eval_node")
        cmp_node = job.get("out_dose_node")
        if ref_node is None or cmp_node is None:
            raise RuntimeError("Missing dose nodes for gamma computation")

        try:
            dose_logic = slicer.modules.dosecomparison.logic()
        except Exception:
            dose_logic = None
        if dose_logic is None:
            raise RuntimeError("DoseComparison module is not available (SlicerRT)")

        dc_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLDoseComparisonNode")
        if dc_node is None:
            raise RuntimeError("Failed to create DoseComparison node")
        job.setdefault("temp_nodes", []).append(dc_node)

        out_name = job.get("out_name") or "metrics"
        gamma_name = job.get("_gamma_node_name") or f"gamma_{out_name}"
        job["_gamma_node_name"] = gamma_name
        gamma_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", gamma_name)
        if gamma_node is None:
            raise RuntimeError("Failed to create gamma volume node")
        job["_gamma_node"] = gamma_node

        try:
            if gamma_node.GetDisplayNode() is None:
                slicer.modules.volumes.logic().CreateDefaultVolumeDisplayNodes(gamma_node)
        except Exception:
            pass

        try:
            ct_id = dose_logic.GetDefaultGammaColorTableNodeId()
        except Exception:
            ct_id = None
        if ct_id:
            try:
                dn = gamma_node.GetDisplayNode()
                if dn is not None:
                    dn.SetAndObserveColorNodeID(ct_id)
            except Exception:
                pass

        dc_node.SetAndObserveReferenceDoseVolumeNode(ref_node)
        dc_node.SetAndObserveCompareDoseVolumeNode(cmp_node)
        dc_node.SetAndObserveGammaVolumeNode(gamma_node)

        params = job.get("_gamma_params", {})
        dc_node.SetDoseDifferenceTolerancePercent(float(params.get("dose_percent_threshold", 2.0)))
        dc_node.SetDtaDistanceToleranceMm(float(params.get("distance_mm_threshold", 2.0)))
        dc_node.SetAnalysisThresholdPercent(float(params.get("lower_percent_dose_cutoff", 30.0)))
        dc_node.SetLocalDoseDifference(1 if bool(params.get("local_gamma", False)) else 0)
        dc_node.SetMaximumGamma(1.5)
        try:
            dc_node.SetDoseThresholdOnReferenceOnly(1)
        except Exception:
            pass
        try:
            dc_node.SetUseMaximumDose(1)
        except Exception:
            pass
        try:
            ref_max = float(np.nanmax(job.get("_ref_arr")))
            if np.isfinite(ref_max) and ref_max > 0:
                dc_node.SetReferenceDoseGy(ref_max)
        except Exception:
            pass

        dose_logic.ComputeGammaDoseDifference(dc_node)

        try:
            job["_gamma_pass_fraction"] = float(dc_node.GetPassFractionPercent())
        except Exception:
            pass

        gamma_arr = np.asarray(slicer.util.arrayFromVolume(gamma_node), dtype=np.float32, copy=False)
        return gamma_arr

    def _metrics_gamma_done(self, gamma_arr) -> None:
        job2 = self._active_job
        if job2 is None:
            return
        if self._metrics_is_cancelled():
            self._metrics_cancel_finish()
            return
        job2["gamma_arr"] = gamma_arr
        # Place gamma node under same child folder as metrics table
        try:
            gamma_node = job2.get("_gamma_node")
            if gamma_node is None:
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
            logger.exception("Failed to place gamma volume into subject hierarchy")
        self._set_progress(40, visible=True)
        self._metrics_start_per_segment_loop()

    def _metrics_gamma_err(self, exc) -> None:
        if self._metrics_is_cancelled():
            self._metrics_cancel_finish()
            return
        self._metrics_fail(f"Gamma failed: {exc}")

    def _metrics_start_per_segment_loop(self) -> None:
        job = self._active_job
        if job is None:
            return
        if self._metrics_is_cancelled():
            self._metrics_cancel_finish()
            return

        seg_ids = list(job.get("selected_seg_ids", []))
        job["_seg_ids"] = seg_ids
        job["_seg_total"] = max(1, len(seg_ids))
        job["_seg_index"] = 0

        self._set_status("Computing per-segment metrics…")
        self._set_progress(40, visible=True)
        try:
            QTimer.singleShot(0, self._metrics_one_segment_tick)
        except Exception:
            self._metrics_one_segment_tick()

    def _metrics_one_segment_tick(self) -> None:
        job2 = self._active_job
        if job2 is None:
            return
        if self._metrics_is_cancelled():
            self._metrics_cancel_finish()
            return
        seg_ids = list(job2.get("_seg_ids") or job2.get("selected_seg_ids", []))
        total = max(1, len(seg_ids))
        i = int(job2.get("_seg_index", 0))
        if i >= len(seg_ids):
            self._metrics_build_table_and_finish()
            return

        seg_id = seg_ids[i]
        mask = self.export_segment_mask(job2["seg_node"], seg_id, job2["out_dose_node"])
        count = int(np.count_nonzero(mask)) if mask is not None else 0

        seg = job2["seg_node"].GetSegmentation()
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
                g = job2["gamma_arr"]
                if mask.shape != g.shape:
                    logger.debug(
                        "mask.shape=%s gamma.shape=%s -- shapes mismatch (will attempt alignment)",
                        getattr(mask, "shape", None),
                        getattr(g, "shape", None),
                    )
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
                maxdiff = float(np.nanmax(np.abs(job2["_ref_arr"][mask] - job2["_out_arr"][mask])))
                logger.debug("Segment %s max abs diff ref/out = %g", seg_id, maxdiff)

                if denom > 0:
                    gamma_pr = 100.0 * passed / float(denom)

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
            QTimer.singleShot(0, self._metrics_one_segment_tick)
        except Exception:
            self._metrics_one_segment_tick()

    def _metrics_fmt_float(self, value) -> str:
        try:
            v = float(value)
        except Exception:
            return ""
        return "" if not np.isfinite(v) else f"{v:.6g}"

    def _metrics_add_table_col(self, table, name: str) -> None:
        col = vtk.vtkStringArray()
        col.SetName(name)
        table.AddColumn(col)

    def _metrics_build_table_and_finish(self) -> None:
        job = self._active_job
        if job is None:
            return
        if self._metrics_is_cancelled():
            self._metrics_cancel_finish()
            return
        self._set_status("Building table…")
        self._set_progress(95, visible=True)

        per_seg = job.get("per_seg", {})
        table_node = job["table_node"]
        table = table_node.GetTable()
        table.Initialize()

        name_col = vtk.vtkStringArray()
        name_col.SetName("Segment")
        table.AddColumn(name_col)

        if self.cb_dose_mean.isChecked():
            self._metrics_add_table_col(table, "Dose_mean")
        if self.cb_dose_minmax_3sigma.isChecked():
            self._metrics_add_table_col(table, "Dose_min")
            self._metrics_add_table_col(table, "Dose_max")
        if self.cb_err_mae.isChecked():
            self._metrics_add_table_col(table, "Err_MAE")
        if self.cb_unc_mean.isChecked():
            self._metrics_add_table_col(table, "Unc_mean")
        if self.cb_gamma_pr.isChecked():
            self._metrics_add_table_col(table, "Gamma_PR_%")

        rows = []
        for seg_id in job.get("selected_seg_ids", []):
            d = per_seg.get(seg_id, {"name": seg_id})
            seg_name = str(d.get("name", seg_id))

            values = []
            if self.cb_dose_mean.isChecked():
                values.append(self._metrics_fmt_float(d.get("mean", np.nan)))
            if self.cb_dose_minmax_3sigma.isChecked():
                mean_v = float(d.get("mean", np.nan))
                unc_v = float(d.get("unc_mean", np.nan))
                if np.isfinite(mean_v) and np.isfinite(unc_v):
                    values.append(self._metrics_fmt_float(max(0.0, mean_v - 3.0 * unc_v)))
                    values.append(self._metrics_fmt_float(mean_v + 3.0 * unc_v))
                else:
                    values.append("")
                    values.append("")
            if self.cb_err_mae.isChecked():
                values.append(self._metrics_fmt_float(d.get("mae", np.nan)))
            if self.cb_unc_mean.isChecked():
                values.append(self._metrics_fmt_float(d.get("unc_mean", np.nan)))
            if self.cb_gamma_pr.isChecked():
                values.append(self._metrics_fmt_float(d.get("gamma_pr", np.nan)))

            if any(v != "" for v in values):
                rows.append((seg_name, values))

        table.SetNumberOfRows(len(rows))
        for r, (seg_name, values) in enumerate(rows):
            table.SetValue(r, 0, seg_name)
            for i, v in enumerate(values, start=1):
                table.SetValue(r, i, v)

        table_node.Modified()

        self._metrics_cleanup_temp_nodes()
        try:
            slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpTableView)
        except Exception:
            pass
        self._set_progress(100, visible=False)
        self._finish_job(True, f"Done. Output table: {job['out_name']}")

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

        selected_seg_ids = [
            sid for sid, cb in self._segment_checkbox_by_id.items() if cb is not None and cb.isChecked()
        ]
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

        self._metrics_start_resample_flow()
