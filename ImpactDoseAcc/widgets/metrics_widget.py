import os
import threading
from uuid import uuid4
from typing import Optional
import inspect

import numpy as np
import slicer
import vtk
from qt import QVBoxLayout, QCheckBox, QMessageBox, QTimer
import logging
import sys
from datetime import datetime

import importlib.util
from pathlib import Path

os.environ["NUMBA_THREADING_LAYER"] = "omp"

base_path = Path(__file__).resolve().parent / "base_widget.py"
spec = importlib.util.spec_from_file_location("impactdoseacc_base_widget", str(base_path))
base_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_mod)  # type: ignore
BaseImpactWidget = getattr(base_mod, "BaseImpactWidget")

logging.basicConfig(
    level=(logging.DEBUG),   # DEBUG | INFO | WARNING
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("pymedphys").setLevel(logging.DEBUG)
logging.getLogger("pymedphys.gamma").setLevel(logging.DEBUG)


def _available_memory_bytes() -> Optional[int]:
    """Return available system memory in bytes.

    Tries psutil first (if present), then falls back to Linux /proc/meminfo.
    """
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    # Linux fallback
    try:
        with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("MemAvailable:"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    kb = int(parts[1])
                    return kb * 1024
    except Exception:
        pass

    return None


def _ram_available_bytes(fraction: float = 0.8) -> Optional[int]:
    """Compute ram_available for pymedphys.gamma in bytes as fraction of free memory."""
    try:
        frac = float(fraction)
    except Exception:
        frac = 0.8
    frac = max(0.0, min(1.0, frac))

    avail_b = _available_memory_bytes()
    if avail_b is None:
        return None
    try:
        return int(frac * float(avail_b))
    except Exception:
        return None


class MetricsEvaluationWidget(BaseImpactWidget):
    """UI widget for Phase 3: Metrics & Evaluation."""

    def __init__(self, logic):
        super().__init__(logic)
        self._ref_dose_node_by_index = {}
        self._out_dose_node_by_index = {}
        self._unc_node_by_index = {}
        self._segment_checkbox_by_id = {}
        self._active_job = None
        self._pymedphys_preheated = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        ui_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "../Resources/UI/MetricsWidget.ui"))
        ui_widget = slicer.util.loadUI(ui_path)
        self.ui = slicer.util.childWidgetVariables(ui_widget)
        self._root_widget = ui_widget

        # Bind widgets
        self.ref_dose_combo = self._w("ref_dose_combo")
        self.out_dose_combo = self._w("out_dose_combo")
        self.unc_combo = self._w("unc_combo")
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
            try:
                self.seg_selector.nodeTypes = ["vtkMRMLSegmentationNode"]
                self.seg_selector.selectNodeUponCreation = False
                self.seg_selector.addEnabled = False
                self.seg_selector.removeEnabled = False
                self.seg_selector.noneEnabled = True
                self.seg_selector.showHidden = False
                self.seg_selector.setMRMLScene(slicer.mrmlScene)
            except Exception:
                pass

        # Buttons & signals
        self._btn("refresh_btn", self._refresh_lists)
        if self.run_btn is not None:
            try:
                self.run_btn.clicked.connect(self._on_compute_metrics)
            except Exception:
                pass
        if self.seg_selector is not None:
            try:
                self.seg_selector.currentNodeChanged.connect(self._on_segmentation_changed)
            except Exception:
                pass

        # Checkbox defaults (gamma off by default)
        for cb in (
            self.cb_dose_mean,
            self.cb_dose_minmax_3sigma,
            self.cb_err_mae,
            self.cb_unc_mean,
        ):
            try:
                if cb is not None:
                    cb.setChecked(True)
            except Exception:
                pass
        try:
            if self.cb_gamma_pr is not None:
                self.cb_gamma_pr.setChecked(False)
        except Exception:
            pass

        def _sync_gamma_params_visibility():
            try:
                if self._gamma_params_widget is not None and self.cb_gamma_pr is not None:
                    self._gamma_params_widget.setVisible(bool(self.cb_gamma_pr.isChecked()))
            except Exception:
                pass

        if self.cb_gamma_pr is not None:
            try:
                self.cb_gamma_pr.toggled.connect(_sync_gamma_params_visibility)
            except Exception:
                pass
        _sync_gamma_params_visibility()

        if self.output_name_edit is not None:
            try:
                self.output_name_edit.setText(self._generate_default_output_name(prefix="metrics"))
            except Exception:
                pass

        if self.progress_bar is not None:
            try:
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(0)
                self.progress_bar.setVisible(False)
            except Exception:
                pass

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
                    w.setParent(None)
        except Exception:
            pass

    # Preheat removed — do not perform JIT warm-up automatically from the module.

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

    def _set_ui_busy(self, busy: bool) -> None:
        try:
            # Keep run button enabled while busy so it can act as a Stop button.
            self.run_btn.setEnabled(True)
            try:
                self.run_btn.setText("Stop" if bool(busy) else "Compute metrics")
            except Exception:
                pass
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

    def _run_cli_async(self, cli_module, params: dict, on_done, on_error):
        """Run a Slicer CLI without blocking the UI."""
        try:
            cli_node = slicer.cli.run(cli_module, None, params, wait_for_completion=False)
        except Exception as exc:
            on_error(exc)
            return None

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
            return None

        return cli_node

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

        out_name = self._line_edit_text(self.output_name_edit).strip() or self._generate_default_output_name(prefix="metrics")
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
            "cancelled": False,
            "_cli_node": None,
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

            try:
                # Avoid forcing float64 + copies: this can double memory and slow things down.
                # Gamma/metrics computations work fine with float32 in practice.
                out_arr = np.asarray(slicer.util.arrayFromVolume(job["out_dose_node"]), dtype=np.float32)
                ref_arr = np.asarray(slicer.util.arrayFromVolume(job["ref_eval_node"]), dtype=np.float32)
                unc_arr = None
                if job.get("unc_eval_node") is not None:
                    unc_arr = np.asarray(slicer.util.arrayFromVolume(job["unc_eval_node"]), dtype=np.float32)
            except Exception as exc:
                _fail(f"Failed to read arrays: {exc}")
                return

            job["_out_arr"] = out_arr
            job["_ref_arr"] = ref_arr
            job["_unc_arr"] = unc_arr

            # Pre-compute ROI bbox on UI thread (uses MRML).
            # Segmentation is mandatory for metrics, so computing this consistently is safe and
            # keeps behavior stable if Gamma is toggled on/off between runs.
            try:
                distance_mm_threshold = float(self._float_from_line_edit(self.gamma_dist_mm_edit, 2.0))
            except Exception:
                distance_mm_threshold = 2.0
            bbox = None
            try:
                bbox = self._segments_bbox(
                    job["seg_node"],
                    job.get("selected_seg_ids", []),
                    job["out_dose_node"],
                    margin_mm=float(distance_mm_threshold),
                )
            except Exception:
                bbox = None
            job["_gamma_bbox"] = bbox

            # Pre-compute RAM budget for PyMedPhys (bytes). Keep it stable during one run.
            job["_gamma_ram_available_bytes"] = _ram_available_bytes(0.8)
            # Gamma params/import preflight only when Gamma is enabled.
            if self.cb_gamma_pr.isChecked():
                if job.get("_gamma_bbox") is None:
                    _fail(
                        "Gamma requires a valid ROI bbox around the selected segments, but it could not be computed. "
                        "Check that selected segments intersect the output dose volume."
                    )
                    return

                # Pre-compute gamma parameters on UI thread (avoid touching Qt widgets from worker thread).
                try:
                    dose_percent_threshold = float(self._float_from_line_edit(self.gamma_dose_percent_edit, 2.0))
                except Exception:
                    dose_percent_threshold = 2.0
                try:
                    lower_percent_dose_cutoff = float(self._float_from_line_edit(self.gamma_low_cutoff_edit, 30.0))
                except Exception:
                    lower_percent_dose_cutoff = 30.0
                try:
                    local_gamma = bool(self._gamma_mode_is_local())
                except Exception:
                    local_gamma = False

                job["_gamma_params"] = {
                    "dose_percent_threshold": dose_percent_threshold,
                    "distance_mm_threshold": distance_mm_threshold,
                    "lower_percent_dose_cutoff": lower_percent_dose_cutoff,
                    "local_gamma": local_gamma,
                    "interp_fraction": 3,
                    "max_gamma": 1,
                }

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
                    import pymedphys

                    try:
                        sx, sy, sz = (float(v) for v in job["ref_eval_node"].GetSpacing())
                    except Exception:
                        sx, sy, sz = (2.0, 2.0, 2.0)

                    params = job.get("_gamma_params") or {}
                    dose_percent_threshold = float(params.get("dose_percent_threshold", 2.0))
                    distance_mm_threshold = float(params.get("distance_mm_threshold", 2.0))
                    lower_percent_dose_cutoff = float(params.get("lower_percent_dose_cutoff", 30.0))
                    local_gamma = bool(params.get("local_gamma", False))
                    interp_fraction = int(params.get("interp_fraction", 3))
                    max_gamma = float(params.get("max_gamma", 2))

                    z0, z1, y0, y1, x0, x1 = job.get("_gamma_bbox")
                    # Ensure contiguous crops for Numba kernels (can have a big perf impact).
                    ref_crop = np.ascontiguousarray(job["_ref_arr"][z0:z1, y0:y1, x0:x1], dtype=np.float32)
                    out_crop = np.ascontiguousarray(job["_out_arr"][z0:z1, y0:y1, x0:x1], dtype=np.float32)
                    # Build only the cropped axes to reduce allocations.
                    zc = (np.arange(z0, z1, dtype=np.float32) * sz)
                    yc = (np.arange(y0, y1, dtype=np.float32) * sy)
                    xc = (np.arange(x0, x1, dtype=np.float32) * sx)
                    axes_crop = (zc, yc, xc)
                    gamma_fn = pymedphys.gamma

                    kwargs = {
                        "dose_percent_threshold": dose_percent_threshold,
                        "distance_mm_threshold": distance_mm_threshold,
                        "lower_percent_dose_cutoff": lower_percent_dose_cutoff,
                        "interp_fraction": interp_fraction,
                        "max_gamma": max_gamma,
                        "local_gamma": local_gamma,
                        "skip_once_passed": True,
                        "ram_available": job.get("_gamma_ram_available_bytes"),
                    }

                    # Match the faster in-script defaults when available.
                    try:
                        sig = inspect.signature(gamma_fn)
                        if "interp_algo" in sig.parameters:
                            kwargs["interp_algo"] = "pymedphys"
                        if "quiet" in sig.parameters:
                            kwargs["quiet"] = True
                    except Exception:
                        pass

                    return gamma_fn(
                        axes_crop,
                        ref_crop,
                        axes_crop,
                        out_crop,
                        **kwargs,
                    )

                def _gamma_done(gamma_arr):
                    job2 = self._active_job
                    if job2 is None:
                        return
                    if _is_cancelled():
                        _cancel_finish()
                        return
                    job2["gamma_arr"] = gamma_arr
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
                        ref_vals = job2["_ref_arr"][mask]
                        mae = float(np.mean(np.abs(out_vals - ref_vals)))

                    unc_mean = np.nan
                    if job2.get("_unc_arr") is not None:
                        unc_mean = float(np.mean(job2["_unc_arr"][mask]))

                    gamma_pr = np.nan
                    if job2.get("gamma_arr") is not None:
                        try:
                            g = job2["gamma_arr"]
                            z0, z1, y0, y1, x0, x1 = job2.get("_gamma_bbox")
                            m = mask[z0:z1, y0:y1, x0:x1]
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
