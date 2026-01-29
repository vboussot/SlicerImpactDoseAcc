from uuid import uuid4

import io
import numpy as np
import os
import slicer
import vtk
from qt import QVBoxLayout, QCheckBox, QMessageBox, QTimer
import importlib.util
from pathlib import Path

base_path = Path(__file__).resolve().parent / "base_widget.py"
spec = importlib.util.spec_from_file_location("impactdoseacc_base_widget", str(base_path))
base_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_mod)  # type: ignore
BaseImpactWidget = getattr(base_mod, "BaseImpactWidget")


class DVHWidget(BaseImpactWidget):
    """UI widget for Phase 4: DVH.

    Features:
    - Compute cumulative DVH per selected segment for up to 2 dose volumes.
    - Filter dose candidates to volumes having 'dose' in the name.
    - Optional uncertainty traces per dose, enabled only if an uncertainty volume exists in the
        same Subject Hierarchy folder as the selected dose.
    - Output as a vtkMRMLTableNode (export-friendly) and an in-tab Plot (PlotChart).
    - Automatic binning based on max dose.
    """

    def __init__(self, logic):
        super().__init__()
        self.logic = logic
        self._dose_node_by_index_a = {0: None}
        self._dose_node_by_index_b = {0: None}
        self._segment_checkbox_by_id = {}
        self._active_job = None
        self._plot_view_node = None
        self._last_png_path = None
        self._last_out_base = None
        self._setup_ui()


    def _generate_default_output_name(self) -> str:
        return f"dvh_{uuid4().hex[:6]}"

    def _setup_ui(self) -> None:
        ui_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "../Resources/UI/DVHWidget.ui"))
        ui_widget = slicer.util.loadUI(ui_path)
        self.ui = slicer.util.childWidgetVariables(ui_widget)
        self._root_widget = ui_widget

        # Bind widgets
        self.dose_combo_a = self._w("dose_combo_a")
        self.dose_combo_b = self._w("dose_combo_b")
        self.cb_unc_a = self._w("cb_unc_a")
        self.cb_unc_b = self._w("cb_unc_b")
        self.seg_selector = self._w("seg_selector")
        self._segments_group = self._w("segments_group")
        self._segments_scroll = self._w("segments_scroll")
        self._segments_scroll_content = self._w("segments_scroll_content")
        self._segments_scroll_layout = self._layout("segments_scroll_content")
        self.output_name_edit = self._w("output_name_edit")
        self.run_btn = self._w("run_btn")
        self.status_label = self._w("status_label")
        self.progress_bar = self._w("progress_bar")
        self.plot_widget = self._w("plot_widget")
        self.legend_group = self._w("legend_group")
        self.legend_label = self._w("legend_label")

        # Configure segmentation selector
        if self.seg_selector is not None:
            try:
                self.seg_selector.nodeTypes = ["vtkMRMLSegmentationNode"]
                self.seg_selector.noneEnabled = True
                self.seg_selector.addEnabled = False
                self.seg_selector.removeEnabled = False
                self.seg_selector.setMRMLScene(slicer.mrmlScene)
            except Exception:
                pass

        # Buttons & signals
        self._btn("refresh_btn", self._refresh_dose_lists)
        if self.run_btn is not None:
            try:
                self.run_btn.clicked.connect(self._on_compute_dvh)
            except Exception:
                pass
        # show_png_btn removed
        try:
            if self.seg_selector is not None:
                self.seg_selector.currentNodeChanged.connect(self._on_segmentation_changed)
        except Exception:
            pass
        try:
            if self.dose_combo_a is not None:
                self.dose_combo_a.currentIndexChanged.connect(self._on_dose_selection_changed)
            if self.dose_combo_b is not None:
                self.dose_combo_b.currentIndexChanged.connect(self._on_dose_selection_changed)
        except Exception:
            pass

        # Defaults
        try:
            if self.cb_unc_a is not None:
                self.cb_unc_a.setChecked(False)
                self.cb_unc_a.setEnabled(False)
            if self.cb_unc_b is not None:
                self.cb_unc_b.setChecked(False)
                self.cb_unc_b.setEnabled(False)
        except Exception:
            pass
        # show_png_btn removed

        if self.output_name_edit is not None:
            try:
                self.output_name_edit.setText(self._generate_default_output_name())
            except Exception:
                pass

        if self.progress_bar is not None:
            try:
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(0)
                self.progress_bar.setVisible(False)
            except Exception:
                pass

        # Embedded plot widget configuration
        if self.plot_widget is not None:
            try:
                self.plot_widget.setMRMLScene(slicer.mrmlScene)
                self.plot_widget.setMinimumHeight(260)
            except Exception:
                pass
            try:
                self._ensure_embedded_plot_view()
            except Exception:
                pass

        layout = QVBoxLayout(self)
        layout.addWidget(ui_widget)
        self.setLayout(layout)

        self._refresh_dose_lists()
        self._on_segmentation_changed(self.seg_selector.currentNode() if self.seg_selector is not None else None)
        self._on_dose_selection_changed()
        self._update_legend(structure_items=[])

    def _set_plot_chart_on_widget(self, chart_node) -> None:
        if self.plot_widget is None or chart_node is None:
            return
        # Slicer API can differ across versions. Prefer PlotViewNode + PlotChartNodeID.
        try:
            self._ensure_embedded_plot_view()
        except Exception:
            pass

        # Attach chart to the plot view node if possible
        try:
            if self._plot_view_node is not None and hasattr(self._plot_view_node, "SetPlotChartNodeID") and hasattr(chart_node, "GetID"):
                self._plot_view_node.SetPlotChartNodeID(chart_node.GetID())
        except Exception:
            pass

        # Fallbacks (older/newer APIs)
        try:
            if hasattr(self.plot_widget, "setMRMLPlotChartNode"):
                self.plot_widget.setMRMLPlotChartNode(chart_node)
            elif hasattr(self.plot_widget, "setMRMLPlotChartNodeID") and hasattr(chart_node, "GetID"):
                self.plot_widget.setMRMLPlotChartNodeID(chart_node.GetID())
        except Exception:
            pass
        try:
            self.plot_widget.setVisible(True)
            self.plot_widget.show()
        except Exception:
            pass

    def _ensure_embedded_plot_view(self) -> None:
        if self.plot_widget is None or slicer.mrmlScene is None:
            return

        # Create (or reuse) a plot view node in the scene
        if self._plot_view_node is None:
            try:
                self._plot_view_node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLPlotViewNode", f"DVHPlotView_{uuid4().hex[:6]}"
                )
            except Exception:
                self._plot_view_node = None

        # Attach view node to widget
        if self._plot_view_node is not None:
            try:
                if hasattr(self.plot_widget, "setMRMLPlotViewNode"):
                    self.plot_widget.setMRMLPlotViewNode(self._plot_view_node)
                elif hasattr(self.plot_widget, "setMRMLPlotViewNodeID") and hasattr(self._plot_view_node, "GetID"):
                    self.plot_widget.setMRMLPlotViewNodeID(self._plot_view_node.GetID())
            except Exception:
                pass

    def _set_chart_legend_visible(self, chart_node, visible: bool) -> None:
        if chart_node is None:
            return
        v = bool(visible)
        # Try common APIs; ignore if not available.
        for meth in (
            "SetLegendVisibility",
            "SetShowLegend",
            "SetLegendVisible",
            "SetShowLegendVisibility",
        ):
            try:
                if hasattr(chart_node, meth):
                    getattr(chart_node, meth)(1 if v else 0)
                    return
            except Exception:
                continue

    def _apply_series_opacity(self, series, alpha: float) -> None:
        if series is None:
            return
        try:
            a = float(alpha)
        except Exception:
            a = 1.0
        a = max(0.0, min(1.0, a))
        for meth in ("SetOpacity", "SetLineOpacity", "SetFillOpacity", "SetBrushOpacity"):
            try:
                if hasattr(series, meth):
                    getattr(series, meth)(a)
                    return
            except Exception:
                continue

    def _rgb01_to_hex(self, rgb):
        try:
            r, g, b = rgb
            r = int(max(0, min(255, round(float(r) * 255.0))))
            g = int(max(0, min(255, round(float(g) * 255.0))))
            b = int(max(0, min(255, round(float(b) * 255.0))))
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return "#000000"

    def _update_legend(self, structure_items):
        """Update custom legend.

        structure_items: list of dicts with keys: name, color(rgb01)
        """
        try:
            structure_lines = []
            for it in structure_items or []:
                name = str(it.get("name", ""))
                hexcol = self._rgb01_to_hex(it.get("color", (0, 0, 0)))
                structure_lines.append(
                    f"<div><span style='color:{hexcol}; font-weight:600;'>━━━━</span>&nbsp;{name}</div>"
                )

            styles = (
                "<div style='margin-top:6px;'>"
                "<div><span style='color:#000; font-weight:600;'>━━━━</span>&nbsp;Dose ref </div>"
                "<div><span style='color:#000; font-weight:600;'>- - - -</span>&nbsp;Dose estimated </div>"
                "</div>"
            )

            ci = (
                "<div style='margin-top:6px;'>"
                "<div><span style='color:#000; font-weight:600;'>━━━━</span>&nbsp;Uncertainty (±3σ)</div>"
                "</div>"
            )


            html = "<div>"
            if structure_lines:
                html += "<div><b>Structures</b></div>" + "".join(structure_lines)
            html += styles + ci + "</div>"

            self.legend_label.setText(html)
        except Exception:
            pass

    def _set_status(self, text: str) -> None:
        try:
            self.status_label.setText(text or "")
        except Exception:
            pass

    def _set_progress(self, value, visible: bool = True) -> None:
        try:
            if self.progress_bar is None:
                return
            if value is None:
                self.progress_bar.setRange(0, 0)
            else:
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(int(max(0, min(100, int(value)))))
            self.progress_bar.setVisible(bool(visible))
        except Exception:
            pass

    def _set_ui_busy(self, busy: bool) -> None:
        try:
            self.run_btn.setEnabled(not bool(busy))
        except Exception:
            pass
        try:
            self.dose_combo_a.setEnabled(not bool(busy))
            self.dose_combo_b.setEnabled(not bool(busy))
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
        # Uncertainty checkboxes: always disable while running; restore eligibility after.
        if bool(busy):
            try:
                self.cb_unc_a.setEnabled(False)
                self.cb_unc_b.setEnabled(False)
            except Exception:
                pass
        else:
            try:
                self._on_dose_selection_changed()
            except Exception:
                pass


    def _selected_dose_node_a(self):
        try:
            return self._dose_node_by_index_a.get(self._combo_current_index(self.dose_combo_a), None)
        except Exception:
            return None

    def _selected_dose_node_b(self):
        try:
            return self._dose_node_by_index_b.get(self._combo_current_index(self.dose_combo_b), None)
        except Exception:
            return None

    def _refresh_dose_lists(self) -> None:
        if slicer.mrmlScene is None:
            return

        # Keep previous selections by node ID.
        prev_a = self._selected_dose_node_a()
        prev_b = self._selected_dose_node_b()
        try:
            prev_a_id = prev_a.GetID() if prev_a is not None and hasattr(prev_a, "GetID") else None
        except Exception:
            prev_a_id = None
        try:
            prev_b_id = prev_b.GetID() if prev_b is not None and hasattr(prev_b, "GetID") else None
        except Exception:
            prev_b_id = None

        try:
            self.dose_combo_a.blockSignals(True)
            self.dose_combo_b.blockSignals(True)
        except Exception:
            pass

        self.dose_combo_a.clear()
        self.dose_combo_b.clear()
        self._dose_node_by_index_a = {0: None}
        self._dose_node_by_index_b = {0: None}

        self.dose_combo_a.addItem("[Select dose A]")
        self.dose_combo_b.addItem("[None]")

        # Filter volumes by name containing 'dose'.
        try:
            volumes = list(slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"))
        except Exception:
            volumes = []

        dose_nodes = [
            n
            for n in volumes
            if n is not None
            and ("dose" in self._safe_node_name(n).lower())
            and ("uncertainty" not in self._safe_node_name(n).lower())
        ]
        dose_nodes.sort(key=lambda n: self._safe_node_name(n).lower())

        match_a = 0
        match_b = 0
        idx = 1
        for n in dose_nodes:
            name = self._safe_node_name(n)
            self.dose_combo_a.addItem(name)
            self._dose_node_by_index_a[idx] = n
            if prev_a_id and hasattr(n, "GetID") and n.GetID() == prev_a_id:
                match_a = idx
            idx += 1

        idx = 1
        for n in dose_nodes:
            name = self._safe_node_name(n)
            self.dose_combo_b.addItem(name)
            self._dose_node_by_index_b[idx] = n
            if prev_b_id and hasattr(n, "GetID") and n.GetID() == prev_b_id:
                match_b = idx
            idx += 1

        try:
            if match_a:
                self.dose_combo_a.setCurrentIndex(match_a)
        except Exception:
            pass
        try:
            if match_b:
                self.dose_combo_b.setCurrentIndex(match_b)
        except Exception:
            pass

        try:
            self.dose_combo_a.blockSignals(False)
            self.dose_combo_b.blockSignals(False)
        except Exception:
            pass

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
            n = int(seg.GetNumberOfSegments())
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
            self._segment_checkbox_by_id[str(seg_id)] = cb
            self._segments_scroll_layout.addWidget(cb)

        self._segments_scroll_layout.addStretch()

    def _selected_segment_ids(self):
        try:
            return [sid for sid, cb in self._segment_checkbox_by_id.items() if cb is not None and cb.isChecked()]
        except Exception:
            return []
    # show_png removed per user request

    def _load_png_node(self, path: str, name: str):
        if slicer.mrmlScene is None or not path or not os.path.exists(path):
            return None
        try:
            existing = slicer.mrmlScene.GetFirstNodeByName(name)
        except Exception:
            existing = None

        if existing is not None and hasattr(existing, "IsA") and existing.IsA("vtkMRMLScalarVolumeNode"):
            try:
                storage = existing.GetStorageNode()
            except Exception:
                storage = None
            if storage is not None:
                try:
                    storage.SetFileName(path)
                    storage.ReadData(existing)
                    return existing
                except Exception:
                    pass
            # Fallback: remove display/storage nodes then node to avoid VTK pipeline warnings
            try:
                if existing.GetScene() == slicer.mrmlScene:
                    try:
                        dn = existing.GetDisplayNode() if hasattr(existing, "GetDisplayNode") else None
                        if dn is not None and dn.GetScene() == slicer.mrmlScene:
                            try:
                                slicer.mrmlScene.RemoveNode(dn)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    try:
                        sn = existing.GetStorageNode() if hasattr(existing, "GetStorageNode") else None
                        if sn is not None and sn.GetScene() == slicer.mrmlScene:
                            try:
                                slicer.mrmlScene.RemoveNode(sn)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    try:
                        slicer.mrmlScene.RemoveNode(existing)
                    except Exception:
                        pass
            except Exception:
                pass

        try:
            n = slicer.util.loadVolume(path, properties={"name": name})
            return n
        except Exception:
            return None

    def _dose_base_name(self, dose_name: str) -> str:
        try:
            s = str(dose_name or "")
        except Exception:
            return ""
        for pfx in ("dose_acc_", "dose_list_", "dose_"):
            if s.startswith(pfx):
                return s[len(pfx) :]
        return ""

    def _find_uncertainty_in_same_folder(self, dose_node):
        """Return an uncertainty volume node in the same SH folder as dose_node, if any."""
        if slicer.mrmlScene is None or dose_node is None:
            return None

        shNode = self._get_sh_node()
        if shNode is None:
            return None

        try:
            dose_item = int(shNode.GetItemByDataNode(dose_node) or 0)
        except Exception:
            dose_item = 0
        if not dose_item:
            return None
        try:
            parent = int(shNode.GetItemParent(dose_item) or 0)
        except Exception:
            parent = 0
        if not parent:
            return None

        base = self._dose_base_name(self._safe_node_name(dose_node))
        preferred = []
        if base:
            preferred = [f"uncertainty_{base}", f"uncertainty_dose_{base}"]

        ids = vtk.vtkIdList()
        try:
            shNode.GetItemChildren(parent, ids)
        except Exception:
            return None

        first_unc = None
        for i in range(ids.GetNumberOfIds()):
            try:
                child = int(ids.GetId(i))
            except Exception:
                continue
            try:
                n = shNode.GetItemDataNode(child)
            except Exception:
                n = None
            if n is None or not hasattr(n, "IsA") or not n.IsA("vtkMRMLScalarVolumeNode"):
                continue
            name = self._safe_node_name(n).lower()
            if "uncertainty" not in name:
                continue
            if first_unc is None:
                first_unc = n
            for pref in preferred:
                if self._safe_node_name(n) == pref:
                    return n
        return first_unc

    def _on_dose_selection_changed(self) -> None:
        dose_a = self._selected_dose_node_a()
        dose_b = self._selected_dose_node_b()

        unc_a = self._find_uncertainty_in_same_folder(dose_a)
        unc_b = self._find_uncertainty_in_same_folder(dose_b)

        try:
            self.cb_unc_a.setEnabled(bool(unc_a is not None))
            if unc_a is None:
                self.cb_unc_a.setChecked(False)
        except Exception:
            pass
        try:
            self.cb_unc_b.setEnabled(bool(unc_b is not None))
            if unc_b is None:
                self.cb_unc_b.setChecked(False)
        except Exception:
            pass

    def _auto_bin_width_gy(self, max_dose_gy: float) -> float:
        # Minimal & robust: use 1 Gy bins.
        # If a Slicer Plot build uses point index on X, this keeps the displayed X values in Gy.
        return 1.0

    def _export_segment_mask(self, segmentation_node, segment_id: str, reference_volume_node):
        """Export one segment to reference volume geometry and return a boolean mask array."""
        if slicer.mrmlScene is None or segmentation_node is None or reference_volume_node is None:
            return None

        try:
            fn = getattr(slicer.util, "arrayFromSegmentBinaryLabelmap", None)
            if callable(fn):
                arr = fn(segmentation_node, segment_id, reference_volume_node)
                if arr is None:
                    return None
                return (np.asarray(arr) > 0)
        except Exception:
            pass

        labelmap = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", f"tmp_dvh_{uuid4().hex[:6]}"
        )
        try:
            labelmap.SetHideFromEditors(1)
            labelmap.SetSelectable(0)
            labelmap.SetSaveWithScene(0)
        except Exception:
            pass

        try:
            seg_logic = slicer.modules.segmentations.logic()
            seg_ids = vtk.vtkStringArray()
            seg_ids.InsertNextValue(str(segment_id))
            seg_logic.ExportSegmentsToLabelmapNode(segmentation_node, seg_ids, labelmap, reference_volume_node)
            arr = slicer.util.arrayFromVolume(labelmap)
            return np.asarray(arr) > 0
        except Exception:
            return None
        finally:
            try:
                if labelmap is not None and labelmap.GetScene() == slicer.mrmlScene:
                    try:
                        dn = labelmap.GetDisplayNode() if hasattr(labelmap, "GetDisplayNode") else None
                        if dn is not None and dn.GetScene() == slicer.mrmlScene:
                            try:
                                slicer.mrmlScene.RemoveNode(dn)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    try:
                        sn = labelmap.GetStorageNode() if hasattr(labelmap, "GetStorageNode") else None
                        if sn is not None and sn.GetScene() == slicer.mrmlScene:
                            try:
                                slicer.mrmlScene.RemoveNode(sn)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    try:
                        slicer.mrmlScene.RemoveNode(labelmap)
                    except Exception:
                        pass
            except Exception:
                pass

    def _compute_cumulative_dvh(self, dose_vals: np.ndarray, bins_edges_gy: np.ndarray, voxel_volume_cc: float):
        if dose_vals is None or bins_edges_gy is None:
            return None, None
        try:
            vals = np.asarray(dose_vals, dtype=np.float32)
            vals = vals[np.isfinite(vals)]
        except Exception:
            return None, None
        if vals.size == 0:
            return None, None

        hist, _ = np.histogram(vals, bins=bins_edges_gy)
        cum = np.cumsum(hist[::-1])[::-1].astype(np.float32)
        if cum.size == 0 or cum[0] <= 0:
            return None, None
        vcc = np.asarray(cum, dtype=np.float32) * float(voxel_volume_cc)
        vpct = np.asarray(100.0 * cum / float(cum[0]), dtype=np.float32)
        return vpct, vcc

    def _on_compute_dvh(self) -> None:
        if self._active_job is not None:
            try:
                QMessageBox.information(self, "Busy", "A DVH computation is already running.")
            except Exception:
                pass
            return
        if slicer.mrmlScene is None:
            QMessageBox.warning(self, "No Scene", "MRML scene is not available.")
            return

        dose_a = self._selected_dose_node_a()
        dose_b = self._selected_dose_node_b()
        seg_node = self.seg_selector.currentNode() if hasattr(self, "seg_selector") else None
        if dose_a is None or seg_node is None:
            QMessageBox.warning(self, "Missing Inputs", "Select a dose A and a segmentation.")
            return

        seg_ids = self._selected_segment_ids()
        if not seg_ids:
            QMessageBox.warning(self, "Missing Inputs", "Select at least one segment.")
            return

        # Resolve uncertainty volumes (only if enabled & available in same folder).
        unc_a = self._find_uncertainty_in_same_folder(dose_a) if bool(self.cb_unc_a.isChecked()) else None
        unc_b = self._find_uncertainty_in_same_folder(dose_b) if (dose_b is not None and bool(self.cb_unc_b.isChecked())) else None

        out_base = (self.output_name_edit.text or "")
        try:
            out_base = out_base() if callable(out_base) else out_base
        except Exception:
            out_base = ""
        out_base = str(out_base).strip() or f"dvh_{uuid4().hex[:6]}"

        # Build job.
        self._active_job = {
            "out_base": out_base,
            "seg_node": seg_node,
            "seg_ids": list(seg_ids),
            "dose_nodes": [dose_a] + ([dose_b] if dose_b is not None else []),
            "unc_nodes": {"A": unc_a, "B": unc_b},
            "curves": [],
            "temp_nodes": [],
            "stage": "start",
            "seg_color_by_id": {},
        }


        self._last_out_base = out_base

        self._last_png_path = None
        # show_png button removed

        self._set_ui_busy(True)
        self._set_status("Preparing…")
        self._set_progress(0, visible=True)

        def _cleanup_temp_nodes():
            j = self._active_job
            if j is None:
                return
            for tn in j.get("temp_nodes", []):
                try:
                    if tn is None or slicer.mrmlScene is None:
                        continue
                    try:
                        in_scene = tn.GetScene() == slicer.mrmlScene
                    except Exception:
                        in_scene = False
                    if not in_scene:
                        continue
                    # Remove display node(s) first to avoid VTK pipeline warnings.
                    try:
                        dn = tn.GetDisplayNode() if hasattr(tn, "GetDisplayNode") else None
                        if dn is not None and dn.GetScene() == slicer.mrmlScene:
                            try:
                                slicer.mrmlScene.RemoveNode(dn)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # Remove storage node if present.
                    try:
                        sn = tn.GetStorageNode() if hasattr(tn, "GetStorageNode") else None
                        if sn is not None and sn.GetScene() == slicer.mrmlScene:
                            try:
                                slicer.mrmlScene.RemoveNode(sn)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # Finally remove the temp node itself.
                    try:
                        slicer.mrmlScene.RemoveNode(tn)
                    except Exception:
                        pass
                except Exception:
                    pass

        def _finish(ok: bool, msg: str = ""):
            _cleanup_temp_nodes()
            self._active_job = None
            self._set_ui_busy(False)
            self._set_progress(0, visible=False)
            if msg:
                try:
                    QMessageBox.information(self, "DVH", str(msg)) if ok else QMessageBox.warning(self, "DVH", str(msg))
                except Exception:
                    pass

        def _fail(msg: str):
            _finish(False, msg)

        # Read dose arrays once (fast path) and pick automatic bins based on max dose.
        j = self._active_job
        dose_arrays = []
        dose_labels = []
        try:
            dose_arrays.append(np.asarray(slicer.util.arrayFromVolume(dose_a), dtype=np.float32))
            dose_labels.append("A")
        except Exception as exc:
            _fail(f"Failed to read dose A array: {exc}")
            return
        if dose_b is not None:
            try:
                dose_arrays.append(np.asarray(slicer.util.arrayFromVolume(dose_b), dtype=np.float32))
                dose_labels.append("B")
            except Exception:
                dose_b = None

        max_dose_raw = 0.0
        for arr in dose_arrays:
            try:
                m = float(np.nanmax(arr))
                if np.isfinite(m):
                    max_dose_raw = max(max_dose_raw, m)
            except Exception:
                pass
        if not np.isfinite(max_dose_raw) or max_dose_raw <= 0.0:
            _fail("Max dose is 0 or invalid.")
            return

        def _dose_grid_scaling_from_node(n) -> float:
            if n is None or not hasattr(n, "GetAttribute"):
                return 1.0
            # Common attribute keys seen in Slicer for DICOM-loaded RTDOSE.
            keys = (
                "DoseGridScaling",
                "DICOM.DoseGridScaling",
                "DICOM.RTDOSE.DoseGridScaling",
                "RTDOSE.DoseGridScaling",
                "DicomRtDoseGridScaling",
            )
            for k in keys:
                try:
                    v = n.GetAttribute(k)
                except Exception:
                    v = None
                if not v:
                    continue
                try:
                    f = float(v)
                    if np.isfinite(f) and f > 0:
                        return float(f)
                except Exception:
                    continue
            return 1.0

        # Determine a per-dose scaling to Gy.
        # Prefer DICOM RTDOSE DoseGridScaling when present; otherwise fallback to heuristic.
        j["_dose_scale_by_label"] = {}
        max_dose_gy = 0.0
        for label, node, arr in zip(dose_labels, [dose_a] + ([dose_b] if dose_b is not None else []), dose_arrays):
            s = 1.0
            try:
                s = float(_dose_grid_scaling_from_node(node))
            except Exception:
                s = 1.0

            # Heuristic fallback if scaling attribute is absent/unity.
            if abs(float(s) - 1.0) < 1e-12:
                try:
                    m = float(np.nanmax(arr))
                except Exception:
                    m = 0.0
                hs = 1.0
                try:
                    for _ in range(12):
                        if not np.isfinite(m):
                            break
                        if (float(m) * float(hs)) > 200.0:
                            hs *= 0.1
                            continue
                        break
                except Exception:
                    hs = 1.0
                s = float(hs)

            j["_dose_scale_by_label"][label] = float(s)
            try:
                m = float(np.nanmax(arr))
                if np.isfinite(m):
                    max_dose_gy = max(max_dose_gy, float(m) * float(s))
            except Exception:
                pass

        # Keep a representative scale for legend (dose A)
        j["_dose_scale"] = float(j["_dose_scale_by_label"].get("A", 1.0))

        max_dose = float(max_dose_gy)

        bw = self._auto_bin_width_gy(max_dose)
        edges = np.arange(0.0, float(max_dose) + float(bw), float(bw), dtype=np.float32)
        if edges.size < 2:
            _fail("Failed to build DVH bins.")
            return
        centers = np.asarray(edges[:-1], dtype=np.float32)
        j["_bins_edges"] = edges
        j["_bins_centers"] = centers

        # Resolve per-dose uncertainty arrays (if requested).
        j["_dose_arr_by_label"] = {}
        j["_unc_arr_by_label"] = {}
        for label, node, darr in zip(dose_labels, [dose_a] + ([dose_b] if dose_b is not None else []), dose_arrays):
            j["_dose_arr_by_label"][label] = darr
            unc_node = unc_a if label == "A" else unc_b
            if unc_node is not None:
                uarr = None
                try:
                    uarr = np.asarray(slicer.util.arrayFromVolume(unc_node), dtype=np.float32)
                except Exception:
                    uarr = None
                j["_unc_arr_by_label"][label] = uarr
            else:
                j["_unc_arr_by_label"][label] = None

        # Pre-compute voxel volume in cc per dose label (use each dose spacing).
        j["_vox_cc_by_label"] = {}
        for label, node in zip(dose_labels, [dose_a] + ([dose_b] if dose_b is not None else [])):
            try:
                sx, sy, sz = (float(v) for v in node.GetSpacing())
                j["_vox_cc_by_label"][label] = (sx * sy * sz) * 1e-3
            except Exception:
                j["_vox_cc_by_label"][label] = 1.0

        # Async loop over (dose label, segment).
        j["_dose_labels"] = list(dose_labels)
        j["_dose_idx"] = 0
        j["_seg_idx"] = 0

        self._set_status("Computing DVH…")
        self._set_progress(5, visible=True)

        def _tick():
            j2 = self._active_job
            if j2 is None:
                return

            dlabels = j2.get("_dose_labels", [])
            if not dlabels:
                _fail("No dose selected.")
                return
            di = int(j2.get("_dose_idx", 0))
            si = int(j2.get("_seg_idx", 0))
            seg_ids2 = j2.get("seg_ids", [])
            total = max(1, len(dlabels) * len(seg_ids2))

            if di >= len(dlabels):
                _build_outputs()
                return
            if si >= len(seg_ids2):
                j2["_dose_idx"] = di + 1
                j2["_seg_idx"] = 0
                try:
                    QTimer.singleShot(0, _tick)
                except Exception:
                    _tick()
                return

            label = str(dlabels[di])
            seg_id = str(seg_ids2[si])
            j2["_seg_idx"] = si + 1

            dose_node = dose_a if label == "A" else dose_b
            dose_arr = j2.get("_dose_arr_by_label", {}).get(label, None)
            if dose_node is None or dose_arr is None:
                try:
                    QTimer.singleShot(0, _tick)
                except Exception:
                    _tick()
                return

            # Segment name + color
            seg_name = seg_id
            seg_color = None
            try:
                seg = j2["seg_node"].GetSegmentation()
                seg_obj = seg.GetSegment(seg_id) if seg is not None else None
                seg_name = seg_obj.GetName() if seg_obj is not None else seg_id
                try:
                    if seg_obj is not None and hasattr(seg_obj, "GetColor"):
                        c = seg_obj.GetColor()
                        seg_color = (float(c[0]), float(c[1]), float(c[2]))
                except Exception:
                    seg_color = None
            except Exception:
                seg_name = seg_id

            if seg_color is not None:
                try:
                    j2.get("seg_color_by_id", {})[seg_id] = seg_color
                except Exception:
                    pass

            mask = self._export_segment_mask(j2["seg_node"], seg_id, dose_node)
            if mask is None:
                pass
            else:
                try:
                    if int(np.count_nonzero(mask)) > 0:
                        bins_edges = j2.get("_bins_edges")
                        vox_cc = float(j2.get("_vox_cc_by_label", {}).get(label, 1.0))
                        scale = float(j2.get("_dose_scale_by_label", {}).get(label, j2.get("_dose_scale", 1.0)))
                        vals = dose_arr[mask] * scale
                        vpct, vcc = self._compute_cumulative_dvh(vals, bins_edges, vox_cc)
                        dose_role = "Ref" if label == "A" else "Estimated"
                        if vpct is not None:
                            j2["curves"].append(
                                {
                                    "label": f"{dose_role} | {seg_name}",
                                    "dose_label": label,
                                    "seg_id": seg_id,
                                    "seg_name": seg_name,
                                    "kind": "mean",
                                    "color": seg_color,
                                    "vpct": vpct,
                                    "vcc": vcc,
                                }
                            )

                        # Optional uncertainty traces as DVH(dose±3σ)
                        uarr = j2.get("_unc_arr_by_label", {}).get(label, None)
                        if uarr is not None:
                            try:
                                uvals = 3.0 * uarr[mask] * scale
                                plus = vals + uvals
                                minus = np.maximum(vals - uvals, 0.0)
                                vpct_p, vcc_p = self._compute_cumulative_dvh(plus, bins_edges, vox_cc)
                                vpct_m, vcc_m = self._compute_cumulative_dvh(minus, bins_edges, vox_cc)
                                if vpct_p is not None:
                                    j2["curves"].append(
                                        {
                                            "label": f"{dose_role} | {seg_name} | max (+3σ)",
                                            "dose_label": label,
                                            "seg_id": seg_id,
                                            "seg_name": seg_name,
                                            "kind": "+3sigma",
                                            "color": seg_color,
                                            "vpct": vpct_p,
                                            "vcc": vcc_p,
                                        }
                                    )
                                if vpct_m is not None:
                                    j2["curves"].append(
                                        {
                                            "label": f"{dose_role} | {seg_name} | min (-3σ)",
                                            "dose_label": label,
                                            "seg_id": seg_id,
                                            "seg_name": seg_name,
                                            "kind": "-3sigma",
                                            "color": seg_color,
                                            "vpct": vpct_m,
                                            "vcc": vcc_m,
                                        }
                                    )
                            except Exception:
                                pass
                except Exception:
                    pass

            done = min(total, (di * len(seg_ids2)) + (si + 1))
            try:
                p = 5 + int(80 * float(done) / float(total))
            except Exception:
                p = 50
            self._set_progress(min(90, p), visible=True)
            self._set_status(f"Computing DVH… ({done}/{total})")

            try:
                QTimer.singleShot(0, _tick)
            except Exception:
                _tick()

        def _create_or_get_table_node(name: str):
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

        def _build_outputs():
            j2 = self._active_job
            if j2 is None:
                return
            self._set_status("Building table/plot…")
            self._set_progress(92, visible=True)

            # Legend is handled in the matplotlib PNG and UI legend; nothing to do here.

            table_name = str(j2.get("out_base"))
            table_node = _create_or_get_table_node(table_name)
            if table_node is None:
                _fail("Could not create output TableNode.")
                return

            table = table_node.GetTable()
            table.Initialize()

            centers2 = np.asarray(j2.get("_bins_centers"), dtype=np.float32)
            nrows = int(centers2.size)

            # X axis (dose in Gy)
            xcol = vtk.vtkFloatArray()
            xcol.SetName("dose_gy")
            xcol.SetNumberOfTuples(nrows)
            table.AddColumn(xcol)

            # Note: vtkTable does not automatically resize columns added after SetNumberOfRows.
            # Ensure every added column has nrows tuples to avoid VTK warnings.
            table.SetNumberOfRows(nrows)
            # Fill X column directly (more robust than vtkTable.SetValue for some builds)
            for r in range(nrows):
                try:
                    xcol.SetValue(r, float(centers2[r]))
                except Exception:
                    pass
            # Ensure column 0 name is set at the vtkTable level
            try:
                if hasattr(table, "SetColumnName"):
                    table.SetColumnName(0, "dose_gy")
            except Exception:
                pass
            try:
                c0 = table.GetColumn(0)
                if c0 is not None and hasattr(c0, "SetName"):
                    c0.SetName("dose_gy")
            except Exception:
                pass

            # Add one pair of columns per curve (V% and Vcc)
            curves = list(j2.get("curves", []))
            col_specs = []

            def _safe_col_base(s: str) -> str:
                try:
                    txt = str(s)
                except Exception:
                    txt = ""
                # Avoid non-ascii characters that can break MRML column lookups in some Slicer builds.
                txt = txt.replace("σ", "sigma").replace("±", "+/-")
                # Keep it readable but conservative.
                try:
                    import re

                    txt = re.sub(r"[^0-9A-Za-z _\-|:+().]", "_", txt)
                except Exception:
                    pass
                return txt

            for c in curves:
                label = str(c.get("label", "curve"))
                col_base = _safe_col_base(label)
                col_vpct = f"{col_base} | V%"
                col_vcc = f"{col_base} | Vcc"
                vp = vtk.vtkFloatArray(); vp.SetName(col_vpct); vp.SetNumberOfTuples(nrows)
                vc = vtk.vtkFloatArray(); vc.SetName(col_vcc); vc.SetNumberOfTuples(nrows)
                table.AddColumn(vp)
                table.AddColumn(vc)
                col_specs.append((col_vpct, col_vcc, c))

            # Fill curve columns
            for ci, (col_vpct, col_vcc, c) in enumerate(col_specs):
                vpct = np.asarray(c.get("vpct"), dtype=np.float32)
                vcc = np.asarray(c.get("vcc"), dtype=np.float32)
                if vpct.size != nrows:
                    continue
                for r in range(nrows):
                    try:
                        table.SetValue(r, 1 + (2 * ci), float(vpct[r]))
                        table.SetValue(r, 1 + (2 * ci) + 1, float(vcc[r]))
                    except Exception:
                        pass

            def _render_matplotlib_png():
                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    try:
                        matplotlib.set_loglevel("warning")
                    except Exception:
                        pass
                    import matplotlib.pyplot as plt
                    import tempfile
                except Exception:
                    return None

                xs = np.asarray(centers2, dtype=np.float32)
                if xs.size == 0:
                    return None

                try:
                    plt.style.use("seaborn-v0_8-colorblind")
                except Exception:
                    pass

                fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=110)
                ax.set_facecolor("#f6f7fb")
                ax.set_xlabel("Dose (Gy)")
                ax.set_ylabel("Volume (%)")
                ax.grid(True, alpha=0.25, color="#cfd3dc", linewidth=0.8)

                grouped = {}
                for c in curves:
                    key = (c.get("dose_label", ""), c.get("seg_id", ""))
                    try:
                        grouped.setdefault(key, {})[str(c.get("kind", "mean"))] = c
                    except Exception:
                        pass

                structure_handles = []
                structure_labels = []
                added_struct_labels = set()
                unc_patch_handle = None

                for (_dl, _sid), ck in grouped.items():
                    dose_label = _dl or ""
                    mean_c = ck.get("mean")
                    if mean_c is None:
                        continue
                    y_mean = np.asarray(mean_c.get("vpct"), dtype=np.float32)
                    if y_mean.size != xs.size:
                        continue
                    rgb = mean_c.get("color", None) or (0.0, 0.0, 0.0)
                    try:
                        struct_label = str(mean_c.get("seg_name", "") or "")
                    except Exception:
                        struct_label = ""

                    # Fill between -3σ et +3σ (gris translucide)
                    upper_c = ck.get("+3sigma")
                    lower_c = ck.get("-3sigma")
                    if upper_c is not None and lower_c is not None:
                        y_up = np.asarray(upper_c.get("vpct"), dtype=np.float32)
                        y_lo = np.asarray(lower_c.get("vpct"), dtype=np.float32)
                        if y_up.size == xs.size and y_lo.size == xs.size:
                            try:
                                ax.fill_between(xs, y_lo, y_up, color=rgb, alpha=0.20, linewidth=0)
                                # keep fill for CI; add a single legend entry later
                                if unc_patch_handle is None:
                                    from matplotlib.patches import Patch

                                    unc_patch_handle = Patch(facecolor="#9ca3af", edgecolor="#4b5563", alpha=0.35, label="Uncertainty (±3σ)")
                            except Exception:
                                pass
                    try:
                        line_style = "--" if dose_label == "B" else "-"
                        ax.plot(xs, y_mean, color=rgb, lw=1.8, alpha=1.0, linestyle=line_style, label=struct_label)
                        try:
                            line_handle = ax.lines[-1]
                            if struct_label and struct_label not in added_struct_labels:
                                added_struct_labels.add(struct_label)
                                structure_handles.append(line_handle)
                                structure_labels.append(struct_label)
                        except Exception:
                            pass
                    except Exception:
                        pass

                try:
                    legend_handles = []
                    legend_labels = []

                    # Structures: one entry per structure (color-coded)
                    legend_handles.extend(structure_handles)
                    legend_labels.extend(structure_labels)

                    # Dose roles (global)
                    from matplotlib.lines import Line2D

                    legend_handles.append(Line2D([], [], color="#111827", lw=1.8, linestyle="-", label="Dose ref"))
                    legend_labels.append("Dose ref")
                    legend_handles.append(Line2D([], [], color="#111827", lw=1.8, linestyle="--", label="Dose estimated"))
                    legend_labels.append("Dose estimated")

                    if unc_patch_handle is not None:
                        legend_handles.append(unc_patch_handle)
                        legend_labels.append("Uncertainty (±3σ)")

                    ax.legend(legend_handles, legend_labels, loc="best", fontsize=8, frameon=True, fancybox=True)
                except Exception:
                    pass
                try:
                    fig.tight_layout()
                except Exception:
                    pass

                buf = io.BytesIO()
                try:
                    fig.savefig(buf, format="png", dpi=110)
                except Exception:
                    return None
                try:
                    out_path = os.path.join(tempfile.gettempdir(), f"{table_name}.png")
                    with open(out_path, "wb") as f:
                        f.write(buf.getvalue())
                    return out_path
                except Exception:
                    return None
                finally:
                    try:
                        plt.close(fig)
                    except Exception:
                        pass

            table_node.Modified()
            try:
                table.Modified()
            except Exception:
                pass

            try:
                png_path = _render_matplotlib_png()
                if png_path:
                    j2["png_path"] = png_path
                    self._last_png_path = png_path
                    # show_png_btn removed
            except Exception:
                pass

            x_col_name = "dose_gy"

            # Plot (MRML only)
            try:
                if self.plot_widget is not None:
                    self.plot_widget.setVisible(True)
            except Exception:
                pass

            # Plot
            chart_node = None
            try:
                chart_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode", f"{table_name}_chart")
            except Exception:
                chart_node = None

            if chart_node is not None:
                # Hide internal legend (we use custom legend below the plot)
                try:
                    self._set_chart_legend_visible(chart_node, False)
                except Exception:
                    pass

                try:
                    chart_node.RemoveAllPlotSeriesNodeIDs()
                except Exception:
                    pass

                # --- Uncertainty visualization: semi-transparent lines (no halo)
                series_index = 0
                created_series_nodes = []

                def _add_line_series(y_col: str, rgb, name_suffix: str, width: int, dashed: bool, opacity=None, label=None) -> None:
                    nonlocal series_index
                    sname = f"{table_name}_{series_index:02d}_{name_suffix}"
                    series_index += 1
                    try:
                        s = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", sname)
                    except Exception:
                        s = None
                    if s is None:
                        return
                    try:
                        s.SetAndObserveTableNodeID(table_node.GetID())
                        if hasattr(s, "SetXColumnName"):
                            try:
                                s.SetXColumnName(x_col_name)
                            except Exception:
                                pass
                        if hasattr(s, "SetYColumnName"):
                            try:
                                s.SetYColumnName(y_col)
                            except Exception:
                                pass
                        s.SetPlotType(s.PlotTypeLine)
                        s.SetMarkerStyle(s.MarkerStyleNone)
                    except Exception:
                        pass
                    try:
                        if dashed:
                            s.SetLineStyle(s.LineStyleDash)
                        else:
                            s.SetLineStyle(s.LineStyleSolid)
                    except Exception:
                        pass
                    try:
                        if hasattr(s, "SetLineWidth"):
                            s.SetLineWidth(int(width))
                    except Exception:
                        pass
                    try:
                        if rgb is not None and hasattr(s, "SetColor"):
                            s.SetColor(float(rgb[0]), float(rgb[1]), float(rgb[2]))
                    except Exception:
                        pass
                    if opacity is not None:
                        try:
                            self._apply_series_opacity(s, float(opacity))
                        except Exception:
                            pass
                    # Set readable series name/title for legend
                    # series naming omitted: legend handled in _update_legend
                    try:
                        s.Modified()
                    except Exception:
                        pass
                    try:
                        chart_node.AddAndObservePlotSeriesNodeID(s.GetID())
                    except Exception:
                        pass
                    try:
                        created_series_nodes.append(s)
                    except Exception:
                        pass

                # Add uncertainty lines first (semi-transparent) then crisp mean lines on top
                for _ci, (col_vpct, _col_vcc, c) in enumerate(col_specs):
                    kind = str(c.get("kind", ""))
                    dose_label = str(c.get("dose_label", ""))
                    rgb = c.get("color", None) or (0.0, 0.0, 0.0)
                    dashed = (dose_label == "B")
                    try:
                        struct_label = str(c.get("seg_name", "") or "")
                    except Exception:
                        struct_label = ""
                    if "sigma" in kind:
                        _add_line_series(col_vpct, rgb, "unc", width=2, dashed=False, opacity=0.35, label=struct_label)
                    else:
                        _add_line_series(col_vpct, rgb, "mean", width=2, dashed=dashed, opacity=None, label=struct_label)


                chart_node.SetTitle(table_name)
                chart_node.SetXAxisTitle("Dose (Gy)")
                chart_node.SetYAxisTitle("Volume (%)")

                # Best-effort: force X axis range from our bin centers
                try:
                    x_max = float(centers2[-1]) if int(nrows) > 0 else None
                except Exception:
                    x_max = None
                if x_max is not None and x_max > 0:
                    for m in ("SetXAxisRange", "SetXRange"):
                        try:
                            if hasattr(chart_node, m):
                                getattr(chart_node, m)(0.0, float(x_max))
                                break
                        except Exception:
                            pass

                self._set_plot_chart_on_widget(chart_node)

                # Build custom legend items from computed curves (unique structures)
                try:
                    structure_items = []
                    seen = set()
                    for c in list(curves):
                        try:
                            name = str(c.get("seg_name", "") or "")
                        except Exception:
                            name = ""
                        try:
                            color = c.get("color", None) or (0.0, 0.0, 0.0)
                        except Exception:
                            color = (0.0, 0.0, 0.0)
                        if name and name not in seen:
                            seen.add(name)
                            structure_items.append({"name": name, "color": color})
                    try:
                        self._update_legend(structure_items=structure_items)
                    except Exception:
                        pass
                except Exception:
                    pass

                # Move created nodes into a SubjectHierarchy folder to reduce node clutter.
                try:
                    shNode = self._get_sh_node()
                    if shNode is not None:
                        # Parent folder: try to use the same folder as Dose A when available.
                        parent_item = 0
                        try:
                            dose_nodes = j2.get("dose_nodes", []) or []
                            dose_a = dose_nodes[0] if len(dose_nodes) > 0 else None
                        except Exception:
                            dose_a = None
                        try:
                            if dose_a is not None:
                                dose_item = int(shNode.GetItemByDataNode(dose_a) or 0)
                                if dose_item:
                                    parent_item = int(shNode.GetItemParent(dose_item) or 0)
                        except Exception:
                            parent_item = 0
                        if not parent_item:
                            try:
                                parent_item = int(shNode.GetSceneItemID() or 0)
                            except Exception:
                                parent_item = 0

                        # Create/locate DVH root folder, then a subfolder for this run.
                        def _find_child_folder(parent, name):
                            try:
                                ids = vtk.vtkIdList()
                                shNode.GetItemChildren(parent, ids)
                                for ii in range(ids.GetNumberOfIds()):
                                    child = ids.GetId(ii)
                                    try:
                                        if shNode.GetItemName(child) == name:
                                            return int(child)
                                    except Exception:
                                        continue
                            except Exception:
                                pass
                            return 0

                        dvh_root = _find_child_folder(parent_item, "DVH")
                        if not dvh_root:
                            try:
                                dvh_root = int(shNode.CreateFolderItem(parent_item, "DVH") or 0)
                            except Exception:
                                dvh_root = 0
                        run_folder_name = str(table_name)
                        run_folder = _find_child_folder(dvh_root, run_folder_name) if dvh_root else 0
                        if dvh_root and not run_folder:
                            try:
                                run_folder = int(shNode.CreateFolderItem(dvh_root, run_folder_name) or 0)
                            except Exception:
                                run_folder = 0

                        def _reparent_node(node, parent_folder_item):
                            if node is None or not parent_folder_item:
                                return
                            try:
                                item = int(shNode.GetItemByDataNode(node) or 0)
                            except Exception:
                                item = 0
                            if not item:
                                return
                            try:
                                shNode.SetItemParent(item, int(parent_folder_item))
                            except Exception:
                                pass

                        target_folder = run_folder or dvh_root
                        _reparent_node(table_node, target_folder)
                        _reparent_node(chart_node, target_folder)
                        for s in created_series_nodes or []:
                            _reparent_node(s, target_folder)

                        png_node = None
                        try:
                            png_path = j2.get("png_path", None)
                        except Exception:
                            png_path = None
                        if png_path:
                            png_node = self._load_png_node(png_path, f"{table_name}_png")
                            _reparent_node(png_node, target_folder)
                except Exception:
                    pass

            self._set_progress(100, visible=False)
            self._set_status("Done.")
            _finish(True, f"DVH written to table: {table_name}")

        try:
            QTimer.singleShot(0, _tick)
        except Exception:
            _tick()
