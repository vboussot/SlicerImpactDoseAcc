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
    QProgressBar,
    QTimer,
)

class DVHWidget(QWidget):
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
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()

        inputs_group = QGroupBox("1. Inputs")
        inputs_layout = QVBoxLayout()

        header = QHBoxLayout()
        header.addWidget(QLabel("Select inputs for DVH:"))
        header.addStretch()
        refresh_btn = QPushButton("⟳")
        refresh_btn.setMaximumWidth(40)
        refresh_btn.setToolTip("Refresh lists")
        refresh_btn.clicked.connect(self._refresh_dose_lists)
        header.addWidget(refresh_btn)
        inputs_layout.addLayout(header)
   

        dose_row_a = QHBoxLayout()
        dose_row_a.addWidget(QLabel("Dose ref:"))
        self.dose_combo_a = QComboBox()
        dose_row_a.addWidget(self.dose_combo_a, 1)
        self.cb_unc_a = QCheckBox("Show uncertainty (±3σ)")
        self.cb_unc_a.setChecked(False)
        self.cb_unc_a.setEnabled(False)
        dose_row_a.addWidget(self.cb_unc_a)
        inputs_layout.addLayout(dose_row_a)

        dose_row_b = QHBoxLayout()
        dose_row_b.addWidget(QLabel("Dose estimated (optional):"))
        self.dose_combo_b = QComboBox()
        dose_row_b.addWidget(self.dose_combo_b, 1)
        self.cb_unc_b = QCheckBox("Show uncertainty (±3σ)")
        self.cb_unc_b.setChecked(False)
        self.cb_unc_b.setEnabled(False)
        dose_row_b.addWidget(self.cb_unc_b)
        inputs_layout.addLayout(dose_row_b)

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
        out_row.addWidget(QLabel("Output base name:"))
        self.output_name_edit = QLineEdit()
        self.output_name_edit.setText(f"dvh_{uuid4().hex[:6]}")
        out_row.addWidget(self.output_name_edit, 1)
        inputs_layout.addLayout(out_row)

        inputs_group.setLayout(inputs_layout)
        layout.addWidget(inputs_group)

        self.run_btn = QPushButton("Compute DVH")
        self.run_btn.clicked.connect(self._on_compute_dvh)
        layout.addWidget(self.run_btn)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Embedded plot widget (shows a PlotChartNode).
        self.plot_widget = None
        try:
            self.plot_widget = slicer.qMRMLPlotWidget()
            self.plot_widget.setMRMLScene(slicer.mrmlScene)
            try:
                self.plot_widget.setMinimumHeight(260)
            except Exception:
                pass
            layout.addWidget(self.plot_widget)
        except Exception:
            self.plot_widget = None

        # Ensure embedded plot has its own PlotViewNode (required for display in some Slicer versions)
        try:
            self._ensure_embedded_plot_view()
        except Exception:
            pass

        # Custom legend (more controllable than PlotChart legend)
        self.legend_group = QGroupBox("Legend")
        legend_layout = QVBoxLayout()
        self.legend_label = QLabel("")
        self.legend_label.setWordWrap(True)
        legend_layout.addWidget(self.legend_label)
        self.legend_group.setLayout(legend_layout)
        layout.addWidget(self.legend_group)

        layout.addStretch()
        self.setLayout(layout)

        try:
            self.seg_selector.currentNodeChanged.connect(self._on_segmentation_changed)
        except Exception:
            pass
        try:
            self.dose_combo_a.currentIndexChanged.connect(self._on_dose_selection_changed)
            self.dose_combo_b.currentIndexChanged.connect(self._on_dose_selection_changed)
        except Exception:
            pass

        self._refresh_dose_lists()
        self._on_segmentation_changed(self.seg_selector.currentNode())
        self._on_dose_selection_changed()

        self._update_legend(structure_items=[], dose_scale=1.0)

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

    def _update_legend(self, structure_items, dose_scale: float = 1.0):
        """Update custom legend.

        structure_items: list of dicts with keys: name, color(rgb01)
        """
        try:
            try:
                s = float(dose_scale)
            except Exception:
                s = 1.0
            scale_txt = "×1"
            try:
                if s > 0 and abs(s - 1.0) > 1e-12:
                    inv = 1.0 / float(s)
                    # Prefer ÷10^n when close to a power of 10.
                    n = int(round(np.log10(inv))) if inv > 0 else 0
                    if n > 0 and abs(inv - (10.0 ** n)) < 1e-6:
                        scale_txt = f"÷1e{n}"
                    else:
                        scale_txt = f"×{s:g}"
            except Exception:
                scale_txt = "×1"

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
                "<div><span style='color:#000; font-weight:600;'>━━━━</span>&nbsp;Incertitude (±3σ)</div>"
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

    def _safe_node_name(self, node) -> str:
        if node is None or not hasattr(node, "GetName"):
            return ""
        try:
            return node.GetName() or ""
        except Exception:
            return ""

    def _combo_current_index(self, combo) -> int:
        if combo is None:
            return 0
        idx_attr = getattr(combo, "currentIndex", 0)
        try:
            return int(idx_attr() if callable(idx_attr) else idx_attr)
        except Exception:
            return 0

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

    def _get_sh_node(self):
        if slicer.mrmlScene is None:
            return None
        try:
            return slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
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
                    slicer.mrmlScene.RemoveNode(labelmap)
            except Exception:
                pass

    def _compute_cumulative_dvh(self, dose_vals: np.ndarray, bins_edges_gy: np.ndarray, voxel_volume_cc: float):
        if dose_vals is None or bins_edges_gy is None:
            return None, None
        try:
            vals = np.asarray(dose_vals, dtype=np.float64)
            vals = vals[np.isfinite(vals)]
        except Exception:
            return None, None
        if vals.size == 0:
            return None, None

        hist, _ = np.histogram(vals, bins=bins_edges_gy)
        cum = np.cumsum(hist[::-1])[::-1].astype(np.float64)
        if cum.size == 0 or cum[0] <= 0:
            return None, None
        vcc = cum * float(voxel_volume_cc)
        vpct = 100.0 * cum / float(cum[0])
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

        self._set_ui_busy(True)
        self._set_status("Preparing…")
        self._set_progress(0, visible=True)

        def _cleanup_temp_nodes():
            j = self._active_job
            if j is None:
                return
            for tn in j.get("temp_nodes", []):
                try:
                    if tn is not None and slicer.mrmlScene is not None and tn.GetScene() == slicer.mrmlScene:
                        slicer.mrmlScene.RemoveNode(tn)
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
            dose_arrays.append(slicer.util.arrayFromVolume(dose_a).astype(np.float64, copy=False))
            dose_labels.append("A")
        except Exception as exc:
            _fail(f"Failed to read dose A array: {exc}")
            return
        if dose_b is not None:
            try:
                dose_arrays.append(slicer.util.arrayFromVolume(dose_b).astype(np.float64, copy=False))
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
        edges = np.arange(0.0, float(max_dose) + float(bw), float(bw), dtype=np.float64)
        if edges.size < 2:
            _fail("Failed to build DVH bins.")
            return
        centers = np.asarray(edges[:-1], dtype=np.float64)
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
                    uarr = slicer.util.arrayFromVolume(unc_node).astype(np.float64, copy=False)
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

            # Update legend (structures + styles + CI)
            try:
                scale_for_legend = float(j2.get("_dose_scale", 1.0))
                seg = j2["seg_node"].GetSegmentation() if j2.get("seg_node") is not None else None
                items = []
                for sid in j2.get("seg_ids", []):
                    sname = str(sid)
                    scol = j2.get("seg_color_by_id", {}).get(str(sid), None)
                    try:
                        if seg is not None:
                            sobj = seg.GetSegment(str(sid))
                            if sobj is not None:
                                sname = sobj.GetName() or sname
                                if scol is None and hasattr(sobj, "GetColor"):
                                    c = sobj.GetColor()
                                    scol = (float(c[0]), float(c[1]), float(c[2]))
                    except Exception:
                        pass
                    if scol is None:
                        scol = (0.0, 0.0, 0.0)
                    items.append({"name": sname, "color": scol})
                self._update_legend(items, dose_scale=scale_for_legend)
            except Exception:
                pass

            table_name = str(j2.get("out_base"))
            table_node = _create_or_get_table_node(table_name)
            if table_node is None:
                _fail("Could not create output TableNode.")
                return

            table = table_node.GetTable()
            table.Initialize()

            centers2 = np.asarray(j2.get("_bins_centers"), dtype=np.float64)
            nrows = int(centers2.size)

            # X axis (dose in Gy)
            xcol = vtk.vtkDoubleArray()
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
                vp = vtk.vtkDoubleArray(); vp.SetName(col_vpct); vp.SetNumberOfTuples(nrows)
                vc = vtk.vtkDoubleArray(); vc.SetName(col_vcc); vc.SetNumberOfTuples(nrows)
                table.AddColumn(vp)
                table.AddColumn(vc)
                col_specs.append((col_vpct, col_vcc, c))

            # Fill curve columns
            for ci, (col_vpct, col_vcc, c) in enumerate(col_specs):
                vpct = np.asarray(c.get("vpct"), dtype=np.float64)
                vcc = np.asarray(c.get("vcc"), dtype=np.float64)
                if vpct.size != nrows:
                    continue
                for r in range(nrows):
                    try:
                        table.SetValue(r, 1 + (2 * ci), float(vpct[r]))
                        table.SetValue(r, 1 + (2 * ci) + 1, float(vcc[r]))
                    except Exception:
                        pass

            # --- One-sided halo for uncertainty curves (±3σ)
            # We approximate clipping by drawing several curves shifted *inside* the interval:
            #   +3σ (upper boundary): shift downward (y - k*delta)
            #   -3σ (lower boundary): shift upward   (y + k*delta)
            # This ensures the halo does not visually extend outside the interval.
            halo_specs = []  # list of (y_column_name, rgb, opacity)
            HALO_LAYERS = 4
            HALO_DELTA_VPCT = 0.8
            for (col_vpct, _col_vcc, c) in col_specs:
                kind = str(c.get("kind", ""))
                if kind not in ("+3sigma", "-3sigma"):
                    continue
                base = np.asarray(c.get("vpct"), dtype=np.float64)
                if base.size != nrows:
                    continue
                rgb = c.get("color", None) or (0.0, 0.0, 0.0)
                sign = -1.0 if kind.startswith("+") else 1.0
                for k in range(1, int(HALO_LAYERS) + 1):
                    y2 = base + (sign * float(k) * float(HALO_DELTA_VPCT))
                    try:
                        y2 = np.clip(y2, 0.0, 100.0)
                    except Exception:
                        pass
                    col_halo = f"{col_vpct} | halo_{k:02d}"
                    arr = vtk.vtkDoubleArray()
                    arr.SetName(col_halo)
                    arr.SetNumberOfTuples(nrows)
                    for r in range(nrows):
                        try:
                            arr.SetValue(r, float(y2[r]))
                        except Exception:
                            arr.SetValue(r, 0.0)
                    table.AddColumn(arr)
                    # Slight opacity falloff with depth
                    opacity = max(0.04, 0.14 - 0.02 * float(k))
                    halo_specs.append((col_halo, rgb, opacity))

            table_node.Modified()
            try:
                table.Modified()
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

                # --- Uncertainty visualization: one-sided halo via shifted curves
                series_index = 0
                created_series_nodes = []

                def _add_line_series(y_col: str, rgb, name_suffix: str, width: int, dashed: bool, opacity=None) -> None:
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

                # Add uncertainty halos first (so mean lines remain crisp on top)
                try:
                    for (y_col, rgb, opacity) in halo_specs or []:
                        _add_line_series(y_col, rgb, "unc_halo", width=10, dashed=False, opacity=opacity)
                except Exception:
                    pass

                # Always add mean curves as crisp lines
                for _ci, (col_vpct, _col_vcc, c) in enumerate(col_specs):
                    kind = str(c.get("kind", ""))
                    if "sigma" in kind:
                        continue
                    dose_label = str(c.get("dose_label", ""))
                    rgb = c.get("color", None) or (0.0, 0.0, 0.0)
                    dashed = (dose_label == "B")
                    _add_line_series(col_vpct, rgb, "mean", width=2, dashed=dashed, opacity=None)


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
                except Exception:
                    pass

            self._set_progress(100, visible=False)
            self._set_status("Done.")
            _finish(True, f"DVH written to table: {table_name}")

        try:
            QTimer.singleShot(0, _tick)
        except Exception:
            _tick()
