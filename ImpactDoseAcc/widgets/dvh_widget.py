import io
import logging
import os
from typing import Any
from uuid import uuid4

import numpy as np
import slicer
import vtk
from qt import QCheckBox, QMessageBox, QTimer, QVBoxLayout
from widgets.base_widget import BaseImpactWidget

logger = logging.getLogger(__name__)


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

    def __init__(self) -> None:
        super().__init__()
        self._dose_node_by_index_a: dict[int, Any] = {0: None}
        self._dose_node_by_index_b: dict[int, Any] = {0: None}
        self._segment_checkbox_by_id: dict[str, QCheckBox] = {}
        self._active_job: dict[str, Any] | None = None
        self._plot_view_node = None
        self._setup_ui()

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
            self.seg_selector.nodeTypes = ["vtkMRMLSegmentationNode"]
            self.seg_selector.noneEnabled = True
            self.seg_selector.addEnabled = False
            self.seg_selector.removeEnabled = False
            self.seg_selector.setMRMLScene(slicer.mrmlScene)

        # Buttons & signals
        self._btn("refresh_btn", self._refresh_dose_lists)
        if self.run_btn is not None:
            self.run_btn.clicked.connect(self._on_compute_dvh)

        # show_png_btn removed
        if self.seg_selector is not None:
            self.seg_selector.currentNodeChanged.connect(self._on_segmentation_changed)

        if self.dose_combo_a is not None:
            self.dose_combo_a.currentIndexChanged.connect(self._on_dose_selection_changed)
        if self.dose_combo_b is not None:
            self.dose_combo_b.currentIndexChanged.connect(self._on_dose_selection_changed)

        if self.cb_unc_a is not None:
            self.cb_unc_a.setChecked(False)
            self.cb_unc_a.setEnabled(False)
        if self.cb_unc_b is not None:
            self.cb_unc_b.setChecked(False)
            self.cb_unc_b.setEnabled(False)

        if self.output_name_edit is not None:
            self.output_name_edit.setText(self._generate_default_output_name(prefix="dvh"))

        if self.progress_bar is not None:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(False)

        # Embedded plot widget configuration
        if self.plot_widget is not None:
            self.plot_widget.setMRMLScene(slicer.mrmlScene)
            self.plot_widget.setMinimumHeight(260)
            self._ensure_embedded_plot_view()

        layout = QVBoxLayout(self)
        layout.addWidget(ui_widget)
        self.setLayout(layout)

        self._refresh_dose_lists()
        self._on_segmentation_changed(self.seg_selector.currentNode() if self.seg_selector is not None else None)
        self._on_dose_selection_changed()
        self._update_legend(structure_items=[])

    def _set_plot_chart_on_widget(self, chart_node: Any) -> None:
        """Attach a vtkMRMLPlotChartNode to the embedded plot widget using standard APIs only.

        This does not force visibility or perform UI event loop tricks — it delegates to the
        widget / plot view APIs and returns silently on failure.
        """
        if self.plot_widget is None or chart_node is None:
            return

        self._ensure_embedded_plot_view()

        if (
            self._plot_view_node is not None
            and hasattr(self._plot_view_node, "SetPlotChartNodeID")
            and hasattr(chart_node, "GetID")
        ):
            self._plot_view_node.SetPlotChartNodeID(chart_node.GetID())

        if hasattr(self.plot_widget, "setMRMLPlotChartNode"):
            self.plot_widget.setMRMLPlotChartNode(chart_node)
        elif hasattr(self.plot_widget, "setMRMLPlotChartNodeID") and hasattr(chart_node, "GetID"):
            self.plot_widget.setMRMLPlotChartNodeID(chart_node.GetID())

    def _ensure_embedded_plot_view(self) -> None:
        if self.plot_widget is None or slicer.mrmlScene is None:
            return

        # Create (or reuse) a plot view node in the scene
        if self._plot_view_node is None:
            self._plot_view_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLPlotViewNode", f"DVHPlotView_{uuid4().hex[:6]}"
            )

        # Attach view node to widget
        if self._plot_view_node is not None:
            if hasattr(self.plot_widget, "setMRMLPlotViewNode"):
                self.plot_widget.setMRMLPlotViewNode(self._plot_view_node)
            elif hasattr(self.plot_widget, "setMRMLPlotViewNodeID") and hasattr(self._plot_view_node, "GetID"):
                self.plot_widget.setMRMLPlotViewNodeID(self._plot_view_node.GetID())

    def _set_chart_legend_visible(self, chart_node: Any, visible: bool) -> None:
        if chart_node is None:
            return
        v = bool(visible)
        for meth in (
            "SetLegendVisibility",
            "SetShowLegend",
            "SetLegendVisible",
            "SetShowLegendVisibility",
        ):
            if hasattr(chart_node, meth):
                getattr(chart_node, meth)(1 if v else 0)
                return

    def _apply_series_opacity(self, series: Any, alpha: float) -> None:
        if series is None:
            return
        a = float(alpha)
        a = max(0.0, min(1.0, a))
        for meth in ("SetOpacity", "SetLineOpacity", "SetFillOpacity", "SetBrushOpacity"):
            if hasattr(series, meth):
                getattr(series, meth)(a)
                return

    def _rgb01_to_hex(self, rgb: tuple[float, float, float]) -> str:
        r, g, b = rgb
        r = int(max(0, min(255, round(float(r) * 255.0))))
        g = int(max(0, min(255, round(float(g) * 255.0))))
        b = int(max(0, min(255, round(float(b) * 255.0))))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _update_legend(self, structure_items: list[dict[str, Any]] | None) -> None:
        """Update custom legend.

        structure_items: list of dicts with keys: name, color(rgb01)
        """
        structure_lines = []
        for it in structure_items or []:
            name = str(it.get("name", ""))
            hexcol = self._rgb01_to_hex(it.get("color", (0, 0, 0)))
            structure_lines.append(f"<div><span style='color:{hexcol}; font-weight:600;'>━━━━</span>&nbsp;{name}</div>")

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

    def _set_ui_busy(self, busy: bool) -> None:
        # Use base implementation for generic behavior, then apply widget-specific enables.
        super()._set_ui_busy(busy)
        self.run_btn.setEnabled(not bool(busy))
        self.dose_combo_a.setEnabled(not bool(busy))
        self.dose_combo_b.setEnabled(not bool(busy))
        self.seg_selector.setEnabled(not bool(busy))
        self.output_name_edit.setEnabled(not bool(busy))

        if bool(busy):
            self.cb_unc_a.setEnabled(False)
            self.cb_unc_b.setEnabled(False)
        else:
            self._on_dose_selection_changed()

    def _selected_dose_node_a(self) -> Any | None:
        return self._dose_node_by_index_a.get(self._combo_current_index(self.dose_combo_a), None)

    def _selected_dose_node_b(self) -> Any | None:
        return self._dose_node_by_index_b.get(self._combo_current_index(self.dose_combo_b), None)

    def _refresh_dose_lists(self) -> None:
        if slicer.mrmlScene is None:
            return

        # Keep previous selections by node ID.
        prev_a = self._selected_dose_node_a()
        prev_b = self._selected_dose_node_b()
        prev_a_id = prev_a.GetID() if prev_a is not None and hasattr(prev_a, "GetID") else None
        prev_b_id = prev_b.GetID() if prev_b is not None and hasattr(prev_b, "GetID") else None

        self.dose_combo_a.blockSignals(True)
        self.dose_combo_b.blockSignals(True)

        self.dose_combo_a.clear()
        self.dose_combo_b.clear()
        self._dose_node_by_index_a = {0: None}
        self._dose_node_by_index_b = {0: None}

        self.dose_combo_a.addItem("[Select dose A]")
        self.dose_combo_b.addItem("[None]")

        # Filter volumes by name containing 'dose'.
        volumes = list(slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"))
        dose_nodes = []
        for n in volumes:
            if n is None:
                continue
            name = self._safe_node_name(n)
            name_l = name.lower()
            if "dose" not in name_l:
                continue
            if "uncertainty" in name_l:
                continue
            dose_nodes.append((name_l, name, n))
        dose_nodes.sort(key=lambda t: t[0])

        match_a = 0
        match_b = 0
        idx = 1
        for _, name, n in dose_nodes:
            self.dose_combo_a.addItem(name)
            self._dose_node_by_index_a[idx] = n
            if prev_a_id and hasattr(n, "GetID") and n.GetID() == prev_a_id:
                match_a = idx

            self.dose_combo_b.addItem(name)
            self._dose_node_by_index_b[idx] = n
            if prev_b_id and hasattr(n, "GetID") and n.GetID() == prev_b_id:
                match_b = idx

            idx += 1

        if match_a:
            self.dose_combo_a.setCurrentIndex(match_a)

        if match_b:
            self.dose_combo_b.setCurrentIndex(match_b)

        self.dose_combo_a.blockSignals(False)
        self.dose_combo_b.blockSignals(False)

    def _clear_layout(self, layout) -> None:
        if layout is None:
            return

        while layout.count():
            item = layout.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.setParent(None)

    def _on_segmentation_changed(self, seg_node: Any) -> None:
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
        n = int(seg.GetNumberOfSegments())
        for i in range(n):
            seg_id = seg.GetNthSegmentID(i)
            seg_obj = seg.GetSegment(seg_id)
            seg_name = seg_obj.GetName() if seg_obj is not None else seg_id

            cb = QCheckBox(str(seg_name))
            cb.setChecked(True)
            self._segment_checkbox_by_id[str(seg_id)] = cb
            self._segments_scroll_layout.addWidget(cb)

        self._segments_scroll_layout.addStretch()

    def _selected_segment_ids(self) -> list[str]:
        return [sid for sid, cb in self._segment_checkbox_by_id.items() if cb is not None and cb.isChecked()]

    def _load_png_node(self, path: str, name: str) -> Any | None:
        if slicer.mrmlScene is None or not path or not os.path.exists(path):
            return None
        existing = slicer.mrmlScene.GetFirstNodeByName(name)

        if existing is not None and hasattr(existing, "IsA") and existing.IsA("vtkMRMLScalarVolumeNode"):
            storage = existing.GetStorageNode()
            if storage is not None:
                storage.SetFileName(path)
                storage.ReadData(existing)
                return existing

            if existing.GetScene() == slicer.mrmlScene:
                dn = existing.GetDisplayNode() if hasattr(existing, "GetDisplayNode") else None
                if dn is not None and dn.GetScene() == slicer.mrmlScene:
                    try:
                        slicer.mrmlScene.RemoveNode(dn)
                    except Exception:
                        pass

                sn = existing.GetStorageNode() if hasattr(existing, "GetStorageNode") else None
                if sn is not None and sn.GetScene() == slicer.mrmlScene:
                    try:
                        slicer.mrmlScene.RemoveNode(sn)
                    except Exception:
                        pass
                try:
                    slicer.mrmlScene.RemoveNode(existing)
                except Exception:
                    pass

        n = slicer.util.loadVolume(path, properties={"name": name})
        return n

    def _on_dose_selection_changed(self) -> None:
        dose_a = self._selected_dose_node_a()
        dose_b = self._selected_dose_node_b()

        unc_a = self._find_uncertainty_in_same_folder(dose_a)
        unc_b = self._find_uncertainty_in_same_folder(dose_b)

        self.cb_unc_a.setEnabled(bool(unc_a is not None))
        if unc_a is None:
            self.cb_unc_a.setChecked(False)

        self.cb_unc_b.setEnabled(bool(unc_b is not None))
        if unc_b is None:
            self.cb_unc_b.setChecked(False)

    def _compute_cumulative_dvh(
        self, dose_vals: np.ndarray, bins_edges_gy: np.ndarray, voxel_volume_cc: float
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if dose_vals is None or bins_edges_gy is None:
            return None, None
        vals = np.asarray(dose_vals, dtype=np.float32)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None, None

        hist, _ = np.histogram(vals, bins=bins_edges_gy)
        cum = np.cumsum(hist[::-1])[::-1].astype(np.float32)
        if cum.size == 0 or cum[0] <= 0:
            return None, None
        vcc = np.asarray(cum, dtype=np.float32) * float(voxel_volume_cc)
        vpct = np.asarray(100.0 * cum / float(cum[0]), dtype=np.float32)
        return vpct, vcc

    def _dvh_cleanup_temp_nodes(self) -> None:
        j = self._active_job
        if j is None:
            return
        for tn in j.get("temp_nodes", []):
            if tn is None or slicer.mrmlScene is None:
                continue
            try:
                in_scene = tn.GetScene() == slicer.mrmlScene
            except Exception:
                in_scene = False
            if not in_scene:
                continue

            dn = tn.GetDisplayNode() if hasattr(tn, "GetDisplayNode") else None
            if dn is not None and dn.GetScene() == slicer.mrmlScene:
                slicer.mrmlScene.RemoveNode(dn)
            sn = tn.GetStorageNode() if hasattr(tn, "GetStorageNode") else None
            if sn is not None and sn.GetScene() == slicer.mrmlScene:
                slicer.mrmlScene.RemoveNode(sn)

            try:
                slicer.mrmlScene.RemoveNode(tn)
            except Exception:
                pass

    def _dvh_finish(self, ok: bool, msg: str = "") -> None:
        self._dvh_cleanup_temp_nodes()
        self._active_job = None
        self._set_ui_busy(False)
        self._set_progress(0, visible=False)
        if msg:
            QMessageBox.information(self, "DVH", str(msg)) if ok else QMessageBox.warning(self, "DVH", str(msg))

    def _dvh_fail(self, msg: str) -> None:
        self._dvh_finish(False, msg)

    def _dvh_dose_grid_scaling_from_node(self, n: Any) -> float:
        if n is None or not hasattr(n, "GetAttribute"):
            return 1.0
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
            f = float(v)
            if np.isfinite(f) and f > 0:
                return float(f)
        return 1.0

    def _dvh_schedule_tick(self) -> None:
        try:
            QTimer.singleShot(0, self._dvh_tick)
        except Exception:
            self._dvh_tick()

    def _dvh_tick(self) -> None:
        j2 = self._active_job
        if j2 is None:
            return

        dlabels = j2.get("_dose_labels", [])
        if not dlabels:
            self._dvh_fail("No dose selected.")
            return
        di = int(j2.get("_dose_idx", 0))
        si = int(j2.get("_seg_idx", 0))
        seg_ids2 = j2.get("seg_ids", [])
        total = max(1, len(dlabels) * len(seg_ids2))

        if di >= len(dlabels):
            self._dvh_build_outputs()
            return
        if si >= len(seg_ids2):
            j2["_dose_idx"] = di + 1
            j2["_seg_idx"] = 0
            self._dvh_schedule_tick()
            return

        label = str(dlabels[di])
        seg_id = str(seg_ids2[si])
        j2["_seg_idx"] = si + 1

        dose_nodes = j2.get("dose_nodes", []) or []
        dose_a = dose_nodes[0] if len(dose_nodes) > 0 else None
        dose_b = dose_nodes[1] if len(dose_nodes) > 1 else None

        dose_node = dose_a if label == "A" else dose_b
        dose_arr = j2.get("_dose_arr_by_label", {}).get(label, None)
        if dose_node is None or dose_arr is None:
            self._dvh_schedule_tick()
            return

        # Segment name + color
        seg_name = seg_id
        seg_color = None
        try:
            seg = j2["seg_node"].GetSegmentation()
            seg_obj = seg.GetSegment(seg_id) if seg is not None else None
            seg_name = seg_obj.GetName() if seg_obj is not None else seg_id
            if seg_obj is not None and hasattr(seg_obj, "GetColor"):
                c = seg_obj.GetColor()
                seg_color = (float(c[0]), float(c[1]), float(c[2]))
        except Exception:
            seg_name = seg_id

        mask = self.export_segment_mask(j2["seg_node"], seg_id, dose_node)
        if mask is not None:
            if int(np.count_nonzero(mask)) > 0:
                bins_edges = j2.get("_bins_edges")
                vox_cc = float(j2.get("_vox_cc_by_label", {}).get(label, 1.0))
                scale = float(j2.get("_dose_scale_by_label", {}).get(label, 1.0))
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

        done = min(total, (di * len(seg_ids2)) + (si + 1))
        p = 5 + int(80 * float(done) / float(total))
        self._set_progress(min(90, p), visible=True)
        self._set_status(f"Computing DVH… ({done}/{total})")

        self._dvh_schedule_tick()

    def _dvh_create_or_get_table_node(self, name: str) -> Any | None:
        if slicer.mrmlScene is None:
            return None
        node = slicer.mrmlScene.GetFirstNodeByName(name)
        if node is not None and hasattr(node, "IsA") and node.IsA("vtkMRMLTableNode"):
            return node
        return slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", name)

    def _dvh_safe_col_base(self, s: str) -> str:
        txt = str(s)
        txt = txt.replace("σ", "sigma").replace("±", "+/-")
        try:
            import re

            txt = re.sub(r"[^0-9A-Za-z _|:+().-]", "_", txt)
        except Exception:
            pass
        return txt

    def _dvh_safe_png_name(self, name: str) -> str:
        s = str(name or "")
        if not s:
            s = "dvh"
        try:
            import re

            s = re.sub(r"[<>:\"/\\|?*]", "_", s)
        except Exception:
            s = s.replace(":", "_").replace("/", "_").replace("\\", "_")
        s = s.strip("._ ")
        return s or "dvh"

    def _dvh_render_matplotlib_png(
        self, table_name: str, centers2: np.ndarray, curves_cached: list[dict[str, Any]]
    ) -> str | None:
        try:
            import matplotlib
        except Exception:
            logger.exception("Matplotlib is not downloaded; download it to enable DVH plotting")
            slicer.util.pip_install("matplotlib")
            import matplotlib

        matplotlib.use("Agg")
        matplotlib.set_loglevel("warning")
        import tempfile

        import matplotlib.pyplot as plt

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

        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for c in curves_cached:
            key = (c.get("_dose_label", ""), c.get("_seg_id", ""))
            grouped.setdefault(key, {})[c.get("_kind", "mean") or "mean"] = c

        structure_handles: list[Any] = []
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
            rgb = mean_c.get("_color", None) or (0.0, 0.0, 0.0)
            struct_label = mean_c.get("_seg_name", "")

            upper_c = ck.get("+3sigma")
            lower_c = ck.get("-3sigma")
            if upper_c is not None and lower_c is not None:
                y_up = np.asarray(upper_c.get("vpct"), dtype=np.float32)
                y_lo = np.asarray(lower_c.get("vpct"), dtype=np.float32)
                if y_up.size == xs.size and y_lo.size == xs.size:
                    ax.fill_between(xs, y_lo, y_up, color=rgb, alpha=0.20, linewidth=0)
                    if unc_patch_handle is None:
                        from matplotlib.patches import Patch

                        unc_patch_handle = Patch(
                            facecolor="#9ca3af", edgecolor="#4b5563", alpha=0.35, label="Uncertainty (±3σ)"
                        )

            line_style = "--" if dose_label == "B" else "-"
            ax.plot(xs, y_mean, color=rgb, lw=1.8, alpha=1.0, linestyle=line_style, label=struct_label)
            line_handle = ax.lines[-1]
            if struct_label and struct_label not in added_struct_labels:
                added_struct_labels.add(struct_label)
                structure_handles.append(line_handle)
                structure_labels.append(struct_label)

        legend_handles = []
        legend_labels = []

        legend_handles.extend(structure_handles)
        legend_labels.extend(structure_labels)

        from matplotlib.lines import Line2D

        legend_handles.append(Line2D([], [], color="#111827", lw=1.8, linestyle="-", label="Dose ref"))
        legend_labels.append("Dose ref")
        legend_handles.append(Line2D([], [], color="#111827", lw=1.8, linestyle="--", label="Dose estimated"))
        legend_labels.append("Dose estimated")

        if unc_patch_handle is not None:
            legend_handles.append(unc_patch_handle)
            legend_labels.append("Uncertainty (±3σ)")

        ax.legend(legend_handles, legend_labels, loc="best", fontsize=8, frameon=True, fancybox=True)
        fig.tight_layout()

        buf = io.BytesIO()
        try:
            fig.savefig(buf, format="png", dpi=110)
        except Exception:
            logger.exception("Failed to render matplotlib figure")
            return None
        try:
            out_name = f"{self._dvh_safe_png_name(table_name)}.png"
            out_path = os.path.join(tempfile.gettempdir(), out_name)
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

    def _dvh_add_line_series(
        self,
        table_node: Any,
        chart_node: Any,
        table_name: str,
        x_col_name: str,
        series_index: int,
        y_col: str,
        rgb: tuple[float, float, float] | None,
        name_suffix: str,
        width: int,
        dashed: bool,
        opacity: float | None = None,
        created_series_nodes: list[Any] | None = None,
    ) -> int:
        sname = f"{table_name}_{series_index:02d}_{name_suffix}"
        series_index += 1
        s = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", sname)
        if s is None:
            return series_index
        try:
            s.SetAndObserveTableNodeID(table_node.GetID())
            if hasattr(s, "SetXColumnName"):
                s.SetXColumnName(x_col_name)
            if hasattr(s, "SetYColumnName"):
                s.SetYColumnName(y_col)
            s.SetPlotType(s.PlotTypeLine)
            s.SetMarkerStyle(s.MarkerStyleNone)
        except Exception:
            pass
        if dashed:
            s.SetLineStyle(s.LineStyleDash)
        else:
            s.SetLineStyle(s.LineStyleSolid)
        if hasattr(s, "SetLineWidth"):
            s.SetLineWidth(int(width))
        if rgb is not None and hasattr(s, "SetColor"):
            s.SetColor(float(rgb[0]), float(rgb[1]), float(rgb[2]))

        if opacity is not None:
            self._apply_series_opacity(s, float(opacity))
        s.Modified()
        chart_node.AddAndObservePlotSeriesNodeID(s.GetID())
        if created_series_nodes is not None:
            created_series_nodes.append(s)
        return series_index

    def _dvh_find_child_folder(self, sh_node: Any, parent: int, name: str) -> int:
        try:
            ids = vtk.vtkIdList()
            sh_node.GetItemChildren(parent, ids)
            for ii in range(ids.GetNumberOfIds()):
                child = ids.GetId(ii)
                if sh_node.GetItemName(child) == name:
                    return int(child)
        except Exception:
            pass
        return 0

    def _dvh_reparent_node(self, sh_node: Any, node: Any, parent_folder_item: int) -> None:
        if node is None or not parent_folder_item:
            return
        item = int(sh_node.GetItemByDataNode(node) or 0)
        if not item:
            return
        sh_node.SetItemParent(item, int(parent_folder_item))

    def _dvh_reparent_outputs(
        self,
        table_node: Any,
        chart_node: Any,
        created_series_nodes: list[Any] | None,
        table_name: str,
        png_path: str | None,
    ) -> None:
        sh_node = self._get_sh_node()
        if sh_node is None:
            return
        parent_item = 0
        dose_nodes = list((self._active_job or {}).get("dose_nodes") or [])
        dose_a = dose_nodes[0] if len(dose_nodes) > 0 else None

        if dose_a is not None:
            dose_item = int(sh_node.GetItemByDataNode(dose_a) or 0)
            if dose_item:
                parent_item = int(sh_node.GetItemParent(dose_item) or 0)

        if not parent_item:
            parent_item = int(sh_node.GetSceneItemID() or 0)

        dvh_root = self._dvh_find_child_folder(sh_node, parent_item, "DVH")
        if not dvh_root:
            dvh_root = int(sh_node.CreateFolderItem(parent_item, "DVH") or 0)

        run_folder_name = str(table_name)
        run_folder = self._dvh_find_child_folder(sh_node, dvh_root, run_folder_name) if dvh_root else 0
        if dvh_root and not run_folder:
            run_folder = int(sh_node.CreateFolderItem(dvh_root, run_folder_name) or 0)

        target_folder = run_folder or dvh_root
        self._dvh_reparent_node(sh_node, table_node, target_folder)
        self._dvh_reparent_node(sh_node, chart_node, target_folder)
        for s in created_series_nodes or []:
            self._dvh_reparent_node(sh_node, s, target_folder)

        if png_path:
            png_node = self._load_png_node(png_path, f"{table_name}_png")
            self._dvh_reparent_node(sh_node, png_node, target_folder)

    def _dvh_build_outputs(self) -> None:
        j2 = self._active_job
        if j2 is None:
            return
        self._set_status("Building table/plot…")
        self._set_progress(92, visible=True)

        table_name = str(j2.get("out_base"))
        table_node = self._dvh_create_or_get_table_node(table_name)
        if table_node is None:
            self._dvh_fail("Could not create output TableNode.")
            return

        table = table_node.GetTable()
        table.Initialize()

        centers2 = np.asarray(j2.get("_bins_centers"), dtype=np.float32)
        nrows = int(centers2.size)

        xcol = vtk.vtkFloatArray()
        xcol.SetName("dose_gy")
        xcol.SetNumberOfTuples(nrows)
        table.AddColumn(xcol)

        table.SetNumberOfRows(nrows)
        for r in range(nrows):
            xcol.SetValue(r, float(centers2[r]))

        if hasattr(table, "SetColumnName"):
            table.SetColumnName(0, "dose_gy")

        c0 = table.GetColumn(0)
        if c0 is not None and hasattr(c0, "SetName"):
            c0.SetName("dose_gy")

        curves = list(j2.get("curves", []))
        curves_cached = []
        for c in curves:
            d = dict(c)
            d["_label"] = str(d.get("label", "curve"))
            d["_seg_name"] = str(d.get("seg_name", "") or "")
            d["_dose_label"] = str(d.get("dose_label", "") or "")
            d["_kind"] = str(d.get("kind", "") or "")
            d["_seg_id"] = str(d.get("seg_id", "") or "")
            d["_color"] = d.get("color", None)
            curves_cached.append(d)
        col_specs = []

        for c in curves_cached:
            col_base = self._dvh_safe_col_base(c.get("_label", "curve"))
            col_vpct = f"{col_base} | V%"
            col_vcc = f"{col_base} | Vcc"
            vp = vtk.vtkFloatArray()
            vp.SetName(col_vpct)
            vp.SetNumberOfTuples(nrows)
            vc = vtk.vtkFloatArray()
            vc.SetName(col_vcc)
            vc.SetNumberOfTuples(nrows)
            table.AddColumn(vp)
            table.AddColumn(vc)
            col_specs.append((col_vpct, col_vcc, c))

        for ci, (_, _, c) in enumerate(col_specs):
            vpct = np.asarray(c.get("vpct"), dtype=np.float32)
            vcc = np.asarray(c.get("vcc"), dtype=np.float32)
            if vpct.size != nrows:
                continue
            for r in range(nrows):
                table.SetValue(r, 1 + (2 * ci), float(vpct[r]))
                table.SetValue(r, 1 + (2 * ci) + 1, float(vcc[r]))

        table_node.Modified()
        table.Modified()

        png_path = None
        try:
            png_path = self._dvh_render_matplotlib_png(table_name, centers2, curves_cached)
        except Exception:
            png_path = None
        if png_path:
            j2["png_path"] = png_path

        x_col_name = "dose_gy"
        if self.plot_widget is not None:
            self.plot_widget.setVisible(True)

        chart_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode", f"{table_name}_chart")

        created_series_nodes: list[Any] = []
        if chart_node is not None:
            self._set_chart_legend_visible(chart_node, False)
            chart_node.RemoveAllPlotSeriesNodeIDs()

            series_index = 0
            for _, (col_vpct, _, c) in enumerate(col_specs):
                kind = c.get("_kind", "")
                dose_label = c.get("_dose_label", "")
                rgb = c.get("_color", None) or (0.0, 0.0, 0.0)
                dashed = dose_label == "B"
                if "sigma" in kind:
                    series_index = self._dvh_add_line_series(
                        table_node,
                        chart_node,
                        table_name,
                        x_col_name,
                        series_index,
                        col_vpct,
                        rgb,
                        "unc",
                        width=2,
                        dashed=False,
                        opacity=0.35,
                        created_series_nodes=created_series_nodes,
                    )
                else:
                    series_index = self._dvh_add_line_series(
                        table_node,
                        chart_node,
                        table_name,
                        x_col_name,
                        series_index,
                        col_vpct,
                        rgb,
                        "mean",
                        width=2,
                        dashed=dashed,
                        opacity=None,
                        created_series_nodes=created_series_nodes,
                    )

            chart_node.SetTitle(table_name)
            chart_node.SetXAxisTitle("Dose (Gy)")
            chart_node.SetYAxisTitle("Volume (%)")

            x_max = float(centers2[-1]) if int(nrows) > 0 else None
            if x_max is not None and x_max > 0:
                for m in ("SetXAxisRange", "SetXRange"):
                    if hasattr(chart_node, m):
                        getattr(chart_node, m)(0.0, float(x_max))
                        break

            self._set_plot_chart_on_widget(chart_node)

        structure_items = []
        seen = set()
        for c in list(curves_cached):
            name = c.get("_seg_name", "")
            color = c.get("_color", None) or (0.0, 0.0, 0.0)
            if name and name not in seen:
                seen.add(name)
                structure_items.append({"name": name, "color": color})
        self._update_legend(structure_items=structure_items)

        self._dvh_reparent_outputs(table_node, chart_node, created_series_nodes, table_name, png_path)
        self._set_progress(100, visible=False)
        self._set_status("Done.")
        self._dvh_finish(True, f"DVH written to table: {table_name}")

    def _on_compute_dvh(self) -> None:
        if self._active_job is not None:
            QMessageBox.information(self, "Busy", "A DVH computation is already running.")
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
        unc_b = (
            self._find_uncertainty_in_same_folder(dose_b)
            if (dose_b is not None and bool(self.cb_unc_b.isChecked()))
            else None
        )

        out_base = self._line_edit_text(self.output_name_edit).strip() or f"dvh_{uuid4().hex[:6]}"

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
        }

        self._set_ui_busy(True)
        self._set_status("Preparing…")
        self._set_progress(0, visible=True)

        # Read dose arrays once (fast path) and pick automatic bins based on max dose.
        j = self._active_job
        dose_arrays = []
        dose_labels = []
        dose_arrays.append(np.asarray(slicer.util.arrayFromVolume(dose_a), dtype=np.float32))
        dose_labels.append("A")

        if dose_b is not None:
            dose_arrays.append(np.asarray(slicer.util.arrayFromVolume(dose_b), dtype=np.float32))
            dose_labels.append("B")

        # Update job dose list in case B failed to load.
        j["dose_nodes"] = [dose_a] + ([dose_b] if dose_b is not None else [])

        # Prefer DICOM RTDOSE DoseGridScaling when present; otherwise fallback to heuristic.
        j["_dose_scale_by_label"] = {}
        max_dose_raw = 0.0
        max_dose_gy = 0.0
        for label, node, arr in zip(dose_labels, j["dose_nodes"], dose_arrays):
            m = float(np.nanmax(arr))
            if np.isfinite(m):
                max_dose_raw = max(max_dose_raw, m)

            s = float(self._dvh_dose_grid_scaling_from_node(node))

            # Heuristic fallback if scaling attribute is absent/unity.
            if abs(float(s) - 1.0) < 1e-12:
                hs = 1.0
                for _ in range(12):
                    if not np.isfinite(m):
                        break
                    if (float(m) * float(hs)) > 200.0:
                        hs *= 0.1
                        continue
                    break
                s = float(hs)

            j["_dose_scale_by_label"][label] = float(s)
            if np.isfinite(m):
                max_dose_gy = max(max_dose_gy, float(m) * float(s))

        if not np.isfinite(max_dose_raw) or max_dose_raw <= 0.0:
            self._dvh_fail("Max dose is 0 or invalid.")
            return

        max_dose = float(max_dose_gy)

        edges = np.arange(0.0, float(max_dose) + 1.0, 1.0, dtype=np.float32)
        if edges.size < 2:
            self._dvh_fail("Failed to build DVH bins.")
            return
        centers = np.asarray(edges[:-1], dtype=np.float32)
        j["_bins_edges"] = edges
        j["_bins_centers"] = centers

        # Resolve per-dose uncertainty arrays (if requested).
        j["_dose_arr_by_label"] = {}
        j["_unc_arr_by_label"] = {}
        for label, darr in zip(dose_labels, dose_arrays):
            j["_dose_arr_by_label"][label] = darr
            unc_node = unc_a if label == "A" else unc_b
            if unc_node is not None:
                uarr = np.asarray(slicer.util.arrayFromVolume(unc_node), dtype=np.float32)
                j["_unc_arr_by_label"][label] = uarr
            else:
                j["_unc_arr_by_label"][label] = None

        # Pre-compute voxel volume in cc per dose label (use each dose spacing).
        j["_vox_cc_by_label"] = {}
        for label, node in zip(dose_labels, j["dose_nodes"]):
            sx, sy, sz = (float(v) for v in node.GetSpacing())
            j["_vox_cc_by_label"][label] = (sx * sy * sz) * 1e-3

        # Async loop over (dose label, segment).
        j["_dose_labels"] = list(dose_labels)
        j["_dose_idx"] = 0
        j["_seg_idx"] = 0

        self._set_status("Computing DVH…")
        self._set_progress(5, visible=True)

        self._dvh_schedule_tick()
