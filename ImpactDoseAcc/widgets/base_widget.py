from uuid import uuid4
import os
from typing import Iterable, Optional

import numpy as np
import slicer
import vtk
from qt import QVBoxLayout, QWidget


class BaseImpactWidget(QWidget):

    def _is_name_match(self, node, needle: str) -> bool:
        try:
            return str(needle).lower() in self._safe_node_name(node).lower()
        except Exception:
            return False

    def __init__(self, logic=None):
        super().__init__()
        self.logic = logic
        self._root_widget = None
        self.ui = None

    def _combo_current_index(self, combo) -> int:
        """Return currentIndex from a QComboBox in a PythonQt-safe way."""
        if combo is None:
            return 0
        idx_attr = getattr(combo, "currentIndex", 0)
        try:
            return int(idx_attr() if callable(idx_attr) else idx_attr)
        except Exception:
            return 0

    # --- UI helpers ---
    def load_ui(self, ui_path: str):
        ui_widget = slicer.util.loadUI(os.path.normpath(ui_path))
        self.ui = slicer.util.childWidgetVariables(ui_widget)
        self._root_widget = ui_widget
        layout = QVBoxLayout(self)
        layout.addWidget(ui_widget)
        self.setLayout(layout)
        return ui_widget

    def _w(self, name):
        return getattr(self.ui, name, None) if hasattr(self, "ui") else None

    def _layout(self, widget_name):
        w = self._w(widget_name)
        return w.layout() if w is not None and hasattr(w, "layout") else None

    def _btn(self, name, cb):
        btn = self._w(name)
        if btn is not None and cb:
            btn.clicked.connect(cb)
        return btn

    def _line_edit_text(self, line_edit) -> str:
        if line_edit is None:
            return ""
        text_attr = getattr(line_edit, "text", "")
        try:
            val = text_attr() if callable(text_attr) else text_attr
            return "" if val is None else str(val)
        except Exception:
            return ""

    def _safe_node_name(self, node) -> str:
        if node is None or not hasattr(node, "GetName"):
            return ""
        try:
            return node.GetName() or ""
        except Exception:
            return ""

    def _set_status(self, text: str) -> None:
        try:
            if hasattr(self, "status_label") and self.status_label is not None:
                self.status_label.setText(text or "")
        except Exception:
            pass

    def _set_progress(self, value, visible: bool = True) -> None:
        try:
            if not hasattr(self, "progress_bar") or self.progress_bar is None:
                return
            if value is None:
                self.progress_bar.setRange(0, 0)
            else:
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(int(max(0, min(100, value))))
            self.progress_bar.setVisible(bool(visible))
        except Exception:
            pass

    def _set_ui_busy(self, busy: bool) -> None:
        target = getattr(self, "_root_widget", None) or self
        try:
            target.setEnabled(not bool(busy))
        except Exception:
            pass

    def _generate_default_output_name(self, prefix: str = "out") -> str:
        return f"{prefix}_{uuid4().hex[:2]}"

    # --- MRML helpers ---
    def _get_sh_node(self):
        if slicer.mrmlScene is None:
            return None
        try:
            return slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        except Exception:
            return None

    def export_segment_mask(self, segmentation_node, segment_id: str, reference_volume_node):
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
            "vtkMRMLLabelMapVolumeNode", f"tmp_dvh_{uuid4().hex[:3]}"
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

    def dose_grid_scaling_from_node(self, node) -> float:
        if node is None or not hasattr(node, "GetAttribute"):
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
                v = node.GetAttribute(k)
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

    def set_background_volume(self, volume_node, layout: str = "FourUp", viewers: Iterable[str] = ("Red", "Yellow", "Green")):
        if slicer.app is None or slicer.mrmlScene is None or volume_node is None:
            return
        try:
            lm = slicer.app.layoutManager()
            if lm is None:
                return
            try:
                layout_const = getattr(slicer.vtkMRMLLayoutNode, f"SlicerLayout{layout}View", None)
                if layout_const is not None:
                    lm.setLayout(layout_const)
            except Exception:
                pass
            for view_name in viewers:
                try:
                    sw = lm.sliceWidget(view_name)
                    if sw is None:
                        continue
                    comp = sw.sliceLogic().GetSliceCompositeNode()
                    comp.SetBackgroundVolumeID(volume_node.GetID())
                except Exception:
                    pass
            try:
                lm.resetSliceViews()
            except Exception:
                pass
        except Exception:
            pass

    def get_or_add_node(self, name: str, class_name: str):
        if slicer.mrmlScene is None:
            return None
        try:
            node = slicer.mrmlScene.GetFirstNodeByName(name)
        except Exception:
            node = None
        if node is not None and hasattr(node, "IsA") and node.IsA(class_name):
            return node
        try:
            return slicer.mrmlScene.AddNewNodeByClass(class_name, name)
        except Exception:
            return None

    def safe_remove(self, node) -> None:
        try:
            if node is not None and slicer.mrmlScene is not None and node.GetScene() == slicer.mrmlScene:
                slicer.mrmlScene.RemoveNode(node)
        except Exception:
            pass
