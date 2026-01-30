import logging
from uuid import uuid4
import os
from typing import Iterable, Optional

import numpy as np
import slicer
import vtk
from qt import QVBoxLayout, QWidget, QTimer

logger = logging.getLogger(__name__)


class BaseImpactWidget(QWidget):

    def _is_name_match(self, node, needle: str) -> bool:
        try:
            return str(needle).lower() in self._safe_node_name(node).lower()
        except Exception as e:
            logger.exception("_is_name_match failed")
            return False

    def _run_cli_async(self, cli_module, params: dict, on_done, on_error) -> None:
        """Run a Slicer CLI without blocking the UI. Shared implementation for widgets.

        on_done/on_error are callables that will be invoked when the job finishes.
        """
        try:
            cli_node = slicer.cli.run(cli_module, None, params, wait_for_completion=False)
        except Exception as exc:
            logger.exception("Failed to start CLI")
            try:
                on_error(exc)
            except Exception:
                logger.exception("on_error raised while handling CLI start failure")
            return

        holder = {"tag": None, "handled": False}

        def _cleanup():
            try:
                if holder["tag"] is not None:
                    cli_node.RemoveObserver(holder["tag"])
            except Exception:
                logger.exception("Error removing CLI observer")
            try:
                if cli_node is not None and cli_node.GetScene() == slicer.mrmlScene:
                    slicer.mrmlScene.RemoveNode(cli_node)
            except Exception:
                logger.exception("Error removing CLI node from scene")

        def _finish(ok: bool, err: Exception = None):
            if holder.get("handled", False):
                return
            holder["handled"] = True

            def _do_finish():
                _cleanup()
                try:
                    if ok:
                        on_done()
                    else:
                        on_error(err or RuntimeError("CLI failed"))
                except Exception:
                    logger.exception("on_done/on_error raised during CLI finish handling")

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
            logger.exception("Failed to add observer to CLI node")
            _cleanup()
            try:
                on_error(exc)
            except Exception:
                logger.exception("on_error raised while handling observer add failure")
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
        return int(idx_attr() if callable(idx_attr) else idx_attr)

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
        except Exception as e:
            logger.exception("_line_edit_text failed")
            return ""

    def _safe_node_name(self, node) -> str:
        if node is None or not hasattr(node, "GetName"):
            return ""
        try:
            return node.GetName() or ""
        except Exception as e:
            logger.exception("_safe_node_name failed")
            return ""

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text or "")

    def _set_progress(self, value, visible: bool = True) -> None:
        if not hasattr(self, "progress_bar") or self.progress_bar is None:
            return
        if value is None:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(int(max(0, min(100, value))))
        self.progress_bar.setVisible(bool(visible))

    def _set_ui_busy(self, busy: bool) -> None:
        target = getattr(self, "_root_widget", None) or self
        try:
            target.setEnabled(not bool(busy))
        except Exception as e:
            logger.exception("_set_ui_busy failed")

    def _generate_default_output_name(self, prefix: str = "out") -> str:
        return f"{prefix}_{uuid4().hex[:2]}"

    # --- MRML helpers ---
    def _get_sh_node(self):
        if slicer.mrmlScene is None:
            return None
        try:
            return slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        except Exception as e:
            logger.exception("_get_sh_node failed")
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
        except Exception as e:
            logger.exception("arrayFromSegmentBinaryLabelmap call failed")

        labelmap = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", f"tmp_dvh_{uuid4().hex[:3]}"
        )
        try:
            labelmap.SetHideFromEditors(1)
            labelmap.SetSelectable(0)
            labelmap.SetSaveWithScene(0)
        except Exception as e:
            logger.exception("Failed to set labelmap properties")

        try:
            seg_logic = slicer.modules.segmentations.logic()
            seg_ids = vtk.vtkStringArray()
            seg_ids.InsertNextValue(str(segment_id))
            seg_logic.ExportSegmentsToLabelmapNode(segmentation_node, seg_ids, labelmap, reference_volume_node)
            arr = slicer.util.arrayFromVolume(labelmap)
            return np.asarray(arr) > 0
        except Exception as e:
            logger.exception("ExportSegmentsToLabelmapNode or arrayFromVolume failed")
            return None
        finally:
            try:
                if labelmap is not None and labelmap.GetScene() == slicer.mrmlScene:
                    slicer.mrmlScene.RemoveNode(labelmap)
            except Exception:
                logger.exception("Failed to remove temporary labelmap node")

    def dose_grid_scaling_from_node(self, node) -> float:
        if node is None or not hasattr(node, "GetAttribute"):
            return 1.0
        keys = (
            "DoseGridScaling",
            "DICOM.DoseGridScaling",
            "DICOM.RTDOSE.DoseGridScaling",
            "RTDOSE.DoseGridScaling",
            "DicomRtDoseGridScaling")
        
        for k in keys:
            v = node.GetAttribute(k)
            if not v:
                continue
            f = float(v)
            if np.isfinite(f) and f > 0:
                return float(f)

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
                logger.exception("set_background_volume: failed while setting layout")
            for view_name in viewers:
                try:
                    sw = lm.sliceWidget(view_name)
                    if sw is None:
                        continue
                    comp = sw.sliceLogic().GetSliceCompositeNode()
                    comp.SetBackgroundVolumeID(volume_node.GetID())
                except Exception as e:
                    logger.exception("set_background_volume: failed while setting background for view %s", view_name)
            try:
                lm.resetSliceViews()
            except Exception as e:
                logger.exception("set_background_volume: resetSliceViews failed")
        except Exception:
            pass

    def get_or_add_node(self, name: str, class_name: str):
        if slicer.mrmlScene is None:
            return None
        try:
            node = slicer.mrmlScene.GetFirstNodeByName(name)
        except Exception as e:
            logger.exception("get_or_add_node: GetFirstNodeByName failed for %s", name)
            node = None
        if node is not None and hasattr(node, "IsA") and node.IsA(class_name):
            return node
        try:
            return slicer.mrmlScene.AddNewNodeByClass(class_name, name)
        except Exception as e:
            logger.exception("get_or_add_node: AddNewNodeByClass failed for %s (%s)", (name, class_name))
            return None

    def safe_remove(self, node) -> None:
        try:
            if node is not None and slicer.mrmlScene is not None and node.GetScene() == slicer.mrmlScene:
                slicer.mrmlScene.RemoveNode(node)
        except Exception as e:
            logger.exception("safe_remove failed")

    def _find_uncertainty_in_same_folder(self, dose_node):
        """Return an uncertainty volume that lives in the same Subject Hierarchy folder as the given dose volume.

        Preference order: prefer names matching 'uncertainty_{base}' or 'uncertainty_dose_{base}', otherwise return first uncertainty in the same folder.
        """
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

        # base name candidates
        name = self._safe_node_name(dose_node)
        base = None
        try:
            # simple heuristic similar to other widgets
            base = name.split("dose_")[-1] if "dose_" in name else name
            base = base.strip()
        except Exception:
            base = None

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
            nm = self._safe_node_name(n).lower()
            if "uncertainty" not in nm:
                continue
            if first_unc is None:
                first_unc = n
            for pref in preferred:
                if self._safe_node_name(n) == pref:
                    return n
        return first_unc
