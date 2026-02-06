import logging
from collections.abc import Callable
from typing import Any
from uuid import uuid4

import numpy as np
import slicer
import vtk
from qt import QTimer, QWidget

logger = logging.getLogger(__name__)


class BaseImpactWidget(QWidget):

    def _is_name_match(self, node, needle: str) -> bool:
        return str(needle).lower() in self._safe_node_name(node).lower()

    def _run_cli_async(self, cli_module, params: dict, on_done, on_error) -> None:
        """Run a Slicer CLI without blocking the UI. Shared implementation for widgets."""
        try:
            cli_node = slicer.cli.run(cli_module, None, params, wait_for_completion=False)
        except Exception as exc:
            logger.exception("Failed to start CLI")
            on_error(exc)
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

        def _finish(ok: bool, err: Exception | None = None):
            if holder["handled"]:
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

            QTimer.singleShot(0, _do_finish)

        def _status_tuple():
            try:
                return (cli_node.GetStatus(), str(cli_node.GetStatusString()))
            except Exception:
                try:
                    return (None, str(cli_node.GetStatusString()))
                except Exception:
                    return (None, "")

        def _on_modified(_caller, _event):
            status, status_str = _status_tuple()

            completed = hasattr(cli_node, "Completed") and status == cli_node.Completed
            failed = hasattr(cli_node, "Failed") and status == cli_node.Failed
            cancelled = hasattr(cli_node, "Cancelled") and status == cli_node.Cancelled

            s = (status_str or "").lower()
            if s in ("completed", "completed with errors"):
                completed = True
            elif s == "failed":
                failed = True
            elif s in ("cancelled", "canceled"):
                cancelled = True

            if not (completed or failed or cancelled):
                return

            if failed:
                try:
                    msg = str(cli_node.GetErrorText())
                except Exception:
                    msg = None
                _finish(False, RuntimeError(msg or "CLI failed"))
            elif cancelled:
                _finish(False, RuntimeError("CLI cancelled"))
            else:
                _finish(True)

        try:
            holder["tag"] = cli_node.AddObserver(vtk.vtkCommand.ModifiedEvent, _on_modified)
        except Exception as exc:
            logger.exception("Failed to add observer to CLI node")
            _cleanup()
            on_error(exc)

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

    def _w(self, name: str) -> Any:
        return getattr(self.ui, name, None) if hasattr(self, "ui") else None

    def _layout(self, widget_name: str) -> Any:
        w = self._w(widget_name)
        return w.layout() if w is not None and hasattr(w, "layout") else None

    def _btn(self, name: str, cb: Callable[[], None] | None) -> Any:
        btn = self._w(name)
        if btn is not None and cb:
            btn.clicked.connect(cb)
        return btn

    def _line_edit_text(self, line_edit: Any) -> str:
        if line_edit is None:
            return ""
        text_attr = getattr(line_edit, "text", "")
        val = text_attr() if callable(text_attr) else text_attr
        return "" if val is None else str(val)

    def _safe_node_name(self, node: Any) -> str:
        if node is None or not hasattr(node, "GetName"):
            return ""
        return node.GetName() or ""

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
        target.setEnabled(not bool(busy))

    def _generate_default_output_name(self, prefix: str) -> str:
        return f"{prefix}_{uuid4().hex[:2]}"

    # --- MRML helpers ---
    def _get_sh_node(self) -> Any:
        if slicer.mrmlScene is None:
            return None
        try:
            return slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        except Exception:
            logger.exception("_get_sh_node failed")
            return None

    def export_segment_mask(
        self, segmentation_node: Any, segment_id: str, reference_volume_node: Any
    ) -> np.ndarray | None:
        if slicer.mrmlScene is None or segmentation_node is None or reference_volume_node is None:
            return None
        fn = getattr(slicer.util, "arrayFromSegmentBinaryLabelmap", None)
        if callable(fn):
            arr = fn(segmentation_node, segment_id, reference_volume_node)
            if arr is None:
                return None
            return np.asarray(arr) > 0

        labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", f"tmp_dvh_{uuid4().hex[:3]}")
        labelmap.SetHideFromEditors(1)
        labelmap.SetSelectable(0)
        labelmap.SetSaveWithScene(0)

        try:
            seg_logic = slicer.modules.segmentations.logic()
            seg_ids = vtk.vtkStringArray()
            seg_ids.InsertNextValue(str(segment_id))
            seg_logic.ExportSegmentsToLabelmapNode(segmentation_node, seg_ids, labelmap, reference_volume_node)
            arr = slicer.util.arrayFromVolume(labelmap)
            return np.asarray(arr) > 0
        except Exception:
            logger.exception("ExportSegmentsToLabelmapNode or arrayFromVolume failed")
            return None
        finally:
            try:
                if labelmap is not None and labelmap.GetScene() == slicer.mrmlScene:
                    slicer.mrmlScene.RemoveNode(labelmap)
            except Exception:
                logger.exception("Failed to remove temporary labelmap node")

    def safe_remove(self, node: Any) -> None:
        try:
            if node is not None and slicer.mrmlScene is not None and node.GetScene() == slicer.mrmlScene:
                slicer.mrmlScene.RemoveNode(node)
        except Exception:
            logger.exception("safe_remove failed")

    def _find_uncertainty_in_same_folder(self, dose_node: Any) -> Any:
        """Return an uncertainty volume that lives in the same Subject Hierarchy folder as the given dose volume.

        Preference order: prefer names matching 'uncertainty_{base}' or 'uncertainty_dose_{base}',
        otherwise return first uncertainty in the same folder.
        """
        if slicer.mrmlScene is None or dose_node is None:
            return None
        sh_node = self._get_sh_node()
        if sh_node is None:
            return None
        dose_item = int(sh_node.GetItemByDataNode(dose_node) or 0)
        if not dose_item:
            return None
        parent = int(sh_node.GetItemParent(dose_item) or 0)
        if not parent:
            return None

        # base name candidates
        name = self._safe_node_name(dose_node)
        base = name.split("dose_")[-1] if "dose_" in name else name
        base = base.strip()

        preferred = []
        if base:
            preferred = [f"uncertainty_{base}", f"uncertainty_dose_{base}"]

        ids = vtk.vtkIdList()
        try:
            sh_node.GetItemChildren(parent, ids)
        except Exception:
            return None

        first_unc = None
        for i in range(ids.GetNumberOfIds()):
            child = int(ids.GetId(i))
            try:
                n = sh_node.GetItemDataNode(child)
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
