# Import Qt classes nécessaires
import logging
import os
from typing import Any
from uuid import uuid4

import numpy as np
import slicer
import vtk
from qt import QCheckBox, QDoubleSpinBox, QHBoxLayout, QLabel, QMessageBox, QTimer, QVBoxLayout, QWidget
from widgets.base_widget import BaseImpactWidget

logger = logging.getLogger(__name__)


class DoseAccumulationWidget(BaseImpactWidget):

    def __init__(self) -> None:
        super().__init__()
        self._patient_item_id_map: dict[int, int] = {}
        self._proxy_nodes_by_id: dict[int, Any] = {}
        self._fraction_checkboxes: list[QCheckBox] = []
        self._weight_spinboxes_by_id: dict[int, QDoubleSpinBox] = {}
        self._active_job: dict[str, Any] | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        ui_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "../Resources/UI/AccumulationWidget.ui"))
        ui_widget = slicer.util.loadUI(ui_path)
        self.ui = slicer.util.childWidgetVariables(ui_widget)
        self._root_widget = ui_widget

        # Bind widgets
        self.patient_combo = self._w("patient_combo")
        self.fractions_scroll = self._w("fractions_scroll")
        self.fractions_container = self._w("fractions_container")
        self.fractions_container_layout = self._layout("fractions_container")
        self.strategy_combo = self._w("strategy_combo")
        self.output_name_edit = self._w("output_name_edit")
        self.run_btn = self._w("run_btn")
        self.status_label = self._w("status_label")
        self.progress_bar = self._w("progress_bar")

        # Buttons & signals
        self._btn("refresh_btn", self._refresh_all)
        if self.patient_combo is not None:
            self.patient_combo.currentIndexChanged.connect(self._on_patient_changed)
        if self.run_btn is not None:
            self.run_btn.clicked.connect(self._on_compute_accumulation)

        # Strategy options
        if self.strategy_combo is not None:
            self.strategy_combo.clear()
            self.strategy_combo.addItem("Dose accumulation (Dose Acc only)")
            self.strategy_combo.addItem("Classic Uncertainty Aware (Dose Acc + Uncertainty)")
            self.strategy_combo.addItem("Robust uncertainty (remove top-k outliers)")
            self.strategy_combo.addItem("Continuous anatomy change (e.g., parotids) – uncertainty weighting 0.5→2")
            self.strategy_combo.addItem("DVF magnitude-driven (anatomy) – uncertainty weighting from dvf_magnitude")

        if self.output_name_edit is not None:
            self.output_name_edit.setText(self._generate_default_output_name(prefix="dose_acc"))

        if self.progress_bar is not None:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(False)

        layout = QVBoxLayout(self)
        layout.addWidget(ui_widget)
        self.setLayout(layout)

        self._refresh_all()

    def _find_uncertainty_node(self, base: str) -> Any | None:
        """Find the uncertainty volume corresponding to a Phase-1 output base."""
        if slicer.mrmlScene is None:
            return None
        b = str(base or "").strip()
        if not b:
            return None

        candidates = (f"uncertainty_{b}", f"uncertainty_dose_{b}")
        for name in candidates:
            n = slicer.mrmlScene.GetFirstNodeByName(name)
            if n is not None:
                return n
        return None

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

    def _refresh_all(self) -> None:
        self._refresh_patient_list()
        self._refresh_fraction_list()

    def _on_patient_changed(self, index: int) -> None:
        # Keep listing synced with patient selection
        self._refresh_fraction_list()

    def _double_spin_value(self, spin: Any) -> float:
        if spin is None:
            return 1.0
        v = spin.value() if callable(getattr(spin, "value", None)) else getattr(spin, "value", 1.0)
        return float(v)

    def _needs_resample_to_reference(self, input_node: Any, reference_node: Any) -> bool:
        """Return True if input_node geometry differs from reference_node.
        We check array shape AND IJK->RAS matrix to avoid silently summing mismatched grids.
        """
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

    def _is_fraction_proxy_name(self, name: str) -> bool:
        """Return True if a MRML node name corresponds to a Phase-1 dose_list output."""
        n = str(name or "")
        return n.startswith("dose_list_")

    def _base_name_from_dose_list(self, dose_list_name: str) -> str:
        """Extract base name from a dose_list_* node name."""
        s = str(dose_list_name or "")
        return s[len("dose_list_") :] if s.startswith("dose_list_") else ""

    def _has_uncertainty_for_any_proxy(self, proxies: list[Any]) -> bool:
        for node in proxies or []:
            name = self._safe_node_name(node)
            base = self._base_name_from_dose_list(name)
            if not base:
                continue
            if self._find_uncertainty_node(base) is not None:
                return True
        return False

    def _clip(self, x: float, lo: float, hi: float) -> float:
        return float(min(max(float(x), float(lo)), float(hi)))

    def _get_sh_node(self) -> Any | None:
        if slicer.mrmlScene is None:
            return None

        return slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)

    def _get_subject_item_id_from_sh(self, sh_node: Any, item_id: int) -> int:
        """Climb SH parents until we reach the Patient (subject) item.

        Fallback: returns the immediate parent if Patient level cannot be detected.
        """
        if sh_node is None or not item_id:
            return 0
        current = int(item_id)
        last_parent = 0
        for _ in range(20):
            level = sh_node.GetItemLevel(current)
            if str(level).lower() == "patient":
                return current
            parent = 0
            parent = sh_node.GetItemParent(current)
            if not parent:
                break
            last_parent = int(parent)
            current = int(parent)
        return int(last_parent)

    def _ensure_node_in_sh_folder(self, node: Any, folder_item_id: int) -> None:
        """Ensure node has an SH item and is parented under folder_item_id."""
        if slicer.mrmlScene is None or node is None or not folder_item_id:
            return
        sh_node = self._get_sh_node()
        if sh_node is None:
            return
        item_id = sh_node.GetItemByDataNode(node)
        if item_id == 0:
            item_id = sh_node.CreateItem(folder_item_id, node)
        sh_node.SetItemParent(item_id, folder_item_id)

    def _get_or_create_output_folder_item(self, reference_node: Any, folder_name: str) -> int | None:
        """Return a SubjectHierarchy folder item under the same subject (patient) as reference_node."""
        if slicer.mrmlScene is None or reference_node is None:
            return None
        sh_node = self._get_sh_node()
        if sh_node is None:
            return None
        ref_item_id = sh_node.GetItemByDataNode(reference_node)
        if ref_item_id == 0:
            return None
        parent_item_id = self._get_subject_item_id_from_sh(sh_node, int(ref_item_id)) or int(ref_item_id)

        children = vtk.vtkIdList()
        sh_node.GetItemChildren(parent_item_id, children, False)
        for i in range(children.GetNumberOfIds()):
            child_id = int(children.GetId(i))
            if sh_node.GetItemName(child_id) == folder_name and sh_node.GetItemDataNode(child_id) is None:
                return child_id
        return sh_node.CreateFolderItem(parent_item_id, folder_name)

    def _create_temp_resampled_volume(self, input_node: Any) -> Any:
        out_name = f"{self._safe_node_name(input_node)}_resampled_{uuid4().hex[:6]}"
        out_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", out_name)
        out_node.SetHideFromEditors(1)
        out_node.SetSelectable(0)
        out_node.SetSaveWithScene(0)

        return out_node

    def _selected_patient_item_id(self) -> int:
        idx = self._combo_current_index(getattr(self, "patient_combo", None))
        return int(self._patient_item_id_map.get(idx, 0) or 0)

    def _refresh_patient_list(self) -> None:
        if not hasattr(self, "patient_combo"):
            return

        prev_item_id = 0
        prev_item_id = int(self._selected_patient_item_id() or 0)

        self._patient_item_id_map = {}
        self.patient_combo.blockSignals(True)
        self.patient_combo.clear()
        self.patient_combo.addItem("[Select patient]")
        self._patient_item_id_map[0] = 0

        if slicer.mrmlScene is None:
            self.patient_combo.blockSignals(False)
            return

        sh_node = self._get_sh_node()
        if sh_node is None:
            self.patient_combo.blockSignals(False)
            return

        root_id = int(sh_node.GetSceneItemID() or 0)
        if not root_id:
            root_id = int(sh_node.GetRootItemID() or 0)

        ids = vtk.vtkIdList()
        try:
            sh_node.GetItemChildren(root_id, ids, True)
        except Exception:
            ids = vtk.vtkIdList()

        patient_ids = []
        for i in range(ids.GetNumberOfIds()):
            item_id = int(ids.GetId(i))
            level = ""
            level = sh_node.GetItemLevel(item_id)
            if str(level).lower() == "patient":
                patient_ids.append(item_id)

        patient_ids.sort(key=lambda pid: (sh_node.GetItemName(pid) or "").lower())

        match_index = 0
        idx = 1
        for pid in patient_ids:
            name = sh_node.GetItemName(pid) or ""
            self.patient_combo.addItem(name or f"Patient {pid}")
            self._patient_item_id_map[idx] = pid
            if prev_item_id and int(pid) == int(prev_item_id):
                match_index = idx
            idx += 1

        if match_index:
            self.patient_combo.setCurrentIndex(match_index)

        self.patient_combo.blockSignals(False)

    def _node_patient_item_id(self, node: Any) -> int:
        if node is None or slicer.mrmlScene is None:
            return 0
        sh_node = self._get_sh_node()
        if sh_node is None:
            return 0
        item_id = int(sh_node.GetItemByDataNode(node) or 0)

        if not item_id:
            return 0
        return int(self._get_subject_item_id_from_sh(sh_node, item_id) or 0)

    def _clear_fraction_list_ui(self) -> None:
        # Reset cached UI state
        self._proxy_nodes_by_id = {}
        self._fraction_checkboxes = []
        self._weight_spinboxes_by_id = {}

        layout = self.fractions_container_layout
        if layout is None:
            return

        while layout.count() > 0:
            item = layout.takeAt(0)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _refresh_fraction_list(self) -> None:
        self._clear_fraction_list_ui()
        if slicer.mrmlScene is None:
            return

        selected_patient_id = self._selected_patient_item_id()
        if not selected_patient_id:
            info = QLabel("Select a patient above to list available dose outputs.")
            info.setWordWrap(True)
            self.fractions_container_layout.insertWidget(0, info)
            self.fractions_container_layout.addStretch()
            return

        volume_nodes = slicer.util.getNodes("vtkMRMLScalarVolumeNode*")
        proxies = []
        for _, node in volume_nodes.items():
            if node is None:
                continue
            node_name = self._safe_node_name(node)
            if not self._is_fraction_proxy_name(node_name):
                continue
            if self._node_patient_item_id(node) != int(selected_patient_id):
                continue
            proxies.append(node)

        proxies.sort(key=lambda n: self._safe_node_name(n).lower())

        has_uncertainty = self._has_uncertainty_for_any_proxy(proxies)
        if self.strategy_combo is not None:
            model = self.strategy_combo.model()
            count = self.strategy_combo.count
            for i in range(count):
                item = model.item(i) if model is not None else None
                if item is not None:
                    item.setEnabled(bool(has_uncertainty or i == 0))
            if not has_uncertainty:
                self.strategy_combo.setCurrentIndex(0)

        for node in proxies:
            node_id = node.GetID() if hasattr(node, "GetID") else None
            if not node_id:
                continue
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)

            node_name = self._safe_node_name(node)
            cb = QCheckBox(node_name)
            cb.setChecked(False)
            cb.setProperty("node_id", node_id)

            w_label = QLabel("w:")
            w_spin = QDoubleSpinBox()
            w_spin.setDecimals(1)
            w_spin.setSingleStep(0.1)
            w_spin.setRange(0.0, 10.0)
            w_spin.setValue(1.0)
            w_spin.setToolTip("Weight applied to this dose in the accumulated sum")

            row_layout.addWidget(cb, 1)
            row_layout.addWidget(w_label)
            row_layout.addWidget(w_spin)

            self._fraction_checkboxes.append(cb)
            self._proxy_nodes_by_id[node_id] = node
            self._weight_spinboxes_by_id[node_id] = w_spin
            self.fractions_container_layout.insertWidget(len(self._fraction_checkboxes) - 1, row)

        if not proxies:
            info = QLabel("No dose outputs found for this patient. Run Phase 1 first.")
            info.setWordWrap(True)
            self.fractions_container_layout.insertWidget(0, info)

        self.fractions_container_layout.addStretch()

    def _create_or_update_volume(
        self, name: str, reference_node: Any, array: np.ndarray, existing_node: Any = None
    ) -> Any:
        if reference_node is None:
            raise ValueError("Reference node is required")
        target_node = existing_node if existing_node is not None else slicer.mrmlScene.GetFirstNodeByName(name)
        volumes_logic = slicer.modules.volumes.logic()
        if volumes_logic is None:
            raise RuntimeError("Volumes module logic is not available")
        if target_node is None:
            target_node = volumes_logic.CloneVolume(reference_node, name)
        reference_array = slicer.util.arrayFromVolume(reference_node)
        slicer.util.updateVolumeFromArray(target_node, array.astype(reference_array.dtype, copy=False))
        slicer.util.arrayFromVolumeModified(target_node)
        return target_node

    def _acc_node_id(self, node: Any) -> str | None:
        return node.GetID() if node is not None and hasattr(node, "GetID") else None

    def _acc_collect_selected_items(self) -> list[tuple[Any, float]]:
        selected_items: list[tuple[Any, float]] = []
        for cb in self._fraction_checkboxes:
            if not cb.isChecked():
                continue
            node_id = cb.property("node_id")
            node = self._proxy_nodes_by_id.get(node_id)
            if node is None:
                continue
            w_spin = self._weight_spinboxes_by_id.get(node_id)
            weight = self._double_spin_value(w_spin)
            selected_items.append((node, float(weight)))
        return selected_items

    def _acc_compute_strategy(self, selected_items: list[tuple[Any, float]], strategy_idx: int) -> dict[str, Any]:
        uncertainty_aware = int(strategy_idx) > 0

        # Strategy 2 (index 2): robust uncertainty (remove worst outlier contribution).
        robust_uncertainty = int(strategy_idx) == 2
        robust_k = 0
        top_vars: list[Any] | None = None
        if robust_uncertainty:
            # Remove k largest per-voxel variance contributions, where:
            # k = ceil(20% of the number of eligible sessions), capped to N-1.
            eligible_n = 0
            for node, w in selected_items:
                if float(w) <= 0.0:
                    continue
                base = self._base_name_from_dose_list(self._safe_node_name(node))
                if not base or slicer.mrmlScene is None:
                    continue
                unc_node = self._find_uncertainty_node(base)
                if unc_node is not None:
                    eligible_n += 1

            if eligible_n >= 2:
                robust_k = int(np.ceil(0.2 * float(eligible_n)))
                robust_k = min(robust_k, eligible_n - 1)

            if robust_k > 0:
                top_vars = [None] * int(robust_k)

        # Strategy 3 (index 3): ramp uncertainty weighting from 0.5 to 2.0 across selected sessions.
        # This affects ONLY the uncertainty propagation, not the accumulated dose.
        uncertainty_ramp = int(strategy_idx) == 3
        alpha_by_node_id: dict[str, float] = {}
        if uncertainty_ramp:
            eligible: list[str] = []
            for node, w in selected_items:
                if float(w) <= 0.0:
                    continue
                node_id = self._acc_node_id(node)
                if node_id:
                    eligible.append(node_id)

            if eligible:
                alphas = np.linspace(0.5, 2.0, num=len(eligible), dtype=np.float32)
                alpha_by_node_id = {nid: float(a) for nid, a in zip(eligible, alphas)}

        # Strategy 4 (index 4): anatomy-driven uncertainty weighting based on dvf_magnitude_{base}.
        dvf_mag_weighting = int(strategy_idx) == 4
        # NOTE: DVF-magnitude weighting is computed later in the async pipeline
        # (see _acc_compute_alpha_by_dvf_magnitude_async). Keep placeholder here.
        if dvf_mag_weighting:
            alpha_by_node_id = {}

        return {
            "uncertainty_aware": bool(uncertainty_aware),
            "robust_uncertainty": bool(robust_uncertainty),
            "robust_k": int(robust_k),
            "top_vars": top_vars,
            "uncertainty_ramp": bool(uncertainty_ramp),
            "dvf_mag_weighting": bool(dvf_mag_weighting),
            "alpha_by_node_id": dict(alpha_by_node_id),
        }

    def _acc_prepare_job(
        self,
        selected_items: list[tuple[Any, float]],
        output_base_name: str,
        strategy_idx: int,
        strategy: dict[str, Any],
        ref_node: Any,
        folder_item_id: int | None,
    ) -> None:
        self._active_job = {
            "selected_items": list(selected_items),
            "output_base_name": output_base_name,
            "strategy_idx": int(strategy_idx),
            "uncertainty_aware": bool(strategy.get("uncertainty_aware", False)),
            "robust_uncertainty": bool(strategy.get("robust_uncertainty", False)),
            "robust_k": int(strategy.get("robust_k", 0)),
            "top_vars": strategy.get("top_vars"),
            "uncertainty_ramp": bool(strategy.get("uncertainty_ramp", False)),
            "dvf_mag_weighting": bool(strategy.get("dvf_mag_weighting", False)),
            "ref_node": ref_node,
            "folder_item_id": folder_item_id,
            "temp_nodes": [],
            "eval": {},  # (kind, key) -> node
            "alpha_by_node_id": dict(strategy.get("alpha_by_node_id") or {}),
            "sum_mean": None,
            "sum_var": None,
            "n_var_contrib": 0,
            "sum_w": 0.0,
            "used": 0,
        }

    def _acc_cleanup_temp_nodes(self) -> None:
        job = self._active_job
        if job is None:
            return
        for tn in job.get("temp_nodes", []):
            if tn is not None and slicer.mrmlScene is not None and tn.GetScene() == slicer.mrmlScene:
                slicer.mrmlScene.RemoveNode(tn)

    def _acc_fail(self, msg: str) -> None:
        self._acc_cleanup_temp_nodes()
        self._finish_job(False, msg)

    def _acc_prepare_resample_tasks(self) -> None:
        job = self._active_job
        if job is None:
            return
        eval_map = job["eval"]
        tasks: list[dict[str, Any]] = []
        ref = job["ref_node"]

        # Dose volumes
        for dose_list_node, _ in job["selected_items"]:
            if dose_list_node is None:
                continue
            key = self._acc_node_id(dose_list_node)
            if not key:
                continue
            eval_map[("dose", key)] = dose_list_node
            if self._needs_resample_to_reference(dose_list_node, ref):
                tasks.append({"kind": "dose", "key": key, "src": dose_list_node, "interp": "linear"})

        # Uncertainty volumes
        if job.get("uncertainty_aware"):
            for dose_list_node, _ in job["selected_items"]:
                name = self._safe_node_name(dose_list_node)
                base = self._base_name_from_dose_list(name)
                if not base:
                    continue
                unc_node = self._find_uncertainty_node(base)
                if unc_node is None:
                    continue
                eval_map[("unc", base)] = unc_node
                if self._needs_resample_to_reference(unc_node, ref):
                    tasks.append({"kind": "unc", "key": base, "src": unc_node, "interp": "linear"})

        # DVF magnitude volumes (for anatomy-driven weighting)
        if job.get("dvf_mag_weighting"):
            for dose_list_node, _ in job["selected_items"]:
                name = self._safe_node_name(dose_list_node)
                base = self._base_name_from_dose_list(name)
                if not base:
                    continue
                dvf_node = slicer.mrmlScene.GetFirstNodeByName(f"dvf_magnitude_{base}")
                eval_map[("dvfmag", base)] = dvf_node
                if self._needs_resample_to_reference(dvf_node, ref):
                    tasks.append({"kind": "dvfmag", "key": base, "src": dvf_node, "interp": "linear"})

        job["_resample_tasks"] = tasks
        job["_resample_index"] = 0

        self._set_status("Resampling inputs (if needed)…")
        self._set_progress(5, visible=True)

    def _acc_resample_done(self) -> None:
        j2 = self._active_job
        if j2 is None:
            return
        info = j2.get("_resample_current", {})
        out_node = info.get("out_node")
        kind = info.get("kind")
        key = info.get("key")
        total = int(info.get("total", len(j2.get("_resample_tasks", []))))
        if out_node is None or kind is None or key is None:
            self._acc_fail("Resample failed: invalid output")
            return

        j2["eval"][(kind, key)] = out_node
        j2["_resample_index"] = int(j2.get("_resample_index", 0)) + 1
        p = 5 + int(35 * float(j2["_resample_index"]) / float(max(1, total)))
        self._set_progress(min(40, p), visible=True)
        try:
            QTimer.singleShot(0, self._acc_run_next_resample)
        except Exception:
            self._acc_run_next_resample()

    def _acc_resample_err(self, exc) -> None:
        logger.exception("Resample failed")
        self._acc_fail(f"Resample failed: {exc}")

    def _acc_run_next_resample(self) -> None:
        j = self._active_job
        if j is None:
            return
        tlist = j.get("_resample_tasks", [])
        i = int(j.get("_resample_index", 0))
        n = int(len(tlist))
        if i >= n:
            self._acc_start_dvfmag_or_accum()
            return

        task = tlist[i]
        src = task.get("src")
        kind = task.get("kind")
        key = task.get("key")
        interp = task.get("interp", "linear")
        if src is None or kind is None or key is None:
            j["_resample_index"] = i + 1
            self._acc_run_next_resample()
            return

        out_node = self._create_temp_resampled_volume(src)
        j["temp_nodes"].append(out_node)
        j["_resample_current"] = {"out_node": out_node, "kind": kind, "key": key, "total": n}
        params = {
            "inputVolume": src.GetID(),
            "referenceVolume": j["ref_node"].GetID(),
            "outputVolume": out_node.GetID(),
            "interpolationType": str(interp),
        }

        self._run_cli_async(
            slicer.modules.resamplescalarvectordwivolume, params, self._acc_resample_done, self._acc_resample_err
        )

    def _acc_compute_alpha_by_dvf_magnitude_async(self) -> None:
        j = self._active_job
        if j is None:
            return
        j["_mags"] = []
        j["_dvf_idx"] = 0
        self._set_status("Computing DVF-magnitude weights…")
        self._set_progress(40, visible=True)
        try:
            QTimer.singleShot(0, self._acc_dvfmag_tick)
        except Exception:
            self._acc_dvfmag_tick()

    def _acc_dvfmag_tick(self) -> None:
        j2 = self._active_job
        if j2 is None:
            return
        items = j2.get("selected_items", [])
        idx = int(j2.get("_dvf_idx", 0))
        if idx >= len(items):
            mags = j2.get("_mags", [])
            if mags:
                m_values = np.array([m for _, m in mags], dtype=np.float32)
                m_values = m_values[np.isfinite(m_values)]
            else:
                m_values = np.array([], dtype=np.float32)

            alpha_by_node_id: dict[str, float] = {}
            if m_values.size > 0:
                m_ref = float(np.median(m_values))
                if np.isfinite(m_ref) and m_ref > 0.0:
                    for nid, m_i in mags:
                        alpha_by_node_id[nid] = self._clip(float(m_i) / float(m_ref), 0.5, 2.0)
            j2["alpha_by_node_id"] = alpha_by_node_id
            self._acc_start_accum_loop()
            return

        dose_list_node, w = items[idx]
        j2["_dvf_idx"] = idx + 1
        if float(w) <= 0.0:
            try:
                QTimer.singleShot(0, self._acc_dvfmag_tick)
            except Exception:
                self._acc_dvfmag_tick()
            return
        node_id = self._acc_node_id(dose_list_node)
        if not node_id:
            try:
                QTimer.singleShot(0, self._acc_dvfmag_tick)
            except Exception:
                self._acc_dvfmag_tick()
            return
        base = self._base_name_from_dose_list(self._safe_node_name(dose_list_node))
        dvf_node = j2["eval"].get(("dvfmag", base), None)
        if dvf_node is None:
            try:
                QTimer.singleShot(0, self._acc_dvfmag_tick)
            except Exception:
                self._acc_dvfmag_tick()
            return
        arr = slicer.util.arrayFromVolume(dvf_node).astype(np.float32, copy=False)
        if arr.size == 0:
            try:
                QTimer.singleShot(0, self._acc_dvfmag_tick)
            except Exception:
                self._acc_dvfmag_tick()
            return
        flat = arr.reshape(-1)
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            try:
                QTimer.singleShot(0, self._acc_dvfmag_tick)
            except Exception:
                self._acc_dvfmag_tick()
            return
        m_i = float(np.percentile(flat, 95))
        j2["_mags"].append((node_id, m_i))

        p = 40 + int(15 * float(idx + 1) / float(max(1, len(items))))
        self._set_progress(min(55, p), visible=True)
        try:
            QTimer.singleShot(0, self._acc_dvfmag_tick)
        except Exception:
            self._acc_dvfmag_tick()

    def _acc_start_dvfmag_or_accum(self) -> None:
        j = self._active_job
        if j is None:
            return
        if j.get("dvf_mag_weighting"):
            self._acc_compute_alpha_by_dvf_magnitude_async()
        else:
            self._acc_start_accum_loop()

    def _acc_start_accum_loop(self) -> None:
        j = self._active_job
        if j is None:
            return

        self._set_status("Accumulating…")
        self._set_progress(55, visible=True)

        j["_acc_idx"] = 0
        try:
            QTimer.singleShot(0, self._acc_accum_tick)
        except Exception:
            self._acc_accum_tick()

    def _acc_accum_tick(self) -> None:
        j2 = self._active_job
        if j2 is None:
            return
        items = j2.get("selected_items", [])
        idx = int(j2.get("_acc_idx", 0))
        if idx >= len(items):
            self._acc_finalize()
            return

        dose_list_node, weight = items[idx]
        j2["_acc_idx"] = idx + 1
        w = float(weight)
        if w <= 0.0:
            try:
                QTimer.singleShot(0, self._acc_accum_tick)
            except Exception:
                self._acc_accum_tick()
            return

        dose_id = self._acc_node_id(dose_list_node)
        eval_dose = j2["eval"].get(("dose", dose_id), dose_list_node)

        arr = slicer.util.arrayFromVolume(eval_dose).astype(np.float32, copy=False)

        if j2["sum_mean"] is None:
            j2["sum_mean"] = np.array(arr, dtype=np.float32, copy=True) * np.float32(w)
        else:
            j2["sum_mean"] = j2["sum_mean"] + (arr * np.float32(w))

        if j2.get("uncertainty_aware"):
            dose_name = self._safe_node_name(dose_list_node)
            base = self._base_name_from_dose_list(dose_name)
            unc_node = j2["eval"].get(("unc", base), None)
            if unc_node is not None:
                std = slicer.util.arrayFromVolume(unc_node).astype(np.float32, copy=False)
                # Use precomputed alphas (ramp or DVF magnitude) for consistency.
                alpha = float(j2.get("alpha_by_node_id", {}).get(dose_id, 1.0)) if dose_id else 1.0

                scale = np.float32(w * alpha)
                var = np.square(std, dtype=np.float32) * (scale * scale)
                if j2["sum_var"] is None:
                    j2["sum_var"] = np.array(var, dtype=np.float32, copy=True)
                else:
                    j2["sum_var"] = j2["sum_var"] + var

                if j2.get("robust_uncertainty") and (j2.get("top_vars") is not None):
                    top_vars2 = j2["top_vars"]
                    candidate = var
                    for jidx in range(len(top_vars2)):
                        tv = top_vars2[jidx]
                        if tv is None:
                            top_vars2[jidx] = np.array(candidate, dtype=np.float32, copy=True)
                            candidate = None
                            break
                        m = candidate > tv
                        if np.any(m):
                            new_tv = np.where(m, candidate, tv)
                            candidate = np.where(m, tv, candidate)
                            top_vars2[jidx] = new_tv
                    j2["top_vars"] = top_vars2

                j2["n_var_contrib"] = int(j2.get("n_var_contrib", 0)) + 1

        j2["sum_w"] = float(j2.get("sum_w", 0.0)) + w
        j2["used"] = int(j2.get("used", 0)) + 1

        p = 55 + int(40 * float(idx + 1) / float(max(1, len(items))))
        self._set_progress(min(95, p), visible=True)
        self._set_status(f"Accumulating… ({j2['_acc_idx']}/{len(items)})")

        try:
            QTimer.singleShot(0, self._acc_accum_tick)
        except Exception:
            self._acc_accum_tick()

    def _acc_finalize(self) -> None:
        j = self._active_job
        if j is None:
            return

        used2 = int(j.get("used", 0))
        sum_mean2 = j.get("sum_mean", None)
        sum_w2 = float(j.get("sum_w", 0.0))
        if used2 == 0 or sum_mean2 is None:
            self._acc_fail("No valid dose_list volumes found for the selected inputs.")
            return
        if sum_w2 <= 0.0:
            self._acc_fail("Sum of weights is 0. Please set positive weights (w > 0).")
            return

        self._set_status("Writing outputs…")
        self._set_progress(95, visible=True)

        # Weighted average (stay in float32)
        sum_mean2 = sum_mean2.astype(np.float32, copy=False) / np.float32(sum_w2)
        sum_var2 = j.get("sum_var", None)
        if j.get("uncertainty_aware") and sum_var2 is not None:
            if j.get("robust_uncertainty") and int(j.get("n_var_contrib", 0)) >= 2:
                top_vars = list(j.get("top_vars") or [])
                for tv in top_vars:
                    if tv is not None:
                        sum_var2 = sum_var2 - tv
                sum_var2 = np.maximum(sum_var2, 0.0)
            sum_var2 = sum_var2.astype(np.float32, copy=False) / np.float32(sum_w2 * sum_w2)

        acc_name = f"{j['output_base_name']}"
        acc_volume = self._create_or_update_volume(acc_name, j["ref_node"], sum_mean2, existing_node=None)
        self._ensure_node_in_sh_folder(acc_volume, j["folder_item_id"])

        unc_volume = None
        if j.get("uncertainty_aware"):
            if sum_var2 is None:
                logger.info(
                    "Classic Uncertainty Aware selected but no matching uncertainty_* volumes"
                    " were found; skipping uncertainty output."
                )
            else:
                unc_name = f"uncertainty_{j['output_base_name']}"
                acc_std = np.sqrt(sum_var2)
                unc_volume = self._create_or_update_volume(unc_name, j["ref_node"], acc_std, existing_node=None)
                self._ensure_node_in_sh_folder(unc_volume, j["folder_item_id"])

        self._acc_cleanup_temp_nodes()

        QMessageBox.information(
            self,
            "Accumulation Complete",
            (
                f"Accumulated {used2} fraction(s).\n"
                f"Output: {self._safe_node_name(acc_volume)}"
                + (f"\nUncertainty: {self._safe_node_name(unc_volume)}" if unc_volume else "")
            ),
        )
        self._finish_job(True, "Done.")

    def _on_compute_accumulation(self) -> None:
        if self._active_job is not None:
            QMessageBox.information(self, "Busy", "An accumulation computation is already running.")
            return

        if slicer.mrmlScene is None:
            QMessageBox.warning(self, "No Scene", "MRML scene is not available.")
            return

        selected_items = self._acc_collect_selected_items()
        if not selected_items:
            QMessageBox.warning(self, "Missing Inputs", "Select at least one dose_list output.")
            return

        output_base_name = self._line_edit_text(self.output_name_edit).strip() or self._generate_default_output_name(
            prefix="dose_acc"
        )
        strategy_idx = self._combo_current_index(getattr(self, "strategy_combo", None))
        strategy = self._acc_compute_strategy(selected_items, int(strategy_idx))

        ref_node = selected_items[0][0]
        folder_item_id = self._get_or_create_output_folder_item(ref_node, output_base_name)

        self._acc_prepare_job(selected_items, output_base_name, int(strategy_idx), strategy, ref_node, folder_item_id)

        self._set_ui_busy(True)
        self._set_status("Preparing…")
        self._set_progress(0, visible=True)

        self._acc_prepare_resample_tasks()
        try:
            QTimer.singleShot(0, self._acc_run_next_resample)
        except Exception:
            self._acc_run_next_resample()
        return
