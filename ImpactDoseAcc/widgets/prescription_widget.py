import logging
import vtk
import numpy as np
from uuid import uuid4
import slicer
from qt import QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QCheckBox, QFileDialog, QWidget, QMessageBox, QTimer

import importlib.util
import os
from pathlib import Path

logger = logging.getLogger(__name__)

base_path = Path(__file__).resolve().parent / "base_widget.py"
spec = importlib.util.spec_from_file_location("impactdoseacc_base_widget", str(base_path))
base_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_mod)  # type: ignore
BaseImpactWidget = getattr(base_mod, "BaseImpactWidget")

class DvfSelectorRow(QWidget):
    """UI row for selecting a single DVF sample.

    Supports native TransformNodes and transform sequences.
    """
    def __init__(self, on_add=None, on_remove=None):
        super().__init__()
        self._dvf_node = None
        self._dvf_node_map = {}
        self._on_add = on_add
        self._on_remove = on_remove
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout()
        dvf_label = QLabel("DVF:")
        self.dvf_combo = QComboBox()
        self.dvf_combo.currentIndexChanged.connect(self._on_dvf_changed)
        layout.addWidget(dvf_label)
        layout.addWidget(self.dvf_combo, 1)

        self._add_btn = QPushButton("+")
        self._add_btn.setMaximumWidth(28)
        self._add_btn.setToolTip("Add DVF")
        self._add_btn.clicked.connect(self._on_add_clicked)
        layout.addWidget(self._add_btn)

        self._remove_btn = QPushButton("-")
        self._remove_btn.setMaximumWidth(28)
        self._remove_btn.setToolTip("Remove DVF")
        self._remove_btn.clicked.connect(self._on_remove_clicked)
        layout.addWidget(self._remove_btn)

        self.setLayout(layout)
        self._populate_dvf_options()

    def _on_add_clicked(self) -> None:
        try:
            if callable(self._on_add):
                self._on_add()
        except Exception:
            pass

    def _on_remove_clicked(self) -> None:
        if not callable(self._on_remove):
            return
        try:
            def _do_remove():
                self._on_remove(self)
            QTimer.singleShot(0, _do_remove)
        except Exception:
            self._on_remove(self)


    def _populate_dvf_options(self) -> None:
        self.dvf_combo.clear()
        self._dvf_node_map.clear()
        self.dvf_combo.addItem("[Select DVF]")
        self._dvf_node_map[0] = (None, None)
        combo_index = 1
        if slicer.mrmlScene is None:
            return

        def _safe_name(node) -> str:
            try:
                return node.GetName()
            except Exception:
                return ""

        def _is_internal_slice_transform_name(name: str) -> bool:
            n = str(name).strip().lower()
            return n in ("red transform", "yellow transform", "green transform")

        # Transform nodes (DVFs)
        transform_nodes = []
        if hasattr(slicer.util, "getNodesByClass"):
            tn = slicer.util.getNodesByClass("vtkMRMLTransformNode")
            transform_nodes = list(tn.values()) if isinstance(tn, dict) else list(tn)
        else:
            tn = slicer.mrmlScene.GetNodesByClass("vtkMRMLTransformNode")
            transform_nodes = [tn.GetItemAsObject(i) for i in range(tn.GetNumberOfItems())]

        for node in sorted([n for n in transform_nodes if n is not None], key=lambda n: _safe_name(n).lower()):
            node_name = _safe_name(node)
            if not node_name or _is_internal_slice_transform_name(node_name):
                continue
            self.dvf_combo.addItem(f"Transform: {node_name}")
            self._dvf_node_map[combo_index] = (node, "transform")
            combo_index += 1

        # Transform sequences
        sequence_nodes = slicer.util.getNodes("vtkMRMLSequenceNode*")
        for node_name, node in sequence_nodes.items():
            try:
                data_class = node.GetDataNodeClassName()
            except Exception:
                continue
            if not data_class or "TransformNode" not in str(data_class):
                continue
            self.dvf_combo.addItem(f"Transform Seq: {node_name}")
            self._dvf_node_map[combo_index] = (node, "transform-sequence")
            combo_index += 1

    def _on_dvf_changed(self, index: int) -> None:
        if index >= 0:
            dvf_data = self._dvf_node_map.get(index)
            if dvf_data:
                self._dvf_node, _ = dvf_data
            else:
                self._dvf_node = None
        else:
            self._dvf_node = None

    def get_dvf_node(self):
        return self._dvf_node

    def get_dvf_type(self):
        idx_attr = getattr(self.dvf_combo, "currentIndex", 0)
        try:
            current_index = idx_attr() if callable(idx_attr) else int(idx_attr)
        except Exception:
            current_index = 0
        dvf_data = self._dvf_node_map.get(current_index)
        if dvf_data:
            _, dvf_type = dvf_data
            return dvf_type
        return None


class DoseSelectorRow(QWidget):
    """UI row for selecting a single RTDOSE volume.

    Multiple rows can be added to select multiple doses.
    """
    def __init__(self, is_rtdose_cb, on_add=None, on_remove=None):
        super().__init__()
        self._is_rtdose = is_rtdose_cb
        self._dose_node = None
        self._dose_node_map = {}
        self._on_add = on_add
        self._on_remove = on_remove
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout()
        dose_label = QLabel("Dose:")
        self.dose_combo = QComboBox()
        self.dose_combo.currentIndexChanged.connect(self._on_dose_changed)
        layout.addWidget(dose_label)
        layout.addWidget(self.dose_combo, 1)

        self._add_btn = QPushButton("+")
        self._add_btn.setMaximumWidth(28)
        self._add_btn.setToolTip("Add dose")
        self._add_btn.clicked.connect(self._on_add_clicked)
        layout.addWidget(self._add_btn)

        self._remove_btn = QPushButton("-")
        self._remove_btn.setMaximumWidth(28)
        self._remove_btn.setToolTip("Remove dose")
        self._remove_btn.clicked.connect(self._on_remove_clicked)
        layout.addWidget(self._remove_btn)

        self.setLayout(layout)
        self.refresh_options()

    def _on_add_clicked(self) -> None:
        try:
            if callable(self._on_add):
                self._on_add()
        except Exception:
            pass

    def _on_remove_clicked(self) -> None:
        # Avoid deleting Qt objects in the middle of the clicked() callback.
        try:
            if not callable(self._on_remove):
                return
            try:
                def _do_remove():
                    self._on_remove(self)

                QTimer.singleShot(0, _do_remove)
            except Exception:
                self._on_remove(self)
        except Exception:
            pass

    def refresh_options(self) -> None:
        self.dose_combo.clear()
        self._dose_node_map.clear()
        self.dose_combo.addItem("[Select dose]")
        self._dose_node_map[0] = None
        self._dose_node = None
        if slicer.mrmlScene is None:
            return

        volume_nodes = slicer.util.getNodes("vtkMRMLScalarVolumeNode*")
        combo_index = 1
        for node_name, node in volume_nodes.items():
            name_l = str(node_name or "").lower()

            if "uncertainty" in name_l or "dvf_magnitude" in name_l:
                continue
            try:
                if not self._is_rtdose(node):
                    continue
            except Exception:
                continue
            self.dose_combo.addItem(node_name)
            self._dose_node_map[combo_index] = node
            combo_index += 1

    def _on_dose_changed(self, index: int) -> None:
        self._dose_node = self._dose_node_map.get(index)

    def get_dose_node(self):
        return self._dose_node


class PrescriptionDoseEstimationWidget(BaseImpactWidget):
    """UI widget for Phase 1: Delivered Dose Estimation."""

    def __init__(
        self,
        logic,
        export_callback=None,
        browse_export_dir_callback=None,
        import_callback=None,
        deform_callback=None,
        refresh_sessions_callback=None,
    ):
        super().__init__(logic)
        # create a logger for this widget
        self._logger = logging.getLogger(self.__class__.__name__)
        self._session_widgets = []
        self._dose_widgets = []
        self.export_dir = ""
        self._sct_node_map = {}
        self._ref_ct_node_map = {}
        self.ref_ct_combo = None
        self.output_name_edit = None
        self.sct_checkboxes = []
        self._active_job = None
        self.status_label = None
        self.progress_bar = None
        self.export_btn = None
        self.browse_export_btn = None
        # Store callbacks as instance attributes so methods can access them
        self._export_callback = export_callback
        self._browse_export_dir_callback = browse_export_dir_callback
        self._import_callback = import_callback
        self._deform_callback = deform_callback
        self._refresh_sessions_callback = refresh_sessions_callback
        self._setup_ui()

    def _remove_row(self, row_list: list, row_widget, layout, reset_combo=None):
        if row_widget not in row_list:
            return
        if len(row_list) <= 1:
            if callable(reset_combo):
                try:
                    reset_combo(row_widget)
                except Exception:
                    pass
            return
        row_list.remove(row_widget)
        if layout is not None:
            layout.removeWidget(row_widget)
        row_widget.hide()
        row_widget.setEnabled(False)
        QTimer.singleShot(0, row_widget.deleteLater)

    def _setup_ui(self) -> None:
        ui_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "../Resources/UI/PrescriptionWidget.ui"))
        ui_widget = slicer.util.loadUI(ui_path)
        self.ui = slicer.util.childWidgetVariables(ui_widget)
        self._root_widget = ui_widget

        # Bind main containers/widgets from the .ui file
        self.ref_ct_combo = self._w("ref_ct_combo")
        self.sct_list_widget = self._w("sct_list_widget")
        self.sct_list_container = self._w("sct_list_container")
        self.sct_list_layout = self._layout("sct_list_container")
        self.sessions_scroll = self._w("sessions_scroll")
        self.sessions_container = self._w("sessions_container")
        self.sessions_container_layout = self._layout("sessions_container")
        self.dose_rows_container = self._w("dose_rows_container")
        self.dose_rows_layout = self._layout("dose_rows_container")
        self.dvf_rows_container = self._w("dvf_rows_container")
        self.dvf_rows_layout = self._layout("dvf_rows_container")
        output_container = self._w("output_options_container")
        self.output_options_layout = output_container.layout() if output_container else None
        self.output_name_edit = self._w("output_name_edit")
        self.deform_button = self._w("deform_button")
        self.status_label = self._w("status_label")
        self.progress_bar = self._w("progress_bar")
        self.export_dir_display = self._w("export_dir_display")
        if self.export_dir_display:
            self.export_dir_display.setStyleSheet("color: gray;")

        if self.output_name_edit is not None:
            self.output_name_edit.setText(self._generate_default_output_name(prefix="session"))

        # Buttons
        self._btn("refresh_tps_btn", self._refresh_tps_lists)
        refresh_sessions_cb = self._refresh_sessions_callback or self._refresh_session_combos
        self._btn("refresh_sessions_btn", refresh_sessions_cb)
        self.browse_export_btn = self._btn("browse_export_btn", self._browse_export_dir_callback or self._on_browse_export_dir)
        self.export_btn = self._btn("export_btn", self._export_callback or self._on_export_sct)

        # Populate combos and lists
        self._populate_reference_ct_combo()
        self.sct_checkboxes = []
        self._update_sct_list()

        # Dynamic rows containers
        self._dose_widgets = []
        self._session_widgets = []
        self._add_dose_selector()
        self._add_session_selector()

        # Output selectors rows
        self.min_output_selector = self._create_volume_selector("Create new min volume on Run")
        self.max_output_selector = self._create_volume_selector("Create new max volume on Run")
        self.min_export_checkbox = self._create_export_checkbox()
        self.max_export_checkbox = self._create_export_checkbox()
        if self.output_options_layout is not None:
            self.output_options_layout.addLayout(
                self._build_selector_row("Min dose volume:", self.min_output_selector, self.min_export_checkbox)
            )
            self.output_options_layout.addLayout(
                self._build_selector_row("Max dose volume:", self.max_output_selector, self.max_export_checkbox)
            )

        # Deform button callback
        if self.deform_button:
            self.deform_button.setToolTip(
                "Apply DVFs, store deformed doses as a sequence, and export min/max/uncertainty volumes"
            )
            if self._deform_callback:
                self.deform_button.clicked.connect(self._deform_callback)
            else:
                self.deform_button.clicked.connect(self._on_deform_and_compute)

        # Progress bar initial state
        if self.progress_bar:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(False)

        # Attach loaded UI to this widget
        layout = QVBoxLayout(self)
        layout.addWidget(ui_widget)
        self.setLayout(layout)

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

    def _run_cli_async(self, cli_module, params: dict, on_done, on_error) -> None:
        """Delegate to base implementation which centralizes CLI handling."""
        return super()._run_cli_async(cli_module, params, on_done, on_error)

    def _add_session_selector(self) -> None:
        row_widget = DvfSelectorRow(on_add=self._add_session_selector, on_remove=self._remove_session_selector)
        self._session_widgets.append(row_widget)
        if hasattr(self, "dvf_rows_layout"):
            self.dvf_rows_layout.insertWidget(len(self._session_widgets) - 1, row_widget)

    def _remove_session_selector(self, row_widget) -> None:
        self._remove_row(
            self._session_widgets,
            row_widget,
            getattr(self, "dvf_rows_layout", None),
            reset_combo=lambda r: getattr(getattr(r, "dvf_combo", None), "setCurrentIndex", lambda *_: None)(0),
        )

    def _add_dose_selector(self) -> None:
        row_widget = DoseSelectorRow(self._is_rtdose, on_add=self._add_dose_selector, on_remove=self._remove_dose_selector)
        self._dose_widgets.append(row_widget)
        if hasattr(self, "dose_rows_layout"):
            self.dose_rows_layout.insertWidget(len(self._dose_widgets) - 1, row_widget)

    def _remove_dose_selector(self, row_widget) -> None:
        self._remove_row(
            self._dose_widgets,
            row_widget,
            getattr(self, "dose_rows_layout", None),
            reset_combo=lambda r: getattr(getattr(r, "dose_combo", None), "setCurrentIndex", lambda *_: None)(0),
        )

    def _refresh_tps_lists(self) -> None:
        self._populate_reference_ct_combo()
        self._update_sct_list()

    def _populate_reference_ct_combo(self):
        previous_node_id = None
        current_index = self._combo_current_index(self.ref_ct_combo)
        previous_node = self._ref_ct_node_map.get(current_index)
        if previous_node is not None and hasattr(previous_node, "GetID"):
            try:
                previous_node_id = previous_node.GetID()
            except Exception:
                previous_node_id = None

        self.ref_ct_combo.clear()
        self._ref_ct_node_map.clear()
        self.ref_ct_combo.addItem("[Select Reference CT]", None)
        self._ref_ct_node_map[0] = None

        if slicer.mrmlScene is None:
            return

        shNode = self._get_sh_node()
        volume_nodes = slicer.util.getNodes("vtkMRMLScalarVolumeNode*")

        def sort_key(kv):
            node_name, node = kv
            in_sh = 0
            if shNode is not None:
                try:
                    in_sh = 0 if shNode.GetItemByDataNode(node) > 0 else 1
                except Exception:
                    in_sh = 1
            return (in_sh, node_name.lower())

        combo_index = 1
        for node_name, node in sorted(volume_nodes.items(), key=sort_key):
            if not self._is_dicom(node):
                continue
            self.ref_ct_combo.addItem(node_name)
            self._ref_ct_node_map[combo_index] = node
            combo_index += 1

        if previous_node_id:
            for idx, node in self._ref_ct_node_map.items():
                if node is not None and hasattr(node, "GetID"):
                    try:
                        if node.GetID() == previous_node_id:
                            self.ref_ct_combo.setCurrentIndex(idx)
                            break
                    except Exception:
                        continue

    def _update_sct_list(self):
        for checkbox in self.sct_checkboxes:
            checkbox.deleteLater()
        self.sct_checkboxes = []
        self._sct_node_map = {}
        if slicer.mrmlScene is None:
            return
        volume_nodes = slicer.util.getNodes("vtkMRMLScalarVolumeNode*")
        sequence_nodes = slicer.util.getNodes("vtkMRMLSequenceNode*")

        def _is_export_candidate_name(name: str) -> bool:
            # Keep sCT list clean: exclude anything that looks like dose.
            return not self._name_contains_dose(name)
        
        for node_name, node in volume_nodes.items():
            if (not _is_export_candidate_name(node_name)) or (node is not None and (not _is_export_candidate_name(self._safe_node_name(node)))):
                continue
            if self._is_dicom_volume(node):
                continue
            checkbox = QCheckBox(f"Volume: {node_name}")
            checkbox.setChecked(False)
            node_id = f"vol_{node_name}"
            checkbox.setProperty("node_id", node_id)
            checkbox.setProperty("node_type", "volume")
            self.sct_checkboxes.append(checkbox)
            self._sct_node_map[node_id] = node
            self.sct_list_layout.insertWidget(len(self.sct_checkboxes) - 1, checkbox)

        # sequences
        for node_name, node in sequence_nodes.items():
            if (not _is_export_candidate_name(node_name)) or (node is not None and (not _is_export_candidate_name(self._safe_node_name(node)))):
                continue
            # Only list sequences of scalar volumes (exclude transform sequences, etc.)
            data_class = node.GetDataNodeClassName()
            if data_class != "vtkMRMLScalarVolumeNode":
                continue

            # Determine sequence length via explicit attribute checks
            num_items = 0
            if hasattr(node, "GetNumberOfDataNodes"):
                try:
                    num_items = node.GetNumberOfDataNodes()
                except Exception:
                    num_items = 0
            elif hasattr(node, "GetSequenceAsNode"):
                idx = 0
                while True:
                    seq_node = node.GetSequenceAsNode(idx)
                    if not seq_node:
                        break
                    num_items += 1
                    idx += 1
            else:
                num_items = 0

            if num_items > 0:
                checkbox = QCheckBox(f"Sequence: {node_name} ({num_items} frames)")
                checkbox.setChecked(False)
                node_id = f"seq_{node_name}"
                checkbox.setProperty("node_id", node_id)
                checkbox.setProperty("node_type", "sequence")
                self.sct_checkboxes.append(checkbox)
                self._sct_node_map[node_id] = node
                self.sct_list_layout.insertWidget(len(self.sct_checkboxes) - 1, checkbox)

    def _refresh_session_combos(self) -> None:
        for dose_widget in self._dose_widgets:
            dose_widget.refresh_options()
        for session_widget in self._session_widgets:
            session_widget._populate_dvf_options()

    def _on_browse_export_dir(self):
        directory = QFileDialog.getExistingDirectory(None, "Select folder to export dose sessions")
        if directory:
            self.export_dir = directory
            display_path = directory if len(directory) < 50 else f"...{directory[-47:]}"
            self.export_dir_display.setText(display_path)
            self.export_dir_display.setStyleSheet("color: black;")

    def _expand_sequence_nodes(self, node, temp_nodes: list):
        if node is None:
            return []
        if isinstance(node, slicer.vtkMRMLSequenceNode):
            expanded = []
            count = getattr(node, "GetNumberOfDataNodes", lambda: 0)()
            for idx in range(count):
                data_node = node.GetNthDataNode(idx)
                cloned = slicer.mrmlScene.AddNewNodeByClass(data_node.GetClassName(), f"{node.GetName()}_{idx}")
                cloned.Copy(data_node)
                temp_nodes.append(cloned)
                expanded.append(cloned)
            return expanded
        return [node]

    def _create_or_update_volume(self, name: str, reference_node, array, existing_node=None):
        if reference_node is None:
            raise ValueError("Reference node is required to create output volumes")
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

    def _displacement_array_from_transform(self, transform_node, reference_volume_node, temp_nodes: list):
        """Return displacement field as a numpy array sampled on reference_volume_node grid.

        Output array is expected shape (k, j, i, 3) in physical units (typically mm).
        Uses available Slicer Transforms logic APIs; falls back gracefully if unavailable.
        """
        if slicer.mrmlScene is None or transform_node is None or reference_volume_node is None:
            return None

        # Snapshot nodes to detect side-effects created by Transforms logic.
        before_ids = set()
        try:
            n = slicer.mrmlScene.GetNumberOfNodes()
            for i in range(n):
                node = slicer.mrmlScene.GetNthNode(i)
                if node is not None:
                    before_ids.add(node.GetID())
        except Exception:
            before_ids = set()

        disp_node = None
        try:
            disp_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLVectorVolumeNode", f"dvf_disp_tmp_{uuid4().hex[:6]}"
            )
            temp_nodes.append(disp_node)
            # Avoid UI/render pipeline churn for temp nodes.
            try:
                disp_node.SetHideFromEditors(1)
                disp_node.SetSelectable(0)
                disp_node.SetSaveWithScene(0)
            except Exception:
                pass
        except Exception:
            disp_node = None

        if disp_node is None:
            return None

        created = False
        returned_arr = None

        def _try_logic(logic_obj) -> None:
            nonlocal created, returned_arr
            if logic_obj is None or created or returned_arr is not None:
                return
            for method_name in ("CreateDisplacementVolumeFromTransform", "CreateDisplacementVolumeFromTransformNode", "CreateDisplacementField"):
                if not hasattr(logic_obj, method_name):
                    continue
                method = getattr(logic_obj, method_name, None)
                if method is None:
                    continue
                for args in ((transform_node, reference_volume_node, disp_node), (transform_node, disp_node, reference_volume_node)):
                    try:
                        method(*args)
                        created = True
                        return
                    except TypeError:
                        continue
                    except Exception:
                        continue
                for args in ((transform_node, reference_volume_node), (transform_node,)):
                    try:
                        out = method(*args)
                    except TypeError:
                        continue
                    except Exception:
                        continue
                    if out is None:
                        continue
                    try:
                        returned_arr = slicer.util.arrayFromVolume(out).astype(np.float32, copy=False)
                    except Exception:
                        returned_arr = None
                    try:
                        if out.GetScene() == slicer.mrmlScene:
                            self.safe_remove(out)
                    except Exception:
                        pass
                    if returned_arr is not None:
                        created = True
                        return

        # Try Transforms module logic first (preferred).
        try:
            transforms_logic = slicer.modules.transforms.logic() if hasattr(slicer.modules, "transforms") else None
        except Exception:
            transforms_logic = None

        _try_logic(transforms_logic)

        # As a fallback, try vtkSlicerTransformLogic if exposed.
        if (not created) and hasattr(slicer, "vtkSlicerTransformLogic"):
            try:
                tl = slicer.vtkSlicerTransformLogic()
                _try_logic(tl)
            except Exception:
                created = False

        if returned_arr is not None:
            return returned_arr

        # Try reading from our vector output node.
        arr = None
        try:
            if disp_node.GetImageData() is not None:
                arr = slicer.util.arrayFromVolume(disp_node).astype(np.float32, copy=False)
        except Exception:
            arr = None

        # Side-effect fallback: find newly created vector volume or scalar "displacement magnitude".
        after_ids = set()
        try:
            n = slicer.mrmlScene.GetNumberOfNodes()
            for i in range(n):
                node = slicer.mrmlScene.GetNthNode(i)
                if node is not None:
                    after_ids.add(node.GetID())
        except Exception:
            after_ids = set()

        new_ids = list(after_ids - before_ids)
        disp_id = None
        try:
            disp_id = disp_node.GetID()
        except Exception:
            disp_id = None

        if arr is None and new_ids:
            for node_id in reversed(new_ids):
                if disp_id and node_id == disp_id:
                    continue
                node = slicer.mrmlScene.GetNodeByID(node_id)
                if node is None:
                    continue
                try:
                    if node.IsA("vtkMRMLVectorVolumeNode"):
                        arr = slicer.util.arrayFromVolume(node).astype(np.float32, copy=False)
                        break
                except Exception:
                    continue

        if arr is None and new_ids:
            for node_id in reversed(new_ids):
                if disp_id and node_id == disp_id:
                    continue
                node = slicer.mrmlScene.GetNodeByID(node_id)
                if node is None:
                    continue
                try:
                    if node.IsA("vtkMRMLScalarVolumeNode") and "displacement magnitude" in (node.GetName() or "").lower():
                        arr = slicer.util.arrayFromVolume(node).astype(np.float32, copy=False)
                        break
                except Exception:
                    continue

        # Clean up side-effects (keep disp_node for later cleanup via temp_nodes).
        for node_id in new_ids:
            if disp_id and node_id == disp_id:
                continue
            node = None
            try:
                node = slicer.mrmlScene.GetNodeByID(node_id)
            except Exception:
                node = None
            if node is not None:
                # Removing display nodes first helps avoid VTK pipeline warnings.
                try:
                    dn = node.GetDisplayNode() if hasattr(node, "GetDisplayNode") else None
                    if dn is not None and dn.GetScene() == slicer.mrmlScene:
                        slicer.mrmlScene.RemoveNode(dn)
                except Exception:
                    pass
                self.safe_remove(node)

        return arr

    def _name_contains_dose(self, name: str) -> bool:
        try:
            return "dose" in str(name).lower()
        except Exception:
            return False

    def _get_attr(self, node, attr_name: str):
        if node is None or not hasattr(node, "GetAttribute"):
            return None
        try:
            return node.GetAttribute(attr_name)
        except Exception:
            return None

    def _is_truthy_attr(self, val) -> bool:
        if val is None:
            return False
        s = str(val).strip().lower()
        return s not in ("", "0", "false", "no", "none")

    def _is_rtdose(self, node) -> bool:
        return self._is_truthy_attr(self._get_attr(node, "DicomRtImport.DoseVolume"))

    def _is_dicom(self, node) -> bool:
        return bool(self._get_attr(node, "DICOM.instanceUIDs")) or bool(self._get_attr(node, "DICOM.SeriesInstanceUID")) or bool(self._get_attr(node, "DICOM.StudyInstanceUID"))

    def _is_dicom_volume(self, node) -> bool:
        return self._is_rtdose(node) or self._is_dicom(node)

    def _safe_path_component(self, text: str, fallback: str = "item") -> str:
        """Sanitize a string for use as a directory name."""
        try:
            s = str(text or "").strip()
        except Exception:
            s = ""
        if not s:
            s = fallback
        cleaned = []
        for ch in s:
            if ch.isalnum() or ch in ("-", "_", "."):
                cleaned.append(ch)
            else:
                cleaned.append("_")
        out = "".join(cleaned).strip("._")
        return out or fallback

    def _ensure_node_in_sh_folder(self, node, folder_item_id):
        """Ensure node has an SH item and is parented under folder_item_id."""
        if slicer.mrmlScene is None or node is None or not folder_item_id:
            return
        shNode = self._get_sh_node()
        if shNode is None:
            return

        item_id = shNode.GetItemByDataNode(node)

        if item_id == 0:
            item_id = shNode.CreateItem(folder_item_id, node)

        shNode.SetItemParent(item_id, folder_item_id)

    def _get_or_create_output_folder_item(self, reference_node, folder_name: str):
        """Return a SubjectHierarchy folder item under the same subject (patient) as reference_node."""
        if slicer.mrmlScene is None or reference_node is None:
            return None
        shNode = self._get_sh_node()
        if shNode is None:
            return None

        ref_item_id = shNode.GetItemByDataNode(reference_node)
        if ref_item_id == 0:
            return None

        # Create the folder as a child of the Subject/Patient item (not under the dose node).
        parent_item_id = self._get_subject_item_id_from_sh(shNode, ref_item_id) or ref_item_id

        children = vtk.vtkIdList()
        shNode.GetItemChildren(parent_item_id, children, False)
        for i in range(children.GetNumberOfIds()):
            child_id = children.GetId(i)
            if shNode.GetItemName(child_id) == folder_name and shNode.GetItemDataNode(child_id) is None:
                return child_id

        return shNode.CreateFolderItem(parent_item_id, folder_name)

    def _get_subject_item_id_from_sh(self, shNode, item_id: int):
        """Climb SH parents until we reach the Patient (subject) item.

        Fallback: returns the immediate parent if Patient level cannot be detected.
        """
        if shNode is None or not item_id:
            return 0
        current = item_id
        last_parent = 0
        for _ in range(20):
            level = ""
            try:
                level = shNode.GetItemLevel(current)
            except Exception:
                level = ""
            if str(level).lower() == "patient":
                return current
            parent = 0
            try:
                parent = shNode.GetItemParent(current)
            except Exception:
                parent = 0
            if not parent:
                break
            last_parent = parent
            current = parent
        return last_parent

    def _build_selector_row(self, label_text: str, selector_widget, checkbox: QCheckBox = None) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel(label_text))
        row.addWidget(selector_widget, 1)
        if checkbox is not None:
            row.addWidget(checkbox)
        return row

    def _create_volume_selector(self, none_display: str):
        selector = slicer.qMRMLNodeComboBox()
        selector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        selector.showChildNodeTypes = False
        selector.noneEnabled = True
        selector.addEnabled = True
        selector.removeEnabled = True
        selector.editEnabled = True
        selector.renameEnabled = True
        selector.noneDisplay = none_display
        if slicer.mrmlScene:
            selector.setMRMLScene(slicer.mrmlScene)
        selector.setToolTip("Select an existing volume to overwrite or leave empty to create a new one on run")
        return selector

    def _create_sequence_selector(self, none_display: str):
        selector = slicer.qMRMLNodeComboBox()
        selector.nodeTypes = ["vtkMRMLSequenceNode"]
        selector.showChildNodeTypes = False
        selector.noneEnabled = True
        selector.addEnabled = True
        selector.removeEnabled = True
        selector.editEnabled = True
        selector.renameEnabled = True
        selector.noneDisplay = none_display
        if slicer.mrmlScene:
            selector.setMRMLScene(slicer.mrmlScene)
        selector.setToolTip("Select an existing sequence to overwrite or leave empty to create a new one on run")
        return selector

    def _create_export_checkbox(self):
        checkbox = QCheckBox("Export")
        checkbox.setChecked(True)
        checkbox.setToolTip("Unchecked to skip exporting this volume to the scene")
        return checkbox

    def _get_sequence_length(self, sequence_node) -> int:
        """Return the number of frames in a sequence node."""
        if sequence_node is None:
            return 0
        if hasattr(sequence_node, "GetNumberOfDataNodes"):
            try:
                return sequence_node.GetNumberOfDataNodes()
            except Exception:
                return 0
        if hasattr(sequence_node, "GetSequenceAsNode"):
            count = 0
            idx = 0
            while True:
                seq_node = sequence_node.GetSequenceAsNode(idx)
                if not seq_node:
                    break
                count += 1
                idx += 1
            return count
        return 0

    def _extract_ref_ct_metadata(self, ref_ct_node) -> dict:
        md = {}

        instance_uids_raw = ref_ct_node.GetAttribute("DICOM.instanceUIDs")
        instance_uid = str(instance_uids_raw).strip().split()[0]

        dicom_db = getattr(slicer, "dicomDatabase", None)
        tag_map = {
            "PatientID": "0010,0020",
            "PatientName": "0010,0010",
            "StudyID": "0020,0010",
            "StudyDescription": "0008,1030",
            "StudyDate": "0008,0020",
            "StudyTime": "0008,0030",
            "SOPClassUID": "0008,0016",
            "StudyInstanceUID": "0020,000D",
            "FrameOfReferenceUID": "0020,0052",
        }
        for out_key, dicom_tag in tag_map.items():
            if md.get(out_key):
                continue
            try:
                val = dicom_db.instanceValue(instance_uid, dicom_tag)
            except Exception:
                val = None
            if val:
                md[out_key] = val

        return md

    def _configure_export_tags(self, exp, directory: str, metadata: dict, series_number: int, series_description: str) -> None:
        exp.directory = directory
        for key, val in metadata.items():
            if key == 'PatientID':
                exp.setTag('PatientID', val)
            elif key == 'PatientName':
                exp.setTag('PatientName', val)
            elif key == 'StudyID':
                exp.setTag('StudyID', val)
            elif key == 'StudyDescription':
                exp.setTag('StudyDescription', val)
            elif key == 'StudyDate':
                exp.setTag('StudyDate', val)
            elif key == 'StudyTime':
                exp.setTag('StudyTime', val)
            elif key == 'SOPClassUID':
                exp.setTag('SOPClassUID', val)
            elif key == 'StudyInstanceUID':
                exp.setTag('StudyInstanceUID', val)
            elif key == 'FrameOfReferenceUID':
                exp.setTag('FrameOfReferenceUID', val)
        exp.setTag('SeriesNumber', str(series_number))
        exp.setTag('SeriesDescription', series_description)

    def _on_export_sct(self):
        if self._active_job is not None:
            try:
                QMessageBox.information(self, "Busy", "A computation is already running.")
            except Exception:
                pass
            return
        current_index = self._combo_current_index(self.ref_ct_combo)
        ref_ct_node = self._ref_ct_node_map.get(current_index)
        if ref_ct_node is None:
            QMessageBox.warning(self, "No Reference CT", "Please select a Reference CT to establish patient context.")
            return
        if not self.export_dir:
            QMessageBox.warning(self, "No Directory", "Please select an export directory.")
            return
        
        selected_nodes = []
        for checkbox in self.sct_checkboxes:
            if checkbox.isChecked():
                node_id = checkbox.property("node_id")
                node_type = checkbox.property("node_type")
                node = self._sct_node_map.get(node_id)
                if node:
                    selected_nodes.append((node, node_type))
        if not selected_nodes:
            QMessageBox.warning(self, "No Selection", "Please select at least one sCT volume or sequence to export.")
            return

        # Keep export simple (no async, no progress bar), but disable inputs to avoid mid-export edits.
        self._set_ui_busy(True)
        self._set_status("Exporting sCTs to DICOM...")
        try:
            os.makedirs(self.export_dir, exist_ok=True)
            exported_count = 0
            for session_idx, (node, node_type) in enumerate(selected_nodes):
                if node_type == "sequence":
                    exported_count += self._export_sequence_as_dicom(node, self.export_dir, ref_ct_node, session_idx)
                else:
                    exported_count += self._export_volume_as_dicom(node, self.export_dir, ref_ct_node, session_idx)
            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {exported_count} sCT volume(s) to DICOM.\n\nLocation: {self.export_dir}",
            )
        finally:
            self._set_ui_busy(False)
            self._set_status("")


    def _on_deform_and_compute(self):
        if self._active_job is not None:
            QMessageBox.information(self, "Busy", "A computation is already running.")
            return

        prev_warn = None
        try:
            prev_warn = vtk.vtkObject.GetGlobalWarningDisplay()
            vtk.vtkObject.SetGlobalWarningDisplay(0)
        except Exception:
            prev_warn = None

        output_name_val = self._line_edit_text(self.output_name_edit)

        base_name = output_name_val.strip() 

        selected_doses = []
        for dose_widget in self._dose_widgets:
            dn = dose_widget.get_dose_node()
            if dn is not None:
                selected_doses.append(dn)

        if not selected_doses:
            QMessageBox.warning(self, "Missing Inputs", "Select at least one dose before computing deformations.")
            return

        reference_volume = selected_doses[0]
        folder_item_id = self._get_or_create_output_folder_item(reference_volume, base_name)

        dvf_samples = []
        for idx, dvf_widget in enumerate(self._session_widgets, start=1):
            dvf_node = dvf_widget.get_dvf_node()
            dvf_type = dvf_widget.get_dvf_type()
            if not dvf_node:
                logger.warning(f"DVF {idx}: Missing DVF, skipping")
                continue
            dvf_samples.append((idx, dvf_node, dvf_type))

        if not dvf_samples:
            QMessageBox.warning(self, "Missing Inputs", "Select at least one DVF before computing deformations.")
            return

        job = {
            "base_name": base_name,
            "selected_doses": list(selected_doses),
            "dvf_samples": list(dvf_samples),
            "reference_volume": reference_volume,
            "folder_item_id": folder_item_id,
            "temp_nodes": [],
            "scratch_volume": None,
            "tasks": [],
            "task_index": 0,
            "sum_mag": None,
            "n_mag": 0,
            "dvf_mag_failures": 0,
            "mag_done": set(),
            "n_samples": 0,
            "sum_arr": None,
            "sumsq_arr": None,
            "min_arr": None,
            "max_arr": None,
            "vtk_warn_prev": prev_warn,
        }

        # Build a sequential task list (dose x DVF x frame).
        for dose_idx, base_dose in enumerate(selected_doses, start=1):
            for dvf_idx, dvf_node, dvf_type in dvf_samples:
                dvf_nodes = (
                    self._expand_sequence_nodes(dvf_node, job["temp_nodes"]) if dvf_type == "transform-sequence" else [dvf_node]
                )
                if not dvf_nodes:
                    logger.warning(f"DVF {dvf_idx}: No DVF available")
                    continue
                for frame_idx, dvf_to_use in enumerate(dvf_nodes):
                    job["tasks"].append(
                        {
                            "dose_idx": int(dose_idx),
                            "base_dose": base_dose,
                            "dvf_idx": int(dvf_idx),
                            "dvf_to_use": dvf_to_use,
                            "frame_idx": int(frame_idx),
                        }
                    )

        if not job["tasks"]:
            QMessageBox.warning(self, "No Valid Inputs", "No DVF frames available to compute deformations.")
            return

        self._active_job = job
        self._set_ui_busy(True)
        self._set_status("Computing deformed dose samples...")
        self._set_progress(0, visible=True)

        def _cleanup_temp_nodes():
            j = self._active_job
            if j is None:
                return
            for node in list(j.get("temp_nodes", [])):
                try:
                    if node is not None and slicer.mrmlScene is not None and node.GetScene() == slicer.mrmlScene:
                        slicer.mrmlScene.RemoveNode(node)
                except Exception:
                    pass
            scratch = j.get("scratch_volume", None)
            if scratch is not None:
                try:
                    if slicer.mrmlScene is not None and scratch.GetScene() == slicer.mrmlScene:
                        slicer.mrmlScene.RemoveNode(scratch)
                except Exception:
                    pass
                j["scratch_volume"] = None

        def _fail(msg: str):
            logger.error(str(msg))
            _cleanup_temp_nodes()
            try:
                prev = job.get("vtk_warn_prev", None)
                if prev is not None:
                    vtk.vtkObject.SetGlobalWarningDisplay(prev)
            except Exception:
                pass
            self._finish_job(False, msg)

        def _finalize():
            j = self._active_job
            if j is None:
                return

            n_samples = int(j.get("n_samples", 0))
            sum_arr = j.get("sum_arr", None)
            sumsq_arr = j.get("sumsq_arr", None)
            min_arr = j.get("min_arr", None)
            max_arr = j.get("max_arr", None)

            if n_samples == 0 or sum_arr is None:
                _fail("No deformed dose could be computed.")
                return

            self._set_status("Writing outputs...")
            self._set_progress(95, visible=True)

            mean_arr = sum_arr / float(n_samples)
            ex2 = (sumsq_arr / float(n_samples)) if sumsq_arr is not None else None
            var_arr = None
            if ex2 is not None:
                var_arr = np.maximum(ex2 - (mean_arr * mean_arr), 0.0)
            std_arr = np.sqrt(var_arr) if var_arr is not None else np.zeros_like(mean_arr)

            min_target = (
                self.min_output_selector.currentNode() if hasattr(self.min_output_selector, "currentNode") else None
            )
            max_target = (
                self.max_output_selector.currentNode() if hasattr(self.max_output_selector, "currentNode") else None
            )

            created_volume_names = []

            mean_name = f"dose_list_{j['base_name']}"
            mean_volume = self._create_or_update_volume(mean_name, j["reference_volume"], mean_arr, existing_node=None)
            created_volume_names.append(mean_volume.GetName())
            self._ensure_node_in_sh_folder(mean_volume, j["folder_item_id"])

            if self.min_export_checkbox.isChecked() and min_arr is not None:
                min_volume = self._create_or_update_volume(
                    f"min_dose_{j['base_name']}", j["reference_volume"], min_arr, min_target
                )
                created_volume_names.append(min_volume.GetName())
                if min_target is None and hasattr(self.min_output_selector, "setCurrentNode"):
                    self.min_output_selector.setCurrentNode(min_volume)
                self._ensure_node_in_sh_folder(min_volume, j["folder_item_id"])

            if self.max_export_checkbox.isChecked() and max_arr is not None:
                max_volume = self._create_or_update_volume(
                    f"max_dose_{j['base_name']}", j["reference_volume"], max_arr, max_target
                )
                created_volume_names.append(max_volume.GetName())
                if max_target is None and hasattr(self.max_output_selector, "setCurrentNode"):
                    self.max_output_selector.setCurrentNode(max_volume)
                self._ensure_node_in_sh_folder(max_volume, j["folder_item_id"])

            try:
                # Keep naming consistent with Phase 2/3 expectations: `uncertainty_{base}`.
                uncertainty_volume = self._create_or_update_volume(
                    f"uncertainty_dose_{j['base_name']}", j["reference_volume"], std_arr, existing_node=None
                )
                created_volume_names.append(uncertainty_volume.GetName())
                self._ensure_node_in_sh_folder(uncertainty_volume, j["folder_item_id"])
            except Exception:
                logger.exception("Failed to create uncertainty volume")

            n_mag = int(j.get("n_mag", 0))
            sum_mag = j.get("sum_mag", None)
            dvf_mag_failures = int(j.get("dvf_mag_failures", 0))
            if n_mag > 0 and sum_mag is not None:
                try:
                    mean_mag = sum_mag / float(n_mag)
                    dvf_mag_volume = self._create_or_update_volume(
                        f"dvf_magnitude_{j['base_name']}", j["reference_volume"], mean_mag, existing_node=None
                    )
                    created_volume_names.append(dvf_mag_volume.GetName())
                    self._ensure_node_in_sh_folder(dvf_mag_volume, j["folder_item_id"])
                except Exception:
                    logger.exception("Failed to compute/export dvf magnitude volume")
            else:
                logger.warning(
                    f"DVF magnitude not generated: n_mag={n_mag}, failures={dvf_mag_failures}. "
                    "Transform-to-displacement/magnitude conversion may be unavailable for the selected DVF(s)."
                )

            _cleanup_temp_nodes()

            try:
                QMessageBox.information(
                    self,
                    "Computation Complete",
                    (
                        f"Computed mean dose from {n_samples} deformed sample(s) (from {len(j['selected_doses'])} dose(s)).\n"
                        + (
                            "Created volumes: " + ", ".join(created_volume_names)
                            if created_volume_names
                            else "No volumes created."
                        )
                        + (
                            "\nDVF magnitude: not generated"
                            if (n_mag == 0)
                            else f"\nDVF magnitude: generated from {n_mag} sample(s)"
                        )
                    ),
                )
            except Exception:
                pass
            self._finish_job(True, "Done.")

            try:
                prev = job.get("vtk_warn_prev", None)
                if prev is not None:
                    vtk.vtkObject.SetGlobalWarningDisplay(prev)
            except Exception:
                pass

        def _tick():
            j = self._active_job
            if j is None:
                return
            tasks = j.get("tasks", [])
            i = int(j.get("task_index", 0))
            n = int(len(tasks))
            if i >= n:
                _finalize()
                return

            task = tasks[i]
            j["task_index"] = i + 1

            base_dose = task["base_dose"]
            dvf_to_use = task["dvf_to_use"]
            dose_idx = int(task.get("dose_idx", 0))
            dvf_idx = int(task.get("dvf_idx", 0))
            frame_idx = int(task.get("frame_idx", 0))

            # DVF magnitude: compute once per dvf/frame (not per dose).
            if dose_idx == 1:
                try:
                    dvf_id = dvf_to_use.GetID() if hasattr(dvf_to_use, "GetID") else None
                except Exception:
                    dvf_id = None
                if dvf_id and (dvf_id not in j.get("mag_done", set())):
                    try:
                        disp = self._displacement_array_from_transform(dvf_to_use, j["reference_volume"], j["temp_nodes"])
                        if disp is not None:
                            disp_arr = np.array(disp, dtype=np.float32, copy=False)
                            if disp_arr.ndim == 4 and disp_arr.shape[-1] == 3:
                                mag_i = np.sqrt(np.sum(np.square(disp_arr), axis=-1))
                            elif disp_arr.ndim == 4 and disp_arr.shape[0] == 3:
                                disp_last = np.moveaxis(disp_arr, 0, -1)
                                mag_i = np.sqrt(np.sum(np.square(disp_last), axis=-1))
                            elif disp_arr.ndim == 3:
                                mag_i = disp_arr
                            else:
                                mag_i = None

                            if mag_i is not None:
                                if j.get("sum_mag", None) is None:
                                    j["sum_mag"] = np.array(mag_i, dtype=np.float32, copy=True)
                                else:
                                    j["sum_mag"] = j["sum_mag"] + mag_i
                                j["n_mag"] = int(j.get("n_mag", 0)) + 1
                            else:
                                j["dvf_mag_failures"] = int(j.get("dvf_mag_failures", 0)) + 1
                        else:
                            j["dvf_mag_failures"] = int(j.get("dvf_mag_failures", 0)) + 1
                    except Exception:
                        j["dvf_mag_failures"] = int(j.get("dvf_mag_failures", 0)) + 1
                    try:
                        j["mag_done"].add(dvf_id)
                    except Exception:
                        pass

            # Deform dose sample using async CLI (avoid runSync).
            warped_volume = j.get("scratch_volume", None)
            if warped_volume is None:
                warped_name = f"{self._safe_node_name(base_dose)}_warped_tmp"
                warped_volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", warped_name)
                try:
                    warped_volume.SetHideFromEditors(1)
                    warped_volume.SetSelectable(0)
                    warped_volume.SetSaveWithScene(0)
                except Exception:
                    pass
                j["scratch_volume"] = warped_volume

            params = {
                "inputVolume": base_dose.GetID(),
                "referenceVolume": j["reference_volume"].GetID(),
                "outputVolume": warped_volume.GetID(),
                "interpolationType": "linear",
                "transformationFile": dvf_to_use.GetID(),
            }

            def _done():
                j2 = self._active_job
                if j2 is None:
                    return
                try:
                    arr = slicer.util.arrayFromVolume(warped_volume).astype(np.float32, copy=False)
                except Exception:
                    logger.warning(
                        f"Dose {dose_idx}: Could not read deformed volume for DVF {dvf_idx} (frame {frame_idx})"
                    )
                    try:
                        QTimer.singleShot(0, _tick)
                    except Exception:
                        _tick()
                    return

                if j2.get("sum_arr", None) is None:
                    j2["sum_arr"] = np.array(arr, dtype=np.float32, copy=True)
                    tmp = np.empty_like(arr)
                    np.multiply(arr, arr, out=tmp, casting="unsafe")
                    j2["sumsq_arr"] = tmp
                    j2["min_arr"] = np.array(arr, dtype=np.float32, copy=True)
                    j2["max_arr"] = np.array(arr, dtype=np.float32, copy=True)
                else:
                    np.add(j2["sum_arr"], arr, out=j2["sum_arr"], casting="unsafe")
                    tmp = np.empty_like(arr)
                    np.multiply(arr, arr, out=tmp, casting="unsafe")
                    np.add(j2["sumsq_arr"], tmp, out=j2["sumsq_arr"], casting="unsafe")
                    np.minimum(j2["min_arr"], arr, out=j2["min_arr"])
                    np.maximum(j2["max_arr"], arr, out=j2["max_arr"])

                j2["n_samples"] = int(j2.get("n_samples", 0)) + 1

                try:
                    p = int(5 + (85 * float(i + 1) / float(max(1, n))))
                except Exception:
                    p = 50
                self._set_progress(min(90, p), visible=True)
                self._set_status(f"Computing deformed dose samples... ({i + 1}/{n})")

                try:
                    QTimer.singleShot(0, _tick)
                except Exception:
                    _tick()

            def _err(exc):
                logger.exception(
                    f"Dose {dose_idx}: Failed to deform with DVF {dvf_idx} (frame {frame_idx}): {exc}"
                )
                _fail(f"Deformation failed: {exc}")

            self._run_cli_async(slicer.modules.resamplescalarvectordwivolume, params, _done, _err)

        try:
            QTimer.singleShot(0, _tick)
        except Exception:
            _tick()

    def _export_volume_as_dicom(self, volume_node, export_dir: str, ref_ct_node, session_idx: int) -> int:
        try:
            base_folder = os.path.join(export_dir, "sCT")
            vol_name = self._safe_node_name(volume_node)
            sct_folder = os.path.join(base_folder, f"{int(session_idx):03d}_{self._safe_path_component(vol_name, 'sCT')}")
            os.makedirs(sct_folder, exist_ok=True)

            import DICOMScalarVolumePlugin

            shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
            export_item_id = shNode.GetItemByDataNode(volume_node)
            if export_item_id == 0:
                logger.error("Could not find subject hierarchy item for volume to export")
                return 0
            refCtItemID = shNode.GetItemByDataNode(ref_ct_node)
            studyItemID = shNode.GetItemParent(refCtItemID)
            shNode.SetItemParent(export_item_id, studyItemID)

            exporter = DICOMScalarVolumePlugin.DICOMScalarVolumePluginClass()
            exportables = exporter.examineForExport(export_item_id)
            metadata = self._extract_ref_ct_metadata(ref_ct_node)
            for exp in exportables:
                self._configure_export_tags(exp, sct_folder, metadata, 1000 + session_idx, f"Synthetic CT - {volume_node.GetName()}")
            exporter.export(exportables)
            logger.info(f"Exported sCT (session {session_idx+1}) to DICOM: {sct_folder}")
            return 1
        except OSError as e:
            logger.error(f"I/O error during export for session {session_idx+1}: {e}")
            return 0
        except AttributeError as e:
            logger.error(f"Unexpected node attribute error: {e}")
            return 0
        except Exception as e:
            logger.exception(f"Failed to export sCT session {session_idx+1}")
            return 0

    def _export_sequence_as_dicom(self, sequence_node, export_dir: str, ref_ct_node, session_idx: int) -> int:
        try:
            exported_count = 0
            num_items = self._get_sequence_length(sequence_node)
            if num_items == 0:
                logger.warning(f"Could not determine sequence length for session {session_idx+1}")
                return 0

            base_folder = os.path.join(export_dir, "sCT")
            seq_name = self._safe_node_name(sequence_node)
            seq_root_folder = os.path.join(base_folder, f"{int(session_idx):03d}_{self._safe_path_component(seq_name, 'sCT_sequence')}")
            os.makedirs(seq_root_folder, exist_ok=True)

            import DICOMScalarVolumePlugin
            
            shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
            refCtItemID = shNode.GetItemByDataNode(ref_ct_node) if ref_ct_node is not None else 0
            if refCtItemID == 0:
                logger.error("Could not find subject hierarchy item for reference CT")
                return 0

            studyItemID = refCtItemID
            metadata = self._extract_ref_ct_metadata(ref_ct_node)
            for item_idx in range(num_items):
                try:
                    frame_node = sequence_node.GetNthDataNode(item_idx)
                    if frame_node is None or not frame_node.GetImageData():
                        continue

                    export_node = frame_node
                    temp_node = None
                    try:
                        temp_node = slicer.mrmlScene.AddNewNodeByClass(frame_node.GetClassName(), f"{sequence_node.GetName()}_{item_idx}")
                        temp_node.Copy(frame_node)
                        export_node = temp_node

                        exporter = DICOMScalarVolumePlugin.DICOMScalarVolumePluginClass()
                        export_item_id = shNode.GetItemByDataNode(export_node)
                       
                        shNode.SetItemParent(export_item_id, studyItemID)
                        exportables = exporter.examineForExport(export_item_id)

                        frame_id = export_node.GetID() if hasattr(export_node, "GetID") else None

                        filtered = []
                        for exp in exportables:
                            try:
                                if frame_id is not None:
                                    if hasattr(exp, "setNodeID"):
                                        exp.setNodeID(frame_id)
                                    elif hasattr(exp, "nodeID"):
                                        exp.nodeID = frame_id
                                if hasattr(exp, "setSubjectHierarchyItemID"):
                                    exp.setSubjectHierarchyItemID(export_item_id)
                                elif hasattr(exp, "subjectHierarchyItemID"):
                                    exp.subjectHierarchyItemID = export_item_id
                            except Exception:
                                pass

                            exp_node_id = getattr(exp, "nodeID", None)
                            if exp_node_id in (None, frame_id):
                                filtered.append(exp)
                        exportables = filtered

                        frame_folder = os.path.join(seq_root_folder, f"frame_{int(item_idx):04d}")
                        os.makedirs(frame_folder, exist_ok=True)
                        for exp in exportables:
                            self._configure_export_tags(exp, frame_folder, metadata, 2000 + session_idx * 100 + item_idx, f"sCT Sequence - Frame {item_idx}")
                        exporter.export(exportables)
                        logger.info(f"Exported sCT sequence item {item_idx} (session {session_idx+1}) to DICOM: {frame_folder}")
                        exported_count += 1
                    finally:
                        if temp_node is not None:
                            try:
                                slicer.mrmlScene.RemoveNode(temp_node)
                            except Exception:
                                pass
                except Exception as e:
                    logger.exception(f"Failed to export sequence item {item_idx}: {e}")
                    continue
            return exported_count
        except OSError as e:
            logger.error(f"I/O error during sequence export for session {session_idx+1}: {e}")
            return 0
        except Exception as e:
            logger.exception(f"Failed to export sequence for session {session_idx+1}: {e}")
            return 0
