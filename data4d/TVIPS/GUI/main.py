# fmt: off
import math
import os
from pathlib import Path
import sys

import PySide2  # needed before Qt5 gets imported
from PySide2.QtCore import QCoreApplication, QPointF, Qt, QThread, Signal
from PySide2.QtWidgets import QApplication, QFileDialog, QMainWindow
import numpy as np
import pyqtgraph as pg
from skimage import exposure

# fmt: on


pg.setConfigOptions(imageAxisOrder="row-major")  # default for numpy

from ...utils import bin_box
from ..tvipsfile import TVIPS
from .PySide2_dynamic import loadUi

# hotfix 3.9 MacOS Big Sur bug
if sys.platform == "darwin":
    os.environ["QT_MAC_WANTS_LAYER"] = "1"


class TVIPSMainWindow(QMainWindow):
    def __init__(self):
        super(TVIPSMainWindow, self).__init__()
        self.tvips = None

        # initialize UI
        self.load_main_ui()
        self.connect_widgets()

        # initalize default attributes
        self._current_diffraction_image = None

    def load_main_ui(self):
        # load ui file into MainWindow
        path = str(Path(__file__).parent.joinpath("main.ui"))
        loadUi(path, self, customWidgets=dict((w.__name__, w) for w in (pg.ImageView,)))
        # self.setWindowIcon(QIcon("icon.png"))
        self.statusBar = self.statusBar()
        self.show()

    def connect_widgets(self):
        self.pushButton_open_tvips_file.clicked.connect(self.open_tvips_file)
        self.spinBox_scan_x.valueChanged.connect(self.update_VR)
        self.spinBox_scan_y.valueChanged.connect(self.update_VR)
        self.spinBox_scan_offset.valueChanged.connect(self.update_VR)

        self.spinBox_binning_factor.valueChanged.connect(self.update_diffraction)
        self.spinBox_clip_min.valueChanged.connect(self.update_diffraction)
        self.spinBox_clip_max.valueChanged.connect(self.update_diffraction)
        self.checkBox_rescale_intensities.stateChanged.connect(self.update_diffraction)
        self.checkBox_show_processed_image.stateChanged.connect(self.update_diffraction)
        self.pushButton_export_blo.clicked.connect(self.export_blo_file)
        self.pushButton_VBF_calculate.clicked.connect(self._calculate_VBF)

        self.graphicsView_VR.scene.sigMouseMoved.connect(self.update_diffraction)
        self.graphicsView_diffraction.scene.sigMouseMoved.connect(
            self._update_diffraction_coords
        )

    def open_tvips_file(self):
        fname, _ = QFileDialog.getOpenFileName(
            parent=self, filter="TVIPS (*.tvips)", caption="Load TVIPS file"
        )

        if fname:
            self.lineEdit_tvips_fname.setText(fname)
            self.tvips = TVIPS(fname)
            self.statusBar.showMessage("TVIPS file loaded.")

            # update scan parameters on GUI, need to temporarily disconnect slots
            self.spinBox_scan_x.valueChanged.disconnect()
            self.spinBox_scan_y.valueChanged.disconnect()
            self.spinBox_scan_offset.valueChanged.disconnect()

            self.scan_x = self.tvips.scan_shape[1]
            self.scan_y = self.tvips.scan_shape[0]
            self.scan_offset = self.tvips.scan_offset

            # reconnect
            self.spinBox_scan_x.valueChanged.connect(self.update_VR)
            self.spinBox_scan_y.valueChanged.connect(self.update_VR)
            self.spinBox_scan_offset.valueChanged.connect(self.update_VR)
            self.update_VR()
        else:
            self.tvips = None
            self.statusBar.showMessage("File not selected.")

    def update_VR(self, run=True):
        if self.tvips is None:
            return

        scan_offset = self.scan_offset
        scan_shape = self.scan_shape
        n = math.prod(scan_shape) + scan_offset

        if n > self.tvips.number_of_frames:
            self.statusBar.showMessage(
                f"Requested scan size: {n} has too many frames for TVIPS file: {self.tvips.number_of_frames}"
            )
        else:
            # update TVIPS scan parameters
            self.tvips.scan_x = self.scan_x
            self.tvips.scan_y = self.scan_y
            self.tvips.scan_offset = self.scan_offset

            # get VBF intensities
            vbf = self.tvips.vbf_intensities
            if vbf is None:
                self._calculate_VBF()
            else:
                image = np.reshape(vbf[scan_offset:n], scan_shape)
                self._update_VR_image(image)

    def _update_VR_image(self, image):
        assert image.ndim == 2, "image should have 2 dimensions."
        self.graphicsView_VR.setImage(image)
        self.statusBar.showMessage(
            f"VR updated with scan shape (x, y): {self.scan_shape[::-1]}"
        )

    def update_diffraction(self, pos=None):
        """Update diffraction GraphicsView."""
        if self.tvips is None:  # no dataset
            return

        if isinstance(pos, QPointF):  # fn called due to mouse moved on plot
            view = self.graphicsView_VR.getView()
            pos = view.mapSceneToView(pos)
            ij = (int(pos.y()), int(pos.x()))

            if all(i >= 0 for i in ij) and all(
                i < s for i, s in zip(ij, self.scan_shape)
            ):  # if in bounds
                # get new diffraction image
                self._current_diffraction_image = self.tvips[ij]
                self.label_scan_coords.setText(f"Scan (x, y) = {ij[::-1]}")

        # get image to work with
        image = self._current_diffraction_image

        if image is None:  # not initially set
            return

        if self.checkBox_show_processed_image.isChecked():
            try:
                image = self.process_image(image)
            except AssertionError:
                self.statusBar.showMessage(
                    f"Array with shape: {image.shape} is not binnable by factor: {self.binning_factor}."
                )
                return

        self.graphicsView_diffraction.setImage(image)

    def _calculate_VBF(self):
        self._thread_vbf = VBF_calculator(
            self.tvips,
            self.spinBox_VBF_aperture_radius.value(),
            self.checkBox_center_VBF_aperture.isChecked(),
            self.spinBox_VBF_center_crop_size.value(),
            self.doubleSpinBox_VBF_center_sigma.value(),
        )
        self._thread_vbf.progress_update.connect(self.update_progressbar)
        self._thread_vbf.finished.connect(self._vbf_calculation_finished)
        self.statusBar.showMessage("Calculating VBF intensities...")
        self._thread_vbf.start()

    def _update_diffraction_coords(self, pos):
        """Update diffraction coords label."""
        view = self.graphicsView_diffraction.getView()
        pos = view.mapSceneToView(pos)
        ij = (int(pos.y()), int(pos.x()))
        self.label_diffraction_coords.setText(f"Diffraction (i, j) = {ij}")

    def process_image(self, image):
        """Apply image processing to a given image."""
        image = bin_box(image, self.binning_factor)

        image = image.clip(*self.clipping_values)

        if self.rescale_intensities:
            image = exposure.rescale_intensity(image, out_range=np.uint8)

        return image

    def export_blo_file(self):
        """Export processed .tvips to .blo file."""
        if self.tvips is None:
            return

        fname, _ = QFileDialog.getSaveFileName(
            parent=self,
            dir=str(Path(self.tvips.files[0]).with_suffix(".blo")),
            filter="BLO (*.blo)",
            caption="Export BLO",
        )

        self._thread_export = Exporter(
            self.tvips, fname, self.binning_factor, self.clipping_values
        )

        self._thread_export.finished.connect(self._export_finished)
        self._thread_export.progress_update.connect(self.update_progressbar)

        self.statusBar.showMessage(f"Exporting BLO file...")
        self._thread_export.start()

    def _export_finished(self):
        self.statusBar.showMessage(f"BLO file exported.")

    def _vbf_calculation_finished(self):
        self.statusBar.showMessage(f"VBF calculated.")
        self.update_VR()  # rerun VR func

    def update_progressbar(self, val):
        self.progressBar.setValue(val)

    @property
    def rescale_intensities(self):
        return self.checkBox_rescale_intensities.isChecked()

    @property
    def clipping_values(self):
        return (self.spinBox_clip_min.value(), self.spinBox_clip_max.value())

    @property
    def binning_factor(self):
        return self.spinBox_binning_factor.value()

    @property
    def scan_x(self):
        return self.spinBox_scan_x.value()

    @scan_x.setter
    def scan_x(self, x):
        self.spinBox_scan_x.setValue(int(x))

    @property
    def scan_y(self):
        return self.spinBox_scan_y.value()

    @scan_y.setter
    def scan_y(self, y):
        self.spinBox_scan_y.setValue(int(y))

    @property
    def scan_offset(self):
        return self.spinBox_scan_offset.value()

    @scan_offset.setter
    def scan_offset(self, o):
        self.spinBox_scan_offset.setValue(int(o))

    @property
    def scan_shape(self):
        """Scan shape in (i, j)."""
        return (self.spinBox_scan_y.value(), self.spinBox_scan_x.value())


class Exporter(QThread):
    """Run .blo export in thread."""

    progress_update = Signal(int)

    def __init__(self, tvips, fname_export, binning_factor, clipping_values):
        super().__init__()
        self.tvips = tvips
        self.fname_export = fname_export
        self.binning_factor = binning_factor
        self.clipping_values = clipping_values

    def run(self):
        """Start export thread."""
        self.tvips.create_blo_file(
            self.fname_export,
            self.binning_factor,
            self.clipping_values,
            check_exists=False,
            signal=self.progress_update,
        )


class VBF_calculator(QThread):
    """Run VBF calculation in thread."""

    progress_update = Signal(int)

    def __init__(self, tvips, radius, recenter, side, sigma):
        super().__init__()
        self.tvips = tvips
        self.radius = radius
        self.recenter = recenter
        self.side = side
        self.sigma = sigma

    def run(self):
        """Start VBF calculation thread."""
        self.tvips.calculate_virtual_bright_field_reconstruction(
            radius=self.radius,
            signal=self.progress_update,
            recenter=self.recenter,
            side=self.side,
            sigma=self.sigma,
        )


def main():
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication.instance()
    if app is None:
        app = QApplication()
    app.setQuitOnLastWindowClosed(True)
    window = TVIPSMainWindow()
    window.show()
    sys.exit(app.exec_())

