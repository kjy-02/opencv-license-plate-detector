import os
import sys
import time
import cv2

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox

from plate_gui import Ui_MainWindow
from plate_detector import PlateParams, process_image, read_image_unicode, write_image_unicode


RESULT_DIR = "./results"


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.current_image_path = None
        self.outputs = None
        self.current_result_key = "best_result"

        self.params = PlateParams()

        self.process_timer = QTimer(self)
        self.process_timer.setSingleShot(True)
        self.process_timer.timeout.connect(self.process_current_image)

        self.configure_ui()

    def configure_ui(self):
        self.setWindowTitle("License Plate Detection GUI")

        self.actionOpenImage.triggered.connect(self.open_image)
        self.actionSaveResult.triggered.connect(self.save_result)
        self.actionExit.triggered.connect(self.close)

        self.btn_open.clicked.connect(self.open_image)
        self.btn_save.clicked.connect(self.save_result)
        self.btn_exit.clicked.connect(self.close)

        if hasattr(self, "btn_reset"):
            self.btn_reset.clicked.connect(self.reset_default_params)

        self.combo_result_mode.clear()
        self.combo_result_mode.addItems([
            "Best Plate",
            "All Candidates",
            "Character Candidates",
            "Adaptive Threshold",
            "Morphology",
            "Gray",
            "CLAHE"
        ])
        self.combo_result_mode.currentIndexChanged.connect(self.change_result_mode)

        self.configure_sliders()
        self.update_detection_info_empty()

        self.statusbar.showMessage("Ready")

    def configure_sliders(self):
        self.slider_adaptive_block.setMinimum(3)
        self.slider_adaptive_block.setMaximum(51)

        self.slider_adaptive_c.setMinimum(0)
        self.slider_adaptive_c.setMaximum(30)

        self.slider_morph_width.setMinimum(3)
        self.slider_morph_width.setMaximum(40)

        self.slider_morph_height.setMinimum(3)
        self.slider_morph_height.setMaximum(20)

        self.slider_min_area.setMinimum(100)
        self.slider_min_area.setMaximum(5000)

        self.slider_max_area.setMinimum(5000)
        self.slider_max_area.setMaximum(80000)

        self.slider_min_aspect.setMinimum(10)
        self.slider_min_aspect.setMaximum(50)

        self.slider_max_aspect.setMinimum(30)
        self.slider_max_aspect.setMaximum(100)

        self.set_sliders_from_params(block_signals=True)

        self.slider_adaptive_block.valueChanged.connect(self.on_parameter_changed)
        self.slider_adaptive_c.valueChanged.connect(self.on_parameter_changed)
        self.slider_morph_width.valueChanged.connect(self.on_parameter_changed)
        self.slider_morph_height.valueChanged.connect(self.on_parameter_changed)
        self.slider_min_area.valueChanged.connect(self.on_parameter_changed)
        self.slider_max_area.valueChanged.connect(self.on_parameter_changed)
        self.slider_min_aspect.valueChanged.connect(self.on_parameter_changed)
        self.slider_max_aspect.valueChanged.connect(self.on_parameter_changed)

        self.update_slider_labels()

    def set_sliders_from_params(self, block_signals=False):
        sliders = [
            self.slider_adaptive_block,
            self.slider_adaptive_c,
            self.slider_morph_width,
            self.slider_morph_height,
            self.slider_min_area,
            self.slider_max_area,
            self.slider_min_aspect,
            self.slider_max_aspect
        ]

        if block_signals:
            for slider in sliders:
                slider.blockSignals(True)

        adaptive_block = self.params.adaptive_block_size
        if adaptive_block % 2 == 0:
            adaptive_block += 1

        self.slider_adaptive_block.setValue(adaptive_block)
        self.slider_adaptive_c.setValue(self.params.adaptive_c)
        self.slider_morph_width.setValue(self.params.morph_width)
        self.slider_morph_height.setValue(self.params.morph_height)
        self.slider_min_area.setValue(self.params.min_area)
        self.slider_max_area.setValue(self.params.max_area)
        self.slider_min_aspect.setValue(int(self.params.min_aspect * 10))
        self.slider_max_aspect.setValue(int(self.params.max_aspect * 10))

        if block_signals:
            for slider in sliders:
                slider.blockSignals(False)

        self.update_slider_labels()

    def update_slider_labels(self):
        block_value = self.slider_adaptive_block.value()

        if block_value % 2 == 0:
            block_value += 1

        self.value_adaptive_block.setText(str(block_value))
        self.value_adaptive_c.setText(str(self.slider_adaptive_c.value()))
        self.value_morph_width.setText(str(self.slider_morph_width.value()))
        self.value_morph_height.setText(str(self.slider_morph_height.value()))
        self.value_min_area.setText(str(self.slider_min_area.value()))
        self.value_max_area.setText(str(self.slider_max_area.value()))
        self.value_min_aspect.setText(f"{self.slider_min_aspect.value() / 10:.1f}")
        self.value_max_aspect.setText(f"{self.slider_max_aspect.value() / 10:.1f}")

    def on_parameter_changed(self):
        self.update_slider_labels()

        if self.current_image_path is None:
            return

        self.statusbar.showMessage("Parameter changed... reprocessing")
        self.process_timer.start(150)

    def update_params_from_sliders(self):
        adaptive_block = self.slider_adaptive_block.value()

        if adaptive_block % 2 == 0:
            adaptive_block += 1
            self.slider_adaptive_block.setValue(adaptive_block)

        self.params.adaptive_block_size = adaptive_block
        self.params.adaptive_c = self.slider_adaptive_c.value()

        self.params.morph_width = self.slider_morph_width.value()
        self.params.morph_height = self.slider_morph_height.value()

        self.params.min_area = self.slider_min_area.value()
        self.params.max_area = self.slider_max_area.value()

        if self.params.min_area >= self.params.max_area:
            self.params.min_area = max(100, self.params.max_area - 100)
            self.slider_min_area.setValue(self.params.min_area)

        self.params.min_aspect = self.slider_min_aspect.value() / 10
        self.params.max_aspect = self.slider_max_aspect.value() / 10

        if self.params.min_aspect >= self.params.max_aspect:
            self.params.min_aspect = max(1.0, self.params.max_aspect - 0.1)
            self.slider_min_aspect.setValue(int(self.params.min_aspect * 10))

    def reset_default_params(self):
        self.params = PlateParams()
        self.set_sliders_from_params(block_signals=True)

        self.statusbar.showMessage("Default parameters restored")

        if self.current_image_path is not None:
            self.process_current_image()
        else:
            self.update_detection_info_empty()

    def update_detection_info_empty(self):
        if not hasattr(self, "label_detection_info"):
            return

        self.label_detection_info.setTextFormat(Qt.RichText)
        self.label_detection_info.setText(
            """
            <b>Detection Info</b>
            <table cellspacing="4" cellpadding="2">
                <tr>
                    <td>Plate Found</td><td>-</td>
                    <td>Best Score</td><td>-</td>
                </tr>
                <tr>
                    <td>Plate Candidates</td><td>-</td>
                    <td>Best Chars</td><td>-</td>
                </tr>
                <tr>
                    <td>Char Candidates</td><td>-</td>
                    <td>Aspect</td><td>-</td>
                </tr>
                <tr>
                    <td>Char Groups</td><td>-</td>
                    <td>Angle</td><td>-</td>
                </tr>
                <tr>
                    <td>Area</td><td>-</td>
                    <td></td><td></td>
                </tr>
            </table>
            """
        )

    def update_detection_info(self):
        if not hasattr(self, "label_detection_info"):
            return

        if self.outputs is None:
            self.update_detection_info_empty()
            return

        plate_found = self.outputs.get("plate_found", False)
        candidate_count = self.outputs.get("candidate_count", 0)
        char_candidate_count = self.outputs.get("char_candidate_count", 0)
        char_group_count = self.outputs.get("char_group_count", 0)
        best_score = self.outputs.get("best_score", 0.0)
        best_aspect = self.outputs.get("best_aspect", 0.0)
        best_area = self.outputs.get("best_area", 0)
        best_char_count = self.outputs.get("best_char_count", 0)
        best_angle = self.outputs.get("best_angle", 0.0)

        self.label_detection_info.setTextFormat(Qt.RichText)
        self.label_detection_info.setText(
            f"""
            <b>Detection Info</b>
            <table cellspacing="4" cellpadding="2">
                <tr>
                    <td>Plate Found</td><td><b>{plate_found}</b></td>
                    <td>Best Score</td><td><b>{best_score:.2f}</b></td>
                </tr>
                <tr>
                    <td>Plate Candidates</td><td>{candidate_count}</td>
                    <td>Best Chars</td><td>{best_char_count}</td>
                </tr>
                <tr>
                    <td>Char Candidates</td><td>{char_candidate_count}</td>
                    <td>Aspect</td><td>{best_aspect:.2f}</td>
                </tr>
                <tr>
                    <td>Char Groups</td><td>{char_group_count}</td>
                    <td>Angle</td><td>{best_angle:.1f}</td>
                </tr>
                <tr>
                    <td>Area</td><td>{best_area}</td>
                    <td></td><td></td>
                </tr>
            </table>
            """
        )

    def open_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "./samples",
            "Image files (*.jpg *.jpeg *.png *.bmp)"
        )

        if not filename:
            return

        self.current_image_path = filename

        src = read_image_unicode(filename)

        if src is None:
            QMessageBox.warning(self, "Error", "이미지를 읽을 수 없습니다.")
            return

        self.show_cv_image(src, self.label_original)
        self.statusbar.showMessage(f"Image loaded: {os.path.basename(filename)}")

        self.process_current_image()

    def process_current_image(self):
        if self.current_image_path is None:
            return

        self.update_params_from_sliders()

        start_time = time.time()

        try:
            self.outputs = process_image(self.current_image_path, self.params)
        except Exception as e:
            QMessageBox.warning(self, "Processing Error", str(e))
            return

        elapsed_ms = (time.time() - start_time) * 1000

        self.update_result_display()
        self.update_plate_crop_display()
        self.update_detection_info()

        status_text = (
            f"Plate found={self.outputs['plate_found']} | "
            f"plate candidates={self.outputs['candidate_count']} | "
            f"char candidates={self.outputs['char_candidate_count']} | "
            f"char groups={self.outputs['char_group_count']} | "
            f"time={elapsed_ms:.1f} ms"
        )

        self.statusbar.showMessage(status_text)

    def change_result_mode(self):
        text = self.combo_result_mode.currentText()

        if text == "Best Plate":
            self.current_result_key = "best_result"
        elif text == "All Candidates":
            self.current_result_key = "candidates_result"
        elif text == "Character Candidates":
            self.current_result_key = "char_candidates_result"
        elif text == "Adaptive Threshold":
            self.current_result_key = "adaptive"
        elif text == "Morphology":
            self.current_result_key = "morph"
        elif text == "Gray":
            self.current_result_key = "gray"
        elif text == "CLAHE":
            self.current_result_key = "gray_clahe"
        else:
            self.current_result_key = "best_result"

        self.update_result_display()

    def update_result_display(self):
        if self.outputs is None:
            return

        result_img = self.outputs[self.current_result_key]
        self.show_cv_image(result_img, self.label_result)

    def update_plate_crop_display(self):
        if self.outputs is None:
            return

        plate_crop = self.outputs["plate_crop"]
        self.show_cv_image(plate_crop, self.label_plate_crop)

    def save_result(self):
        if self.outputs is None:
            QMessageBox.information(self, "Info", "저장할 결과가 없습니다.")
            return

        os.makedirs(RESULT_DIR, exist_ok=True)

        if self.current_image_path is None:
            base_name = "result"
        else:
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]

        default_name = f"{self.current_result_key}_{base_name}.jpg"
        default_path = os.path.join(RESULT_DIR, default_name)

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Result",
            default_path,
            "JPG files (*.jpg);;PNG files (*.png)"
        )

        if not filename:
            return

        result_img = self.outputs[self.current_result_key]
        success = write_image_unicode(filename, result_img)

        if success:
            self.statusbar.showMessage(f"Saved: {filename}")
        else:
            QMessageBox.warning(self, "Save Error", "이미지 저장에 실패했습니다.")

    def show_cv_image(self, cv_img, label):
        if cv_img is None:
            label.setText("No Image")
            return

        if len(cv_img.shape) == 2:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        height, width, channel = rgb_img.shape
        bytes_per_line = channel * width

        q_img = QImage(
            rgb_img.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(q_img)

        scaled_pixmap = pixmap.scaled(
            label.width(),
            label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        label.setPixmap(scaled_pixmap)
        label.setAlignment(Qt.AlignCenter)

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if self.current_image_path is not None:
            src = read_image_unicode(self.current_image_path)

            if src is not None:
                self.show_cv_image(src, self.label_original)

        if self.outputs is not None:
            self.update_result_display()
            self.update_plate_crop_display()
            self.update_detection_info()


def main():
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
