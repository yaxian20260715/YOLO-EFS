import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QPushButton, QFileDialog, QSlider, QVBoxLayout,
    QHBoxLayout, QGroupBox, QTextEdit, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

class VideoProcessingThread(QThread):
    frame_processed = pyqtSignal(QImage, dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, stream_source, model_obj, conf_slider_func):
        super().__init__()
        self.stream_source = stream_source
        self.model = model_obj
        self.get_current_conf = conf_slider_func
        self.running = True

    def run(self):
        try:
            cap = cv2.VideoCapture(self.stream_source)
            if not cap.isOpened():
                self.error_occurred.emit("无法打开指定的视频流输入源，请核对硬件或文件路径。")
                return
            while cap.isOpened() and self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                conf = self.get_current_conf()
                results = self.model(frame, conf=conf)
                detection_summary = {"fire": 0, "smoke": 0, "fps": 0}
                annotated_frame = frame.copy()

                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf_val = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id]
                    if label in detection_summary:
                        detection_summary[label] += 1
                    color = (0, 0, 255) if label == "fire" else (255, 255, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)
                    label_text = f"{label} {conf_val:.2f}"
                    cv2.putText(annotated_frame, label_text, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                detection_summary["fps"] = int(fps)

                height, width, channel = annotated_frame.shape
                bytes_per_line = 3 * width
                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                q_img = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                self.frame_processed.emit(q_img, detection_summary)
            cap.release()
        except Exception as e:
            self.error_occurred.emit(f"连续流处理流产生异常: {str(e)}")

    def stop(self):
        self.running = False
        self.wait()

class YOLOEFSGuiWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 让程序自动找到和它同目录下的 best.pt
        self.model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
        self.model = None
        self.video_thread = None
        self.init_ui_layout()
        self.auto_load_model()

    def init_ui_layout(self):
        self.setWindowTitle("YOLO-EFS 林火烟雾早期检测与预警系统")
        self.setGeometry(100, 100, 1280, 720)
        self.setMinimumSize(1024, 600)
        self.setStyleSheet("background-color: #F5F5F5; font-family: 'Segoe UI', Arial, sans-serif;")

        self.view_label = QLabel("正在自动初始化系统，请稍候...")
        self.view_label.setAlignment(Qt.AlignCenter)
        self.view_label.setScaledContents(False)
        self.view_label.setSizePolicy(QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored))
        self.view_label.setStyleSheet(
            "background-color: #FFFFFF; border: 1px solid #DCDCDC; border-radius: 4px; font-size: 14px; color: #666666;")

        self.open_img_btn = QPushButton("图片检测")
        self.open_video_btn = QPushButton("连续视频流")
        self.open_cam_btn = QPushButton("打开摄像头")
        self.stop_btn = QPushButton("暂停检测")
        self.snapshot_btn = QPushButton("截图保存")

        btn_style = """
            QPushButton { background-color: #2196F3; color: white; border-radius: 4px; padding: 8px 16px; font-size: 13px; font-weight: bold; }
            QPushButton:hover { background-color: #1E88E5; }
            QPushButton:pressed { background-color: #1565C0; }
            QPushButton:disabled { background-color: #B0BEC5; color: #FFFFFF; }
        """
        for btn in [self.open_img_btn, self.open_video_btn, self.open_cam_btn, self.stop_btn, self.snapshot_btn]:
            btn.setStyleSheet(btn_style)
        self.stop_btn.setStyleSheet("background-color: #E53935;")

        self.open_img_btn.setEnabled(False)
        self.open_video_btn.setEnabled(False)
        self.open_cam_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.snapshot_btn.setEnabled(False)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.open_img_btn)
        bottom_layout.addWidget(self.open_video_btn)
        bottom_layout.addWidget(self.open_cam_btn)
        bottom_layout.addWidget(self.stop_btn)
        bottom_layout.addWidget(self.snapshot_btn)

        left_vertical_layout = QVBoxLayout()
        left_vertical_layout.addWidget(self.view_label, stretch=8)
        left_vertical_layout.addLayout(bottom_layout, stretch=1)

        right_sidebar_layout = QVBoxLayout()

        stat_group = QGroupBox("检测结果")
        stat_group.setStyleSheet("font-weight: bold; font-size: 13px;")
        stat_layout = QVBoxLayout()
        self.fire_num_label = QLabel("火焰检测： 尚未开始")
        self.smoke_num_label = QLabel("烟雾检测： 尚未开始")
        self.total_num_label = QLabel("目标总数： 0")
        label_style = "font-weight: normal; font-size: 13px; margin: 4px 0;"
        self.fire_num_label.setStyleSheet(label_style)
        self.smoke_num_label.setStyleSheet(label_style)
        self.total_num_label.setStyleSheet(label_style)
        stat_layout.addWidget(self.fire_num_label)
        stat_layout.addWidget(self.smoke_num_label)
        stat_layout.addWidget(self.total_num_label)
        stat_group.setLayout(stat_layout)

        warn_group = QGroupBox("告警提示")
        warn_group.setStyleSheet("font-weight: bold; font-size: 13px; color: #E53935;")
        warn_layout = QVBoxLayout()
        self.warn_text_box = QTextEdit()
        self.warn_text_box.setReadOnly(True)
        self.warn_text_box.setPlainText("系统正在读取预设权重路径。")
        self.warn_text_box.setStyleSheet(
            "background-color: #FFFFFF; font-weight: normal; font-size: 12px; border: 1px solid #DCDCDC;")
        warn_layout.addWidget(self.warn_text_box)
        warn_group.setLayout(warn_layout)

        param_group = QGroupBox("检测参数")
        param_group.setStyleSheet("font-weight: bold; font-size: 13px;")
        param_layout = QVBoxLayout()
        conf_title_layout = QHBoxLayout()
        conf_title_label = QLabel("检测置信度阈值：")
        conf_title_label.setStyleSheet("font-weight: normal;")
        self.conf_num_indicator = QLabel("45%")
        self.conf_num_indicator.setStyleSheet("font-weight: bold; color: #2196F3;")
        conf_title_layout.addWidget(conf_title_label)
        conf_title_layout.addWidget(self.conf_num_indicator)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(10)
        self.conf_slider.setMaximum(95)
        self.conf_slider.setValue(45)
        self.fps_indicator_label = QLabel("当前帧率: -- FPS")
        self.fps_indicator_label.setStyleSheet("font-weight: normal; margin-top: 8px;")
        param_layout.addLayout(conf_title_layout)
        param_layout.addWidget(self.conf_slider)
        param_layout.addWidget(self.fps_indicator_label)
        param_group.setLayout(param_layout)

        status_group = QGroupBox("系统状态")
        status_group.setStyleSheet("font-weight: bold; font-size: 13px;")
        status_layout = QVBoxLayout()
        self.sys_status_label = QLabel("正在初始化")
        self.sys_status_label.setStyleSheet("color: #FF9800; font-size: 13px;")
        status_layout.addWidget(self.sys_status_label)
        status_group.setLayout(status_layout)

        right_sidebar_layout.addWidget(stat_group, stretch=2)
        right_sidebar_layout.addWidget(warn_group, stretch=4)
        right_sidebar_layout.addWidget(param_group, stretch=2)
        right_sidebar_layout.addWidget(status_group, stretch=1)

        main_central_widget = QWidget()
        global_hbox_layout = QHBoxLayout()
        global_hbox_layout.addLayout(left_vertical_layout, stretch=3)
        global_hbox_layout.addLayout(right_sidebar_layout, stretch=1)
        main_central_widget.setLayout(global_hbox_layout)
        self.setCentralWidget(main_central_widget)

        self.open_img_btn.clicked.connect(self.action_process_single_image)
        self.open_video_btn.clicked.connect(self.action_process_video_stream)
        self.open_cam_btn.clicked.connect(self.action_process_webcam_stream)
        self.stop_btn.clicked.connect(self.action_terminate_detection)
        self.snapshot_btn.clicked.connect(self.action_save_snapshot)
        self.conf_slider.valueChanged.connect(self.action_update_slider_metric)

    def auto_load_model(self):
        if not os.path.exists(self.model_path):
            self.sys_status_label.setText("权重未找到")
            self.sys_status_label.setStyleSheet("color: #E53935;")
            self.view_label.setText(f"未找到预设路径下的权重文件best.pt，请检查预设物理路径。")
            return
        try:
            # 这里改成用本地ultralytics加载，不需要下载
            self.model = YOLO(self.model_path)
            self.sys_status_label.setText("系统就绪")
            self.sys_status_label.setStyleSheet("color: #4CAF50;")
            self.view_label.setText("YOLO-EFS 核心网络初始化成功，功能面板已全面整合。")
            self.warn_text_box.setPlainText("加载完毕。置信度动态调谐机制已接管模型内核。")
            self.open_img_btn.setEnabled(True)
            self.open_video_btn.setEnabled(True)
            self.open_cam_btn.setEnabled(True)
        except Exception as err:
            self.sys_status_label.setText("初始化失败")
            self.sys_status_label.setStyleSheet("color: #E53935;")
            self.warn_text_box.setPlainText(f"模型解析产生未知冲突，报错回溯: {str(err)}")

    def action_update_slider_metric(self):
        slider_val = self.conf_slider.value()
        self.conf_num_indicator.setText(f"{slider_val}%")

    def get_current_slider_conf_float(self):
        return self.conf_slider.value() / 100.0

    def action_process_single_image(self):
        if not self.model:
            return
        self.action_terminate_detection()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择待检测图片", "", "图片文件 (*.jpg *.png *.jpeg *.bmp)")
        if file_path:
            try:
                frame = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    self.warn_text_box.setPlainText("图片读取失败，矩阵为空。")
                    return
                self.sys_status_label.setText("正在进行图像识别...")
                QApplication.processEvents()
                conf_threshold = self.get_current_slider_conf_float()
                results = self.model(frame, conf=conf_threshold)
                annotated_frame = frame.copy()
                fire_count = 0
                smoke_count = 0
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf_val = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id]
                    if label == "fire":
                        fire_count += 1
                    elif label == "smoke":
                        smoke_count += 1
                    color = (0, 0, 255) if label == "fire" else (255, 255, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)
                    cv2.putText(annotated_frame, f"{label} {conf_val:.2f}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                self.render_image_to_label(annotated_frame)
                self.update_summary_board(fire_count, smoke_count)
                self.sys_status_label.setText("图像检测完成")
                self.sys_status_label.setStyleSheet("color: #4CAF50;")
                self.snapshot_btn.setEnabled(True)
            except Exception as e:
                self.warn_text_box.setPlainText(f"静态图片识别出错: {str(e)}")

    def action_process_video_stream(self):
        if not self.model:
            return
        self.action_terminate_detection()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择输入视频文件", "", "视频文件 (*.mp4 *.avi *.mkv)")
        if file_path:
            self.start_thread_engine(file_path)

    def action_process_webcam_stream(self):
        if not self.model:
            return
        self.action_terminate_detection()
        self.start_thread_engine(0)

    def start_thread_engine(self, stream_source):
        self.sys_status_label.setText("动态流监测中...")
        self.sys_status_label.setStyleSheet("color: #FF9800;")
        self.stop_btn.setEnabled(True)
        self.snapshot_btn.setEnabled(True)
        self.video_thread = VideoProcessingThread(stream_source, self.model, self.get_current_slider_conf_float)
        self.video_thread.frame_processed.connect(self.callback_receive_video_frame)
        self.video_thread.error_occurred.connect(self.callback_handle_thread_error)
        self.video_thread.start()

    def callback_receive_video_frame(self, q_img, metrics_dict):
        target_width = self.view_label.width()
        target_height = self.view_label.height()
        if target_width <= 30 or target_height <= 30:
            return
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(target_width, target_height, Qt.KeepAspectRatio,
                                                            Qt.SmoothTransformation)
        self.view_label.setPixmap(scaled_pixmap)
        self.update_summary_board(metrics_dict["fire"], metrics_dict["smoke"])
        self.fps_indicator_label.setText(f"当前帧率: {metrics_dict['fps']} FPS")

    def callback_handle_thread_error(self, err_msg):
        self.warn_text_box.setPlainText(err_msg)
        self.sys_status_label.setText("发生错误")
        self.sys_status_label.setStyleSheet("color: #E53935;")

    def action_terminate_detection(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread = None
        self.sys_status_label.setText("监测暂停")
        self.sys_status_label.setStyleSheet("color: #2196F3;")
        self.stop_btn.setEnabled(False)

    def action_save_snapshot(self):
        current_pixmap = self.view_label.pixmap()
        if current_pixmap and not current_pixmap.isNull():
            save_path, _ = QFileDialog.getSaveFileName(self, "导出并存储当前截帧图", "林火检测捕获成果.jpg",
                                                       "JPEG Image (*.jpg);;PNG Image (*.png)")
            if save_path:
                current_pixmap.save(save_path)
                self.warn_text_box.append(f"\n[系统提示] 截图成功保存至: {os.path.basename(save_path)}")

    def render_image_to_label(self, opencv_matrix):
        target_width = self.view_label.width()
        target_height = self.view_label.height()
        if target_width <= 30 or target_height <= 30:
            return
        rgb_img = cv2.cvtColor(opencv_matrix, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_pix = QPixmap.fromImage(q_img).scaled(target_width, target_height, Qt.KeepAspectRatio,
                                                     Qt.SmoothTransformation)
        self.view_label.setPixmap(scaled_pix)

    def update_summary_board(self, fire_num, smoke_num):
        self.fire_num_label.setText(f"火焰检测： 已检测到，数量 {fire_num}")
        self.smoke_num_label.setText(f"烟雾检测： 已检测到，数量 {smoke_num}")
        self.total_num_label.setText(f"目标总数： {fire_num + smoke_num}")
        if fire_num > 0 or smoke_num > 0:
            warn_msg = (
                f"【核心高危预警触发】\n"
                f"系统检测到疑似林火早期目标！\n"
                f"当前画面内包含特征如下：\n"
                f" - 早期烟雾特征群: {smoke_num} 处\n"
                f" - 初期火焰露头点: {fire_num} 处\n\n"
                f"物理视场内存在明显的早期火情风险。请林防相关人员及时核查现场状况，并采取处置措施！"
            )
            self.warn_text_box.setPlainText(warn_msg)
            self.warn_text_box.setStyleSheet(
                "background-color: #FFEBEE; font-size: 12px; border: 1px solid #E53935; color: #B71C1C;")
        else:
            self.warn_text_box.setPlainText("当前视场内特征稳定，未提取到早期弥散烟雾与红橙色调火焰，状态保持就绪。")
            self.warn_text_box.setStyleSheet(
                "background-color: #FFFFFF; font-size: 12px; border: 1px solid #DCDCDC; color: #000000;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOEFSGuiWindow()
    window.show()
    sys.exit(app.exec_())