import cv2
import numpy as np
import logging
import sys
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QLabel, QWidget, QGridLayout, \
    QDialog, QSizePolicy, QFileDialog, QMenu, QAction, QInputDialog, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QIcon, QColor, QPainter, QFont
from src.CaptureIpCameraFramesWorker import CaptureIpCameraFramesWorker
from GUI.SurveillanceCameraGUI import Ui_MainWindow
from src.ip_address_dialog import IPAddressDialog
from src.face_recognition_service import FaceRecognitionService, FaceRecognitionWorker, DetectionWorker
from src.face_caching import FaceCache
from src import db_func
from src.face_database import FaceDatabase
import ipaddress
from urllib.parse import urlparse
from src.db_func import Room, Camera, User, DatabaseInitializer
import os
import sqlite3
from src.device import list_capture_devices


# Global flags
is_partial_expanded = False
full_screen_active = False

class FullScreenWindow(QMainWindow):
    def __init__(self, parent=None, label_indices=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
        self.showMaximized()
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QGridLayout(self.central_widget)
        self.labels = {}

        if label_indices is not None:
            for i, idx in enumerate(label_indices):
                label = QLabel(self)
                label.setScaledContents(True)
                label.setAlignment(Qt.AlignCenter)
                label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                row = i // 2
                col = i % 2
                self.layout.addWidget(label, row, col)
                self.labels[idx] = label

        # Make the video labels expand to fill the window
        for i in range(self.layout.rowCount()):
            self.layout.setRowStretch(i, 1)
        for i in range(self.layout.columnCount()):
            self.layout.setColumnStretch(i, 1)

    def closeEvent(self, event):
        global full_screen_active
        parent = self.parent()
        if parent:
            full_screen_active = False
        event.accept()

class VideoLabel(QWidget):
    def __init__(self, index, main_window, parent=None):
        super().__init__(parent)
        self.index = index
        self.main_window = main_window
        self.label = QLabel(self)
        self.label.setScaledContents(True)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create the icon button with a white background
        self.icon_button = QPushButton(self)
        self.icon_button.setIcon(QIcon("GUI/resources/image_2024_07_08T14_12_00_872Z.png"))
        self.icon_button.setFixedSize(24, 24)
        self.icon_button.setStyleSheet("background-color: white; padding: 2px; border: none;")
        self.icon_button.clicked.connect(self.icon_clicked)

        # Layout for positioning
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)

        # Position the button in the bottom-left corner over the label
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.icon_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        layout.setContentsMargins(0, 0, 0, 0)

        self.is_expanded = False

    def icon_clicked(self):
        print(f"Icon clicked on label {self.index}")
        self.main_window.enter_partial_expand(self.index)

    def adjust_for_expansion(self, is_expanded):
        """Adjust info size and position based on expansion state."""
        self.is_expanded = is_expanded

class MethodMapping(QMainWindow, Ui_MainWindow):
    frame_updated = pyqtSignal(np.ndarray, int)  # Emit numpy.ndarray and label index (int)

    def __init__(self, title="", user_id=None):
        try:
            super().__init__()
            self.full_screen_window = None
            self.title = title
            self.user_id = user_id
            self.available_cameras = []
            self.context_actions = ['Change Camera', 'Change Mapping', 'Show', 'Properties', 'Turn Off']
            self.selected_camera_id = None
            self.ip_camera_threads = {}
            self.face_recognition_threads = {}  # Dictionary to store face recognition threads
            self.ip_cameras = []
            self.placeholder_image = QPixmap("GUI/resources/Black Image.png")
            self.view_camera_ids = []
            self.current_page = 0
            self.max_cameras_per_page = 4
            self.video_labels = []
            self.caps = {}
            self.current_camera_ids = {}
            self.timers = {}
            self.label_valid_flags = {}
            self.use_face_recognition = False  # Initialize face recognition flag
            
            # Initialize database classes before using them
            self.room_db = Room()
            self.camera_db = Camera()
            self.user_db = User()

            # Initialize face recognition components
            self.face_cache = FaceCache(max_size=100)
            self.face_db = FaceDatabase()

            # Define data paths using os.path.join
            self.known_faces_dir = os.path.join("datasets", "known_faces")
            self.captures_dir = os.path.join("datasets", "Captures")

            # Ensure directories exist
            os.makedirs(self.known_faces_dir, exist_ok=True)
            os.makedirs(self.captures_dir, exist_ok=True)

            # Debug the path values
            print(f"known_faces_dir: {self.known_faces_dir}, type: {type(self.known_faces_dir)}")
            print(f"captures_dir: {self.captures_dir}, type: {type(self.captures_dir)}")

            self.detection_worker = DetectionWorker(skip_frames=10)
            # Initialize FaceRecognitionWorker with keyword arguments
            self.face_recognition_worker = FaceRecognitionWorker(
                cache=self.face_cache,
                database=self.face_db,
                known_faces_dir=self.known_faces_dir,
                tolerance=0.6
            )
            # Connect DetectionWorker output to FaceRecognitionWorker
            self.detection_worker.face_locations_signal.connect(self.face_recognition_worker.recognize_faces)
            self.face_recognition_worker.frame_processed.connect(self.on_frame_processed)

            self.detection_worker.start_detection()
            self.face_recognition_worker.start_recognition()

            self.setupUi()
            print("MethodMapping initialized")

            self.frame_updated.connect(self.on_frame_updated)

            # Now it's safe to use methods that rely on the initialized attributes
            self.populate_rooms_combobox()
            self.rooms_list_combobox.currentIndexChanged.connect(self.show_combobox_context_menu)
            self.populate_mapping_list_and_camera_view()

            # New additions for overlay
            self.overlay_cache = {}
            self.overlay_update_timer = QTimer(self)
            self.overlay_update_timer.timeout.connect(self.update_overlay_cache)
            self.overlay_update_timer.start(5000)  # Update every 5 seconds

        except Exception as e:
            print(f"Exception during initialization: {e}")
            logging.exception("Exception during initialization")

    def setupUi(self):
        super().setupUi(self)
        self.setWindowTitle(self.title)

        self.vision_button.clicked.connect(self.toggle_face_recognition)  # Ensure this connection is set
        self.refresh_button.clicked.connect(self.refreshbutton)
        self.edit_mapping.clicked.connect(self.open_mapping_tab)
        self.add_room_button.clicked.connect(self.add_room)
        self.change_map_button.clicked.connect(self.change_map)
        self.all_camera_off_button.clicked.connect(self.stop_all_threads)
        self.expand_all_button.clicked.connect(self.toggle_expand_all)
        self.tabWidget.currentChanged.connect(self.resize_based_on_tab)
        self.add_camera_button.clicked.connect(self.add_ip_camera)

        self.next_button.clicked.connect(self.next_page)
        self.previous_button.clicked.connect(self.previous_page)

        # Initialize video display layout dynamically
        self.video_widget_container = QWidget(self)
        self.video_layout = QGridLayout(self.video_widget_container)
        self.gridLayout.addWidget(self.video_widget_container, 0, 0, 1, 1)

        # Initialize video labels and add to layout
        for i in range(self.max_cameras_per_page):
            video_label = VideoLabel(i, self)
            video_label.label.setPixmap(self.placeholder_image)
            self.video_layout.addWidget(video_label, i // 2, i % 2)
            self.video_labels.append(video_label)
            self.label_valid_flags[i] = False

        self.show_placeholder_image()

    def resize_based_on_tab(self, index):
        if index == self.tabWidget.indexOf(self.alarm_tab):
            self.setFixedSize(600, 600)
        elif index == self.tabWidget.indexOf(self.mapping_tab):
            self.setFixedSize(646, 618)
        elif index == self.tabWidget.indexOf(self.camera_tab):
            self.setFixedSize(987, 607)

    def next_page(self):
        if (self.current_page + 1) * self.max_cameras_per_page < len(self.view_camera_ids):
            self.current_page += 1
            self.update_video_display()

    def previous_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_video_display()

    def update_video_display(self):
        try:
            self.stop_all_threads()

            while self.video_layout.count():
                item = self.video_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    self.video_layout.removeWidget(widget)

            self.label_valid_flags = {}

            start_index = self.current_page * self.max_cameras_per_page
            end_index = min(start_index + self.max_cameras_per_page, len(self.view_camera_ids))
            current_cameras = self.view_camera_ids[start_index:end_index]

            for i in range(self.max_cameras_per_page):
                if i < len(current_cameras):
                    camera_id = current_cameras[i]
                    video_label = self.video_labels[i]
                    self.video_layout.addWidget(video_label, i // 2, i % 2)
                    self.label_valid_flags[i] = True

                    self.turn_on_camera(camera_id, i)
                else:
                    video_label = self.video_labels[i]
                    self.video_layout.addWidget(video_label, i // 2, i % 2)
                    video_label.label.setPixmap(self.placeholder_image)
                    self.label_valid_flags[i] = False

            total_pages = (len(self.view_camera_ids) - 1) // self.max_cameras_per_page
            self.next_button.setEnabled(self.current_page < total_pages)
            self.previous_button.setEnabled(self.current_page > 0)
        except Exception as e:
            print(f"Exception in update_video_display: {e}")

    def toggle_expand_all(self):
        global full_screen_active 
        if full_screen_active:
            self.full_screen_window.close()
            full_screen_active = False
        else:
            self.full_screen_window = FullScreenWindow(self, label_indices=[0, 1, 2, 3])
            self.full_screen_window.show()
            full_screen_active = True

    def enter_partial_expand(self, index):
        global is_partial_expanded
        if is_partial_expanded:
            for video_label in self.video_labels:
                video_label.setVisible(True)
                video_label.adjust_for_expansion(False)
            is_partial_expanded = False
        else:
            for i, video_label in enumerate(self.video_labels):
                if i != index:
                    video_label.setVisible(False)
                video_label.adjust_for_expansion(i == index)
            is_partial_expanded = True
        self.adjust_layouts()

    def adjust_layouts(self):
        global is_partial_expanded
        if is_partial_expanded:
            for i, video_label in enumerate(self.video_labels):
                if video_label.isVisible():
                    self.video_layout.addWidget(video_label, 0, 0, 1, 1)
        else:
            for i, video_label in enumerate(self.video_labels):
                row = i // 2
                col = i % 2
                self.video_layout.addWidget(video_label, row, col)

    @pyqtSlot(np.ndarray, int)
    def on_frame_updated(self, frame, label_index):
        """Update the video label with the new frame, with optional face recognition processing."""
        try:
            if self.use_face_recognition:
                # Send the frame to the detection worker to identify face locations
                self.detection_worker.process_frame(frame, label_index)
            else:
                # Directly update the label with the frame if face recognition is off
                self.update_frame_in_label(frame, label_index)
        except Exception as e:
            print(f"Exception in on_frame_updated: {e}")
            logging.exception("Exception in on_frame_updated")


    def turn_on_camera(self, camera_id, label_index):
        try:
            print(f"Attempting to turn on camera {camera_id} for label {label_index}")

            self.stop_camera_feed(label_index)
            self.current_camera_ids[label_index] = camera_id
            video_label = self.video_labels[label_index].label
            video_label.setVisible(True)

            if camera_id:
                if self.is_valid_ip_address(camera_id):
                    print(f"Trying to connect to IP camera at {camera_id}")
                    # Start the IP camera feed
                    ip_thread = CaptureIpCameraFramesWorker(camera_id, label_index)
                    ip_thread.ImageUpdated.connect(self.on_frame_updated)
                    self.ip_camera_threads[label_index] = ip_thread
                    ip_thread.start()
                    print(f"Connected to IP camera at {camera_id}")

                    # Enable face recognition for this camera if needed
                    if self.use_face_recognition:
                        self.face_recognition_worker.start_camera(label_index)
                else:
                    # Handle local cameras with cv2.VideoCapture
                    cap = cv2.VideoCapture(int(camera_id))
                    if cap.isOpened():
                        self.caps[label_index] = cap
                        timer = QTimer()
                        timer.timeout.connect(lambda cp=cap, idx=label_index: self.capture_frame(cp, idx))
                        timer.start(30)
                        self.timers[label_index] = timer
                        
                        # Enable face recognition for this camera if needed
                        if self.use_face_recognition:
                            self.face_recognition_worker.start_camera(label_index)
                    else:
                        print(f"Failed to open camera {camera_id}")
        except Exception as e:
            print(f"Exception in turn_on_camera: {e}")

    def capture_frame(self, cap, label_index):
        try:
            ret, frame = cap.read()
            if ret:
                if self.use_face_recognition:
                    # Send frame to face recognition worker
                    self.detection_worker.process_frame(frame, label_index)
                else:
                    # Display the frame directly
                    self.update_image(frame, self.video_labels[label_index].label)
            else:
                print(f"Failed to capture frame from camera {self.current_camera_ids[label_index]} at label {label_index}")
        except Exception as e:
            print(f"Exception in capture_frame: {e}")


    def stop_all_threads(self):
        try:
            # Stop all face recognition threads
            self.stop_face_recognition_threads()

            # Stop all camera feeds
            for label_index in range(self.max_cameras_per_page):
                self.stop_camera_feed(label_index)

            self.label_valid_flags = {}
            self.current_camera_ids = {}
        except Exception as e:
            print(f"Exception in stop_all_threads: {e}")

    def stop_camera_feed(self, label_index):
        """Stop the camera feed and associated face recognition."""
        if label_index in self.caps:
            cap = self.caps.pop(label_index)
            cap.release()

        if label_index in self.timers:
            timer = self.timers.pop(label_index)
            timer.stop()

        if label_index in self.ip_camera_threads:
            thread = self.ip_camera_threads.pop(label_index)
            thread.stop()
            thread.wait()

        # Stop face recognition for this camera
        if self.use_face_recognition:
            self.face_recognition_worker.stop_camera(label_index)

        if label_index in self.current_camera_ids:
            del self.current_camera_ids[label_index]

    def show_placeholder_image(self):
        """Show placeholder image on all labels."""
        for label_index in range(self.max_cameras_per_page):
            self.video_labels[label_index].label.setPixmap(self.placeholder_image)
            self.label_valid_flags[label_index] = False  # No active feed
            self.stop_camera_feed(label_index)

    def populate_mapping_list_and_camera_view(self):
        self.mapping_list.clear()
        rooms_with_cameras = self.get_all_rooms_with_cameras()
        
        available_cameras = self.camera_db.get_available_cameras()
        all_cameras = set()
        for cameras in rooms_with_cameras.values():
            all_cameras.update(cameras)
        all_cameras.update(available_cameras)
        all_cameras.update(self.ip_cameras)

        sorted_cameras = sorted(all_cameras, key=lambda x: str(x))

        for room_name, cameras in rooms_with_cameras.items():
            for camera in cameras:
                list_item_text = f"{room_name}: Camera {camera}"
                self.mapping_list.addItem(list_item_text)

        self.view_camera_ids = sorted_cameras
        self.update_video_display()

    def populate_rooms_combobox(self):
        self.rooms_list_combobox.clear()
        rooms_with_cameras = self.get_all_rooms_with_cameras()
        for room_name, cameras in rooms_with_cameras.items():
            camera_list = ', '.join(cameras)
            display_text = f"{room_name}: {camera_list}"
            self.rooms_list_combobox.addItem(display_text)

    def get_all_rooms_with_cameras(self):
        rooms = self.room_db.get_rooms_by_user_id(self.user_id)
        rooms_with_cameras = {}
        for room in rooms:
            room_name = room[0]
            cameras = self.camera_db.get_cameras_by_room_name(room_name)
            rooms_with_cameras[room_name] = cameras
        return rooms_with_cameras

    def toggle_face_recognition(self):
        try:
            if self.use_face_recognition:
                # Stop both detection and recognition
                self.detection_worker.stop_detection()
                self.face_recognition_worker.stop_recognition()
                self.use_face_recognition = False
                print("Face recognition stopped.")
            else:
                # Start both detection and recognition
                self.detection_worker.start_detection()
                self.face_recognition_worker.start_recognition()
                self.use_face_recognition = True
                print("Face recognition started.")
        except Exception as e:
            print(f"Exception in toggle_face_recognition: {e}")


    def update_image(self, image, label):
        """Update the given label with the new image."""
        try:
            # Convert the OpenCV image (numpy.ndarray) to QImage
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Set the QImage to the label
            pixmap = QPixmap.fromImage(q_image)
            label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            print(f"Exception in update_image: {e}")

    def handle_face_recognition(self, face_locations, face_names):
        try:
            print(f"Faces recognized: {face_names}")
            # Implement additional logic if needed
        except Exception as e:
            print(f"Exception in handle_face_recognition: {e}")
            logging.exception("Exception in handle_face_recognition")

    def change_map(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Map Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            map_image = QPixmap(file_path)
            if not map_image.isNull():
                scaled_image = map_image.scaled(self.map_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.map_display.setPixmap(scaled_image)
            else:
                print("Failed to load the image. Check the file format and path.")
        else:
            print("No file selected.")

    def refreshbutton(self):
        # Reload known faces
        self.face_recognition_worker.processor.load_known_faces(self.known_faces_dir)
        
        new_camera_count = self.add_new_cameras()
        self.populate_mapping_list_and_camera_view()
        self.show_message(f"Loading cameras finished. {new_camera_count} new cameras added and known faces reloaded.")


    def show_combobox_context_menu(self, index):
        if index < 0:
            return  # No item selected

        room_text = self.rooms_list_combobox.itemText(index)
        room_name = room_text.split(': ')[0]

        contextMenu = QMenu(self)

        delete_room_action = QAction("Delete Room", self)
        delete_camera_action = QAction("Delete Camera Assignment", self)
        add_camera_action = QAction("Add Camera", self)

        contextMenu.addAction(delete_room_action)
        contextMenu.addAction(delete_camera_action)
        contextMenu.addAction(add_camera_action)

        delete_room_action.triggered.connect(lambda: self.delete_room(room_name))
        delete_camera_action.triggered.connect(lambda: self.delete_assignment(room_name))
        add_camera_action.triggered.connect(lambda: self.add_camera_to_room(room_name))

        contextMenu.exec_(self.rooms_list_combobox.mapToGlobal(self.rooms_list_combobox.rect().bottomLeft()))

    def add_room(self):
        room_name, ok = QInputDialog.getText(self, "Add Room", "Enter room name:")
        if ok and room_name:
            rooms = self.room_db.get_rooms_by_user_id(self.user_id)
            if any(room_name == existing_name[0] for existing_name in rooms):
                self.show_message(f"Room '{room_name}' already exists.")
            else:
                self.room_db.add_room(self.user_id, room_name)
                self.populate_rooms_combobox()
                self.show_message(f"Room '{room_name}' added successfully.")

    def delete_room(self, room_name):
        try:
            self.room_db.delete_room(room_name)
            self.populate_rooms_combobox()
            self.show_message(f"Room '{room_name}' and its camera assignments deleted.")
        except Exception as e:
            self.show_message(f"Failed to delete room '{room_name}': {str(e)}")

    def delete_assignment(self, room_name):
        """
        Display a combobox to select and delete a camera from the room.
        """
        try:
            cameras = self.camera_db.get_cameras_by_room_name(room_name)
            if not cameras:
                self.show_message(f"No cameras found in room '{room_name}'.")
                return

            camera_id, ok = QInputDialog.getItem(self, "Delete Camera", "Select camera to delete:", cameras, 0, False)
            if ok and camera_id:
                self.remove_camera_from_room(camera_id)
        except Exception as e:
            self.show_message(f"Failed to delete camera from room '{room_name}': {str(e)}")

    def assign_camera_to_room(self, room_name, camera_id):
        try:
            self.camera_db.assign_camera_to_room(room_name, camera_id)
            self.populate_rooms_combobox()
            self.show_message(f"Camera {camera_id} assigned to room '{room_name}' successfully.")
        except Exception as e:
            self.show_message(f"Failed to assign camera {camera_id}: {str(e)}")

    def open_mapping_tab(self):
        self.tabWidget.setCurrentIndex(self.tabWidget.indexOf(self.mapping_tab))

    def show_message(self, message):
        QMessageBox.information(self, "Information", message)
    
    def is_valid_ip_address(self, address):
        try:
            parsed_url = urlparse(address)
            hostname = parsed_url.hostname
            if hostname:
                ipaddress.ip_address(hostname)
                return True
        except ValueError:
            pass
        return False

    def add_ip_camera(self):
        address, ok = QInputDialog.getText(self, "Add IP Camera", "Enter IP camera address (e.g., https://192.168.0.101:8080/video):")
        if not ok or not address:
            return

        if not self.is_valid_ip_address(address):
            self.show_message("Invalid IP camera address format.")
            return

        try:
            self.camera_db.add_new_camera(address)
            self.populate_mapping_list_and_camera_view()
            self.show_message(f"IP camera {address} added successfully.")
        except Exception as e:
            self.show_message(f"Failed to add IP camera: {str(e)}")
            
    def remove_camera_from_room(self, camera_id):
        try:
            self.camera_db.unassign_camera(camera_id)
            self.populate_rooms_combobox()
            self.show_message(f"Camera {camera_id} unassigned successfully.")
        except Exception as e:
            self.show_message(f"Failed to unassign camera {camera_id}': {str(e)}")

    def add_camera_to_room(self, room_name):
        """
        Add a camera to the specified room by selecting from unassigned cameras.
        """
        try:
            # Fetch unassigned cameras from the database
            unassigned_cameras = self.camera_db.get_unassigned_cameras()
            
            if not unassigned_cameras:
                self.show_message("No unassigned cameras available.")
                return

            # Display a combobox to select an unassigned camera
            camera_id, ok = QInputDialog.getItem(self, "Add Camera", "Select camera to add:", unassigned_cameras, 0, False)
            if ok and camera_id:
                self.assign_camera_to_room(room_name, camera_id)
        except Exception as e:
            self.show_message(f"Failed to add camera to room '{room_name}': {str(e)}")

    def update_overlay_cache(self):
        for label_index, camera_id in self.current_camera_ids.items():
            if camera_id:
                room_name = self.camera_db.get_room_name_by_camera_id(camera_id)
                text = f"Camera: {camera_id} | Room: {room_name if room_name else 'Unassigned'}"
                self.overlay_cache[label_index] = self.create_overlay_image(text)

    def create_overlay_image(self, text):
        # Adjust the size of the image to accommodate larger text
        image = QImage(800, 50, QImage.Format_ARGB32)
        image.fill(QColor(0, 0, 0, 128))
        painter = QPainter(image)
        painter.setPen(Qt.white)
        
        # Set a larger font size
        painter.setFont(QFont("Arial", 14))  # Increase the font size here
        
        # Draw the text
        painter.drawText(image.rect(), Qt.AlignLeft | Qt.AlignVCenter, text)
        painter.end()
        return image

    @pyqtSlot(np.ndarray, int)
    def on_frame_processed(self, processed_frame, label_index):
        """Handle the frame processed by face recognition."""
        self.update_frame_in_label(processed_frame, label_index)

    def update_frame_in_label(self, frame, label_index):
        global full_screen_active
        """Update the given label with the new frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_frame.shape
        bytes_per_line = channel * width
        q_image = QImage(rgb_frame.data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        
        video_label = self.video_labels[label_index].label
        pixmap = QPixmap.fromImage(q_image)
        
        # Overlay handling (if you have any overlay information to add)
        if label_index in self.overlay_cache:
            overlay = self.overlay_cache[label_index]
            painter = QPainter(pixmap)
            painter.drawImage(10, pixmap.height() - overlay.height() - 10, overlay)
            painter.end()

        # Set the frame to the label
        video_label.setPixmap(pixmap.scaled(video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.label_valid_flags[label_index] = True

        # Handle full-screen mode if applicable
        if full_screen_active and self.full_screen_window:
            fs_label = self.full_screen_window.labels.get(label_index)
            if fs_label:
                fs_label.setPixmap(pixmap.scaled(fs_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


    def closeEvent(self, event):
        # Stop detection and recognition workers
        self.detection_worker.stop_detection()
        self.face_recognition_worker.stop_recognition()

        # Stop threads and resources
        self.face_recognition_worker.thread().quit()
        self.detection_worker.quit()

        event.accept()


    def add_new_cameras(self):
        available_cameras = list_capture_devices()
        available_cameras = [str(camera) for camera in available_cameras]

        existing_cameras = self.camera_db.get_available_cameras()
        new_cameras = [camera for camera in available_cameras if camera not in existing_cameras]

        for camera_id in new_cameras:
            try:
                self.camera_db.add_new_camera(camera_id)
            except sqlite3.IntegrityError as e:
                print(f"Failed to add camera {camera_id}: {e}")

        return len(new_cameras)

    def stop_face_recognition_threads(self):
        """Stop all face recognition threads."""
        for thread in self.face_recognition_threads.values():
            thread.stop()
            thread.wait()
        self.face_recognition_threads.clear()

if __name__ == "__main__":
    from GUI.LoginGUI import LoginWindow

    db_initializer = DatabaseInitializer()
    db_initializer.init_db()

    app = QApplication(sys.argv)
    login_window = LoginWindow()

    if login_window.exec_() == QDialog.Accepted:
        user_id = login_window.get_user_id()
        ui = MethodMapping("Surveillance Camera", user_id=user_id)
        ui.show()

    sys.exit(app.exec_())