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






@pyqtSlot(np.ndarray, int)
def process_frame(self, frame, label_index):
    """Process frame for face recognition."""
    if self.running:
        self.frame_counter += 1
        face_locations, face_names = self.recognize_faces_with_cache(frame)
        processed_frame = self.face_recognition_processor.draw_faces(frame, face_locations, face_names)
        # Emit the processed frame for UI updates
        self.frame_processed.emit(processed_frame, label_index)