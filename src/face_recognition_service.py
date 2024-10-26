import os
import beepy
import cv2
import numpy as np
import face_recognition
import time
from queue import Queue
from ultralytics import YOLO
from PyQt5.QtCore import pyqtSignal, QObject, QThread, pyqtSlot
import logging
from src import Data
from src.face_caching import FaceCache
import torch
import faiss
import numpy as np

logging.basicConfig(level=logging.INFO)



class FaissIndex:
    def __init__(self, dimension, min_points=50):
        self.index = None
        self.dimension = dimension
        self.min_points = min_points
        self.names = []  # Add this line to store names

    def initialize_index(self, encodings):
        if len(encodings) >= self.min_points:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(np.array(encodings).astype('float32'))
        else:
            print(f"Insufficient data: {len(encodings)} points found, {self.min_points} required for clustering.")

    def add(self, encodings, labels):
        if self.index is None:
            self.initialize_index(encodings)
        else:
            self.index.add(np.array(encodings).astype('float32'))
        self.names.extend(labels)  # Store the names
    
    def search(self, encoding, k=1):
        encoding = np.array([encoding], dtype=np.float32)
        _, indices = self.index.search(encoding, k)
        return [self.names[i] if i < len(self.names) else "Unknown" for i in indices[0]]



class FaceRecognitionService(QThread):
    ImageUpdated = pyqtSignal(np.ndarray, int)

    def __init__(self, camera_address, label_index, cache, database, known_faces_dir, tolerance=0.6):
        super().__init__()
        self.camera_address = camera_address
        self.label_index = label_index
        self.cache = cache
        self.database = database
        self.recognition_processor = FaceRecognitionProcessor(
            cache=cache,
            database=database,
            known_faces_dir=known_faces_dir,
            tolerance=tolerance,
            skip_frames=5
        )
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.camera_address)
        if not cap.isOpened():
            print(f"Failed to open camera {self.camera_address}")
            return
        while self.running:
            ret, frame = cap.read()
            if ret:
                # Change the order of unpacking to match the return values
                face_locations, face_names, processed_frame = self.recognition_processor.recognize_faces(frame)
                self.ImageUpdated.emit(processed_frame, self.label_index)
        cap.release()

    def stop(self):
        self.running = False


class FaceRecognitionProcessor:
    def __init__(self, cache, database, known_faces_dir, tolerance=0.6, skip_frames=5, inference_size=320):
        self.cache = cache
        self.database = database
        self.tolerance = tolerance
        self.faiss_index = FaissIndex(dimension=128,min_points=50)
        self.known_face_encodings = []
        self.known_face_names = []
        
        # YOLO model with GPU
        self.model = YOLO("yolov8n.pt")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.frame_count = 0
        self.skip_frames = skip_frames
        self.inference_size = inference_size
        self.load_known_faces(known_faces_dir)

    def process_frame(self, frame):
        """Resize frame for YOLO processing."""
        height, width = frame.shape[:2]
        scale_factor = self.inference_size / max(height, width)
        return cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))

    def recognize_faces(self, frame):
        """Detect and recognize faces in the given frame."""
        try:
            # Process the frame through YOLO model
            processed_frame = self.process_frame(frame)
            results = self.model(processed_frame, imgsz=self.inference_size)  # Run YOLO inference

            # Get face locations using YOLO results
            face_locations = self.get_face_locations(results, frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            face_names = []
            for encoding in face_encodings:
                # Check Redis cache first
                cached_name = self.cache.get(encoding)
                if cached_name:
                    face_names.append(cached_name)
                else:
                    # Check if FAISS index is initialized
                    if self.faiss_index.index is not None:
                        # Query Faiss if index exists
                        faiss_name = self.faiss_index.search(encoding)
                        face_names.append(faiss_name[0] if faiss_name else "Unknown")
                        if faiss_name and faiss_name[0] != "Unknown":
                            self.cache.add(encoding, faiss_name[0])  # Cache recognized face
                    else:
                        face_names.append("Unknown")

            # Draw faces on the frame
            processed_frame = self.draw_faces(frame.copy(), face_locations, face_names)
            return face_locations, face_names, processed_frame

        except Exception as e:
            print(f"Exception in recognize_faces: {e}")
            return [], [], frame  # Return original frame if processing fails

    def load_known_faces(self, known_faces_dir, min_points=50):
        known_encodings = []
        known_names = []

        for filename in os.listdir(known_faces_dir):
            filepath = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                encoding = encodings[0]
                name = os.path.splitext(filename)[0]
                
                # Check if face is already in database
                if not self.database.get_known_face(encoding):
                    # Save encoding and name in the database
                    self.database.add_known_face(encoding, name)
                    known_encodings.append(encoding)
                    known_names.append(name)
                    self.cache.add(encoding, name)  # Cache each known face in Redis

        # Check if we have enough encodings to initialize FAISS
        if len(known_encodings) >= min_points:
            self.faiss_index.add(known_encodings, known_names)  # Add to FAISS index
            self.known_face_names = known_names  # Save names for reverse lookup
            print(f"FAISS index initialized with {len(known_encodings)} faces.")
        else:
            print(f"Insufficient encodings for FAISS initialization: {len(known_encodings)} available, {min_points} required.")



    def get_face_locations(self, results, frame):
        """Map YOLO results to face locations."""
        face_locations = []
        height, width = frame.shape[:2]
        scale_x = width / self.inference_size
        scale_y = height / self.inference_size
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                face_locations.append((y1, x2, y2, x1))
        return face_locations

    def get_face_encodings(self, frame, face_locations):
        """Extract face encodings for identified face locations."""
        try:
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            return face_encodings
        except Exception as e:
            print(f"Exception in get_face_encodings: {e}")
            return []

    def draw_faces(self, frame, face_locations, face_names):
        """Draw bounding boxes and labels on the frame with a professional appearance."""
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a red bounding box with a slightly thicker border
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw the filled rectangle for the name label
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            
            # Add the name with a shadow effect
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            font_thickness = 1

            # Shadow effect (draw text slightly offset in black)
            cv2.putText(frame, name, (left + 7, bottom - 7), font, font_scale, (0, 0, 0), font_thickness + 1)
            
            # Main text in white
            cv2.putText(frame, name, (left + 6, bottom - 8), font, font_scale, (255, 255, 255), font_thickness)
        
        return frame


    def lookup_name(self, face_encoding):
        """Retrieve the name for a given face encoding."""
        cached_name = self.cache.get(face_encoding)
        if cached_name:
            return cached_name
        db_name = self.database.get_known_face(face_encoding)
        if db_name:
            self.cache.add(face_encoding, db_name)
            return db_name
        return "Unknown"




class FaceRecognitionWorker(QObject):
    frame_processed = pyqtSignal(np.ndarray, int)

    def __init__(self, cache, database, known_faces_dir='datasets/known_faces', skip_frames=4, tolerance=0.6):
        super().__init__()
        self.running = False
        self.processor = FaceRecognitionProcessor(
            cache=cache,
            database=database,
            known_faces_dir=known_faces_dir,
            tolerance=tolerance
        )

    @pyqtSlot(np.ndarray, list, int)
    def recognize_faces(self, frame, face_locations, label_index):
        """Recognize faces based on detected locations."""
        if not self.running:
            return

        face_encodings = face_recognition.face_encodings(frame, face_locations)
        face_names = [self.processor.lookup_name(encoding) for encoding in face_encodings]
        processed_frame = self.processor.draw_faces(frame, face_locations, face_names)
        self.frame_processed.emit(processed_frame, label_index)

    def start_recognition(self):
        self.running = True

    def stop_recognition(self):
        self.running = False


class DetectionWorker(QThread):
    face_locations_signal = pyqtSignal(np.ndarray, list, int)  # Send frame, face locations, and label index

    def __init__(self, skip_frames=10, parent=None):
        super().__init__(parent)
        self.skip_frames = skip_frames
        self.frame_counter = {}
        self.running = False

    def process_frame(self, frame, label_index):
        """Process frame to detect faces and emit detected locations."""
        if not self.running:
            return

        if label_index not in self.frame_counter:
            self.frame_counter[label_index] = 0

        self.frame_counter[label_index] += 1
        if self.frame_counter[label_index] % self.skip_frames == 0:
            # Run face detection every `skip_frames`
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            face_locations = face_recognition.face_locations(small_frame)
            scaled_locations = [(int(top * 2), int(right * 2), int(bottom * 2), int(left * 2)) 
                                for (top, right, bottom, left) in face_locations]
            self.face_locations_signal.emit(frame, scaled_locations, label_index)

    def start_detection(self):
        self.running = True

    def stop_detection(self):
        self.running = False
        self.frame_counter.clear()
