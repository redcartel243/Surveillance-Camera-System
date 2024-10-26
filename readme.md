# Surveillance Camera System with Advanced Face Recognition and Real-Time Processing

This project is a sophisticated surveillance camera system featuring high-performance face recognition, real-time video processing, and advanced GPU-optimized detection and recognition capabilities. Designed with modular, asynchronous processing threads, it efficiently handles multiple camera streams, dynamic caching, and advanced event-triggered mechanisms, making it ideal for large-scale surveillance and high-security applications.

## Features

### Core Functionality

- **Multi-Camera Support**: Integrates with both IP and local cameras, handling multiple streams simultaneously.
- **YOLO-Based Face Detection**: Uses the YOLO model for face detection, optimized for GPU processing.
- **Face Recognition**: Identifies faces from a database, caches known faces with Redis for faster recognition, and leverages Faiss indexing for efficient lookups.
- **Real-Time Video Handling**: Dynamically adjusts video resolution based on system load, providing an adaptable solution that optimizes for both performance and clarity.

### Advanced Features

- **Parallel Processing Pipeline**: Each stage of video processing (decoding, detection, recognition, and GUI updating) runs in independent threads, allowing for smooth and asynchronous operations.
- **High-Performance Caching Mechanism**: Redis integration ensures frequently encountered faces are recognized quickly, and Faiss clustering further optimizes recognition by organizing "hot" faces.
- **Event Triggering and Alerts**: System triggers notifications based on specific events, like detecting unknown faces or unusual motion patterns.

## Planned Upgrades

This project is under active development, with additional features and improvements planned:

### Future Milestones

- **Parallel Processing & GPU Utilization (Expected Completion: Q1 2024)**
  - Enhanced GPU utilization across detection and recognition stages.
  - Batch processing support for face recognition, enabling efficient processing across multiple frames.

- **Advanced Caching Mechanism (Expected Completion: Q2 2024)**
  - Integration of Redis for distributed caching, shared across instances.
  - Faiss clustering with a focused "hot" index, improving recognition times for frequently encountered faces.

- **Enhanced Real-Time Video Handling (Expected Completion: Q2 2024)**
  - Intelligent frame skipping based on inactivity or stable scenes.
  - Dynamic resolution adjustments to balance performance and image clarity under varying system loads.

- **AI-Based Event Triggering & Anomaly Detection (Expected Completion: Q3 2024)**
  - Hooks for event-based detection, enabling responses to unknown faces and unusual motion patterns.
  - Anomaly detection to alert users of unexpected activity within restricted areas.

- **Automated Data Labeling & Model Updating (Expected Completion: Q4 2024)**
  - Feedback loop to improve face recognition by retraining models on confirmed corrections.
  - Automated updates to YOLO and face recognition models, seamlessly improving accuracy.

- **Comprehensive Monitoring Dashboard (Expected Completion: Q4 2024)**
  - Real-time dashboard for tracking system health, resource usage, and event logs.
  - Structured analytics on detection accuracy, system performance, and incident tracking.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/surveillance-camera-system.git
   cd surveillance-camera-system
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Redis**:

   Ensure Redis is installed and running on `localhost:6379`. Modify the Redis connection if needed in the configuration.

4. **Prepare YOLO & Face Recognition Models**:

   - Download the YOLO weights (`yolov8n.pt`) and add them to the project directory.
   - Add known face images in `datasets/known_faces` directory for face recognition training.

5. **Database Initialization**:

   Initialize the SQLite database using `DatabaseInitializer` class.

6. **Run the application**:

   ```bash
   python main.py
   ```

## Usage

- **Adding Cameras**: Configure IP/local cameras via the GUI. The system supports multi-camera setups, with dynamic switching available through the interface.
- **Real-Time Monitoring**: Access face recognition, event logging, and camera controls in real-time via the GUI.
- **Event Triggers**: Alerts for unknown faces and unusual movements can be configured through the settings.

## Project Structure

- `src/`: Contains all core modules, including detection, recognition, caching, and database handling.
- `GUI/`: Manages the graphical user interface components.
- `datasets/`: Stores images of known faces for face recognition.
- `models/`: Holds pretrained YOLO and face recognition models.
- `tests/`: Unit and integration tests for key functionalities.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -am 'Add a new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## License

This project is licensed under the MIT License.

---

This README provides a complete guide for users and developers to understand, install, and use the project, along with future upgrade plans to demonstrate ongoing development. Let me know if there are any further details you'd like to add.
