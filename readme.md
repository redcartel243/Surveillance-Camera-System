# Surveillance Camera System

## Overview

The Surveillance Camera System is a comprehensive application designed for real-time video monitoring and face recognition. Built using Python and PyQt5, this system allows users to manage multiple camera feeds, recognize faces, and interact with a user-friendly graphical interface. The application is suitable for security purposes or managing access control in various environments (Only for personal use).

## Features

- **Multi-Camera Support**: Connect and manage multiple cameras simultaneously.
- **Face Recognition**: Utilize advanced face recognition technology to identify individuals in real-time.
- **User-Friendly Interface**: Intuitive GUI built with PyQt5 for easy navigation and control.
- **Dynamic Video Display**: Automatically updates video feeds based on user interactions.
- **Full-Screen Mode**: View camera feeds in full-screen for enhanced visibility.
- **Camera Management**: Easily add, remove, or modify camera settings.
- **Real-Time Notifications**: Receive alerts for recognized faces or other significant events.

## Installation

### Prerequisites

- Python 3.x
- Required libraries:
  - OpenCV
  - NumPy
  - PyQt5
  - face_recognition
  - beepy
  - Other dependencies as specified in the `requirements.txt` file.

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/redcartel243/surveillance-camera-system.git
   cd surveillance-camera-system
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up the Database**:
   Initialize the database by running the following command:
   ```bash
   python -m src.db_func
   ```

4. **Run the Application**:
   Start the application using:
   ```bash
   python src/SurveillanceCameraGUIMethods.py
   ```

## Usage

1. **Login**: Upon starting the application, you will be prompted to log in. Enter your credentials to access the system.
2. **Camera Management**: Use the interface to add or modify camera settings. You can also view live feeds from connected cameras.
3. **Face Recognition**: Enable face recognition to start identifying individuals in the camera feeds. Recognized faces will be displayed on the screen.
4. **Notifications**: Stay informed with real-time notifications for recognized faces or other events.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.


## Contact

For any inquiries or support, please contact [applitwebsite1@outlook.com.com](mailto:applitwebsite1@outlook.com).

---

Thank you for using the Surveillance Camera System! Feel free to contribute and make it even better!