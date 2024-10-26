import cv2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def list_capture_devices(max_devices=10):
    # List to hold available capture devices
    available_devices = []

    # Try to open sequential device IDs
    for device_id in range(max_devices):
        cap = cv2.VideoCapture(device_id)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            # Check if FPS is greater than 0 to determine if it is a usable camera
            if fps > 0:
                available_devices.append(device_id)
            else:
                logging.warning(f"Device {device_id} has FPS {fps}, likely not a usable camera.")
            cap.release()
        else:
            logging.warning(f"Device {device_id} could not be opened.")

    return available_devices

def get_device_info(device_id):
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        logging.error(f"Failed to open device {device_id}")
        return None

    # Get some basic properties
    info = {
        "Device ID": device_id,
        "Frame Width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "Frame Height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "FPS": cap.get(cv2.CAP_PROP_FPS)
    }
    cap.release()
    return info

"""def main():
    devices = list_capture_devices()
    if not devices:
        logging.info("No capture devices found.")
        return

    logging.info(f"Found {len(devices)} usable capture device(s):")
    for device_id in devices:
        info = get_device_info(device_id)
        if info:
            logging.info(f"\nDevice ID: {device_id}")
            for key, value in info.items():
                logging.info(f"  {key}: {value}")
        else:
            logging.warning(f"Failed to retrieve info for device {device_id}")
if __name__ == "__main__":
    main()"""
