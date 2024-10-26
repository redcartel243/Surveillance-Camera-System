import sqlite3
import bcrypt
from src.device import list_capture_devices

class BaseDatabase:
    def __init__(self, db_path='CAM_SURV.db'):
        self.db_path = db_path

    def connect(self):
        """Establish a database connection."""
        return sqlite3.connect(self.db_path)

    def execute_query(self, query, params=()):
        """Execute a query and commit changes."""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"Database error in execute_query: {e}")
            raise
        finally:
            conn.close()

    def fetch_one(self, query, params=()):
        """Fetch a single result."""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            result = cursor.fetchone()
            return result
        except Exception as e:
            print(f"Database error in fetch_one: {e}")
            raise
        finally:
            conn.close()

    def fetch_all(self, query, params=()):
        """Fetch all results."""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            result = cursor.fetchall()
            return result
        except Exception as e:
            print(f"Database error in fetch_all: {e}")
            raise
        finally:
            conn.close()

class Room(BaseDatabase):
    def __init__(self):
        super().__init__()

    def add_room(self, user_id, room_name):
        query = 'INSERT INTO rooms (room_name, user_id) VALUES (?, ?)'
        self.execute_query(query, (room_name, user_id))

    def get_rooms_by_user_id(self, user_id):
        query = 'SELECT room_name FROM rooms WHERE user_id = ?'
        result = self.fetch_all(query, (user_id,))
        rooms = [row[0] for row in result]
        return rooms

    def delete_room(self, room_name):
        query = 'DELETE FROM rooms WHERE room_name = ?'
        self.execute_query(query, (room_name,))

    def get_room_name_by_camera_id(self, camera_id):
        query = 'SELECT room_name FROM cameras WHERE camera_id = ?'
        result = self.fetch_one(query, (camera_id,))
        return result[0] if result else None

class Camera(BaseDatabase):
    def __init__(self):
        super().__init__()

    def get_cameras_by_room_name(self, room_name):
        """Retrieve all camera IDs assigned to a specific room."""
        query = 'SELECT camera_id FROM cameras WHERE room_name = ?'
        result = self.fetch_all(query, (room_name,))
        camera_ids = [row[0] for row in result]
        return camera_ids

    def get_available_cameras(self):
        """Retrieve all available cameras (those not assigned to any room)."""
        query = 'SELECT camera_id FROM cameras WHERE room_name IS NULL OR room_name = ""'
        result = self.fetch_all(query)
        camera_ids = [row[0] for row in result]
        return camera_ids

    def assign_camera_to_room(self, room_name, camera_id):
        """Assign a camera to a specific room."""
        query = 'UPDATE cameras SET room_name = ? WHERE camera_id = ?'
        self.execute_query(query, (room_name, camera_id))

    def unassign_camera(self, camera_id):
        """Unassign a camera from its current room."""
        query = 'UPDATE cameras SET room_name = NULL WHERE camera_id = ?'
        self.execute_query(query, (camera_id,))

    def add_new_camera(self, camera_id):
        """Add a new camera to the database."""
        query = 'INSERT INTO cameras (camera_id) VALUES (?)'
        self.execute_query(query, (camera_id,))

    def get_room_name_by_camera_id(self, camera_id):
        """Retrieve the room name associated with a specific camera ID."""
        query = 'SELECT room_name FROM cameras WHERE camera_id = ?'
        result = self.fetch_one(query, (camera_id,))
        return result[0] if result else None

import bcrypt

class User(BaseDatabase):
    def __init__(self):
        super().__init__()

    def store_user(self, username, password):
        hashed_password = self.hash_password(password)
        query = 'INSERT INTO users (username, password) VALUES (?, ?)'
        try:
            self.execute_query(query, (username, hashed_password))
        except sqlite3.IntegrityError:
            raise ValueError("Username already exists.")

    def verify_password(self, username, password):
        query = 'SELECT password FROM users WHERE username = ?'
        stored_password = self.fetch_one(query, (username,))
        if stored_password and bcrypt.checkpw(password.encode('utf-8'), stored_password[0]):
            return True
        return False

    def hash_password(self, password):
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    def get_user(self, username):
        query = 'SELECT id, username FROM users WHERE username = ?'
        return self.fetch_one(query, (username,))

class DatabaseInitializer(BaseDatabase):
    def __init__(self):
        super().__init__()

    def init_db(self):
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rooms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    room_name TEXT UNIQUE NOT NULL,
                    user_id INTEGER,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cameras (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT UNIQUE NOT NULL,
                    room_name TEXT,
                    status TEXT NOT NULL DEFAULT 'OFF',
                    FOREIGN KEY (room_name) REFERENCES rooms(room_name)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS camera_status (
                    camera_id TEXT PRIMARY KEY,
                    is_assigned INTEGER DEFAULT 0,
                    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
                )
            ''')
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"Error in init_db: {e}")
            raise
        finally:
            conn.close()
