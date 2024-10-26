import sqlite3
import numpy as np
import faiss
import face_recognition

class FaceDatabase:
    def __init__(self, db_path='CAM_SURV.db'):
        self.db_path = db_path
        self.faiss_index = FaissIndex(dimension=128)
        self.init_db()

    def connect_db(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        conn = self.connect_db()
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS known_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_encoding BLOB NOT NULL,
                name TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def add_known_face(self, encoding, name):
        conn = self.connect_db()
        c = conn.cursor()
        encoding_bytes = encoding.tobytes()
        c.execute('INSERT INTO known_faces (face_encoding, name) VALUES (?, ?)', (encoding_bytes, name))
        conn.commit()
        conn.close()
        self.faiss_index.add([encoding], [name])

    def search(self, face_encoding):
        distances, indices = self.faiss_index.search(face_encoding)
        if distances[0][0] < 0.6:  # Adjust tolerance if necessary
            return indices[0][0]
        return None
    
    def get_known_face(self, face_encoding):
        """
        Retrieve a known face from the database using face encoding.
        This uses a distance comparison to find the closest match.
        :param face_encoding: The face encoding to match.
        :return: The name of the person if a match is found, otherwise None.
        """
        conn = self.connect_db()
        c = conn.cursor()
        # Retrieve all encodings and names from the database
        c.execute('SELECT face_encoding, name FROM known_faces')
        results = c.fetchall()

        # Convert the encodings from binary to numpy arrays and find the closest match
        known_encodings = [np.frombuffer(row[0], dtype=np.float64) for row in results]
        known_names = [row[1] for row in results]

        conn.close()

        if not known_encodings:
            return None

        # Use face_recognition to compare the encoding with the database
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        # Find the closest match
        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None
        if best_match_index is not None and matches[best_match_index]:
            return known_names[best_match_index]

        return None

class FaissIndex:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)

    def add(self, encodings, labels):
        self.index.add(np.array(encodings))

    def search(self, encoding, k=1):
        distances, indices = self.index.search(np.array([encoding]), k)
        return distances, indices


    
