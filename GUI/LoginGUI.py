import sys
from PyQt5 import QtWidgets, QtCore
from src.db_func import User, DatabaseInitializer


class LoginWindow(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()

        self.register_button = None
        self.login_button = None
        self.password_input = None
        self.username_input = None
        self.username_label = None
        self.password_label = None
        self.user_id = None  # To store the logged-in user's ID
        self.user_db = User()  # Initialize the User database object
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Login")
        self.setGeometry(400, 200, 300, 150)

        # Create widgets
        self.username_label = QtWidgets.QLabel("Username:", self)
        self.username_input = QtWidgets.QLineEdit(self)

        self.password_label = QtWidgets.QLabel("Password:", self)
        self.password_input = QtWidgets.QLineEdit(self)
        self.password_input.setEchoMode(QtWidgets.QLineEdit.Password)

        self.login_button = QtWidgets.QPushButton("Login", self)
        self.register_button = QtWidgets.QPushButton("Register", self)

        # Set layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.username_label)
        layout.addWidget(self.username_input)
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_input)
        layout.addWidget(self.login_button)
        layout.addWidget(self.register_button)
        self.setLayout(layout)

        # Connect buttons to functions
        self.login_button.clicked.connect(self.login)
        self.register_button.clicked.connect(self.register)

    def login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()

        if not username or not password:
            QtWidgets.QMessageBox.warning(self, "Error", "Please enter both username and password.")
            return

        try:
            if self.user_db.verify_password(username, password):
                QtWidgets.QMessageBox.information(self, "Success", "Login successful!")
                user = self.user_db.get_user(username)
                if user:
                    self.user_id = user[0]  # Get the user ID from the database
                    self.accept()  # Close the dialog and indicate success
                else:
                    QtWidgets.QMessageBox.warning(self, "Error", "User not found.")
            else:
                QtWidgets.QMessageBox.warning(self, "Error", "Invalid username or password.")
                self.password_input.clear()  # Clear password field after failed attempt
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def register(self):
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()

        if not username or not password:
            QtWidgets.QMessageBox.warning(self, "Error", "Please enter both username and password.")
            return

        try:
            self.user_db.store_user(username, password)
            QtWidgets.QMessageBox.information(self, "Success", "User registered successfully.")
            self.clear_inputs()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Error", str(e))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def clear_inputs(self):
        """Clear both the username and password input fields."""
        self.username_input.clear()
        self.password_input.clear()

    def get_user_id(self):
        return self.user_id


if __name__ == "__main__":
    db_initializer = DatabaseInitializer()
    db_initializer.init_db()  # Initialize the database

    app = QtWidgets.QApplication(sys.argv)
    login_window = LoginWindow()

    if login_window.exec_() == QtWidgets.QDialog.Accepted:
        print(f"User ID: {login_window.get_user_id()} logged in successfully!")
        # You can proceed to launch your main application here

    sys.exit(app.exec_())
