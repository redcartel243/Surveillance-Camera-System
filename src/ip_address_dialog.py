from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout

class IPAddressDialog(QDialog):
    def __init__(self, parent=None):
        super(IPAddressDialog, self).__init__(parent)
        self.setWindowTitle('Add IP Address')

        layout = QVBoxLayout()

        self.label = QLabel('Enter IP Address:')
        layout.addWidget(self.label)

        self.ip_address_input = QLineEdit(self)
        layout.addWidget(self.ip_address_input)

        button_layout = QHBoxLayout()
        self.ok_button = QPushButton('OK')
        self.cancel_button = QPushButton('Cancel')
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def get_ip_address(self):
        return self.ip_address_input.text()
