import smtplib
import os
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
try:
    from dataloader import load_config
except ImportError:
    try:
        from .dataloader import load_config
    except ImportError:
        from src.dataloader import load_config

config = load_config()

def send_email(image_path, csv_path):
    sender_email = config['email']['sender']
    sender_password = config['email']['password']
    receiver_email = config['email']['receiver']

    msg = MIMEMultipart()
    msg['Subject'] = 'Intruder detected'
    msg['From'] = sender_email
    msg['To'] = receiver_email

    text = MIMEText("An intruder has been detected. Please open this website to manage the camera: https://my.ivideon.com/cameras/groups/own")
    msg.attach(text)

    with open(image_path, 'rb') as img_file:
        img_data = img_file.read()
        image = MIMEImage(img_data, name=os.path.basename(image_path))
        msg.attach(image)

    with open(csv_path, 'rb') as attachment:
        p = MIMEBase('application', 'octet-stream')
        p.set_payload(attachment.read())
        encoders.encode_base64(p)
        p.add_header('Content-Disposition', f"attachment; filename={os.path.basename(csv_path)}")
        msg.attach(p)

    with smtplib.SMTP('smtp-mail.outlook.com', 587) as s:
        s.starttls()
        s.login(sender_email, sender_password)
        s.sendmail(sender_email, receiver_email, msg.as_string())
