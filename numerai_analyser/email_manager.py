import smtplib
import ssl

from email import encoders
from email.mime.base import MIMEBase

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import zipfile
import os
from pathlib import Path

MAX_EMAIL_SIZE = 25e6

def create_attachment(file):

    if os.stat(file).st_size > MAX_EMAIL_SIZE:

        output_file = file.parent / 'preds.zip'

        zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED).write(file)

        file = output_file

    with open(file, 'rb') as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    encoders.encode_base64(part)

    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {file}",
    )

    return part


class EmailManager:

    def __init__(self, sender_email, password, receiver_email, port = 465):

        self.context = ssl.create_default_context()

        self.sender_email = sender_email
        self.receiver_email = receiver_email
        self.password = password

        self.port = port

    def send_email(self, body = None, html = None, header = None, attachment = None):

        message = MIMEMultipart()

        message['Subject'] = header
        message['From'] = self.sender_email
        message['To'] = self.receiver_email

        if body is not None:
            message.attach(MIMEText(body, 'plain'))

        if html is not None:
            message.attach(MIMEText(html, 'html'))

        if attachment is not None:

            if attachment is str:
                attachment = Path(attachment)
            elif not issubclass(type(attachment), Path):
                raise TypeError('attachment field must be string or Path type')

            message.attach(create_attachment(attachment))

        with smtplib.SMTP_SSL('smtp.gmail.com', port = self.port, context = self.context) as server:

            try:
                server.login(self.sender_email, self.password)
            except smtplib.SMTPAuthenticationError as error:
                print('Email login failed')
                print(str(error))
                return

            message_text = message.as_string()

            try:
                server.sendmail(self.sender_email, self.receiver_email, message_text)
            except smtplib.SMTPSenderRefused as error:
                print(str(error))

            server.quit()



if __name__ == "__main__":

    import pandas as pd

    pred_file = Path("/Users/glenmoutrie/Documents/numerai_scripts/datasets/numerai_dataset_232/2020_10_03_212550_predictions.csv")

    email_manager = EmailManager("GBOT.NUMERAI@gmail.com", input('Please provide password:'),  "glen.moutrie@gmail.com")

    email_manager.send_email(body = "Hello World!")

    email_manager.send_email("Check out these predictions.", "Numerai Predictions comp X", pred_file)

    test = pd.read_csv(pred_file)

    email_manager.send_email(body = 'Look at the results in this table', html = test.describe().to_html(), header="Here's a table")



