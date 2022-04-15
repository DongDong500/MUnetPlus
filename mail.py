import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


if __name__ == "__main__":

    smtp = smtplib.SMTP('smtp.gmail.com', 587)
    smtp.starttls()
    smtp.login('singkuserver', 'agcmvqybdqetmsef')

    msg = MIMEMultipart()
    msg['Subject'] = 'Test Mail'
    msg.attach(MIMEText('Test Main', 'plain'))

    #attachment = open('')

    smtp.sendmail("do-not-reply@gmail.com", "sdimivy014@gmail.com", msg.as_string())

    smtp.quit()