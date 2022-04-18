import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class MailSend(object):

    def __init__(self, from_addr: list = [], 
                to_addr: list = [], 
                msg: list = [], 
                attach: list = []):
        """
        Args:
            from_addr: list of sender address
            to_addr: list of receiver address
            msg: Body message (type: dictionary)
            attach: list of attachment (images) directory
        """
        self.from_addr = from_addr
        self.to_addr = to_addr
        self.message = msg
        self.attach = attach
    
    def __call__(self):
        """
        Args:

        Encryption Method
         TTL: smtplib.SMTP(smtp.gmail.com, 587)
         SSL: smtplib.SMTP_SSL(smtp.gmail.com, 465)

        """
        smtp = smtplib.SMTP('smtp.gmail.com', 587)
        smtp.starttls()
        smtp.login('singkuserver', 'agcmvqybdqetmsef')

        msg = MIMEMultipart()
        msg['Subject'] = 'Do Not reply'
        msg.attach(MIMEText('Testing mail transfer system ... Learning is over\n', 'plain'))
        idx = 0
        for body in self.message:
            idx += 1
            msg.attach(MIMEText(str(idx) + '\n'))
            msg.attach(MIMEText(str(body) + '\n', 'plain'))

        #attachment = open('')

        smtp.sendmail(self.from_addr, self.to_addr, msg.as_string())

        smtp.quit()
        print("Send Mail to {}".format(self.to_addr))

    def append_msg(self, msg):

        self.message.append(msg)
    
    def append_from_addr(self, addr):

        self.from_addr.append(addr)

    def append_to_addr(self, addr):

        self.to_addr.append(addr)

if __name__ == "__main__":

    ms = MailSend()
    sample = {"F1" : [0.1, 0.9],
                "IoU" : [0.5, 0.4]}
    ms.append_msg(sample)
    ms.append_msg(sample)
    ms.append_msg('Test append message func')
    ms.append_from_addr('doNotReply@gmail.com')
    ms.append_to_addr('sdimivy014@korea.ac.kr')
    ms.append_to_addr('sdimivy014@gmail.com')
    ms()
