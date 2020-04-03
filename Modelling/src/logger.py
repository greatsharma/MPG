import os
from datetime import datetime


LOGS_PATH = os.environ.get('LOGS_PATH')


def log(file_obj, msg, date_time=False, clear=False):
    if clear:
        open(LOGS_PATH, 'w').close()

    if date_time:
        now = datetime.now()
        date = now.date()
        current_time = now.strftime("%H:%M:%S")
        file_obj.write('\n\n\n' + '[' + str(date) + "/" + str(current_time) + ']' + msg)
    else:
        file_obj.write(msg)


file_obj = open(LOGS_PATH, "a")