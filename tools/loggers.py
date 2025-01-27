import requests
from logging import DEBUG, FileHandler, Formatter, StreamHandler, getLogger


class myLogger:
    def __init__(self, log_filename=None):
        self.logger = getLogger(__name__)
        self._logInit(log_filename)

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def warn(self, message):
        self.logger.warn(message)

    def warning(self, message):
        self.logger.warning(message)

    def send_line_notification(self, message):
        self.logger.info(message)
        line_token = 'IBlg78go7s0iR5J0k9V6TMiIffvhgIGQuTqbQBKXQeU'
        endpoint = 'https://notify-api.line.me/api/notify'
        message = "\n{}".format(message)
        payload = {'message': message}
        headers = {'Authorization': 'Bearer {}'.format(line_token)}
        requests.post(endpoint, data=payload, headers=headers)

    def _logInit(self, log_filename):
        log_fmt = Formatter('%(asctime)s %(name)s \
                %(lineno)d [%(levelname)s] [%(funcName)s] %(message)s ')
        handler = StreamHandler()
        handler.setLevel('INFO')
        handler.setFormatter(log_fmt)
        self.logger.addHandler(handler)

        if log_filename:
            handler = FileHandler(log_filename, 'a')
            handler.setLevel(DEBUG)
            handler.setFormatter(log_fmt)
            self.logger.setLevel(DEBUG)
            self.logger.addHandler(handler)
