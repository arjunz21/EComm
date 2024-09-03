import sys
import logger

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        _, _, exc_tb = error_detail.exc_info()
        super().__init__(error_message)
        self.error_message = f"Error occured in Line number: [{exc_tb.tb_lineno}] \nPython Script Name: [{exc_tb.tb_frame.f_code.co_filename}] \nError Message: [{error_message}]"
        logger.logging.error(self.error_message)

    def __str__(self):
        return self.error_message
