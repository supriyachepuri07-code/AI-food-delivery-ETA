"""
Custom exception handling for the project.
"""

import sys


class CustomException(Exception):
    """
    Custom exception class that provides detailed error information.
    """

    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)

        _, _, exc_tb = error_details.exc_info()

        self.file_name = exc_tb.tb_frame.f_code.co_filename
        self.line_number = exc_tb.tb_lineno
        self.error_message = error_message

    def __str__(self):
        return (
            f"\nError occurred in file: {self.file_name}"
            f"\nLine Number: {self.line_number}"
            f"\nError Message: {self.error_message}"
        )