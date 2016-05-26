"""
Contains methods used for generating and interacting with generic measurements.
"""

class Measurement():

    def __init__(self, operators):
        self._operators = operators
        self.num_outcomes = len(self.operators)
        # The size of the matrix associated with this measurement.
        self._size = self.operators[0].shape[0]