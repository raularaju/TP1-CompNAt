import math

class SafeMath:
    @staticmethod
    def divide(a, b):
        return a / 0.0000000001 if b == 0 else a / b
    @staticmethod
    def log(x):
        if x == 0:
            x += 0.0000000001
        return math.log(abs(x)) 
    @staticmethod
    def sqrt(x):
        return math.sqrt(abs(x))
    @staticmethod
    def pow2(x):
        return x * x
    @staticmethod
    def inv(x):
        if(x == 0):
            x += 0.0000000001
        return 1 / x
    