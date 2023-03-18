import numpy as np

def binary(integer: int, n: int) -> list:
    assert integer > 0, f"Error input"
    assert n > 0, f"Error input"
    # Create list of n elements of 0
    re =[]
    for i in range(n):
        re.append(0)
    # Convert decimal to binary
    bin = format(integer, 'b')
    # Store binary digits into a list
    l = []
    for i in bin:
        l.append(int(i))
    # Insert binary list into list of zeros
    for i in range(len(l)):
        i += 1
        re[-i] = l[-i]
    return re

def gauss(mean: float, std: float, n: int) -> list:
    assert n > 0, f"Error input"
    gauss = np.random.normal(loc=mean, scale=std, size=n)
    return gauss

def num(bin: list) -> int:
    num = f""
    for i in bin:
        num += str(i)
    return int(num, 2)



