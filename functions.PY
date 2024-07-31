import numpy as np
a=np.array([[1,2],
            [6,7],
            [5,9]])
b=np.array([[3,6],
            [5,23],
            [3,2]])
d=np.hstack((a,b))
print(d)
e=np.hsplit(d,3)
print(e)