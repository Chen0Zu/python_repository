import numpy as np

# 1
A = np.array([[1, 3], [2, 6], [3, 9]])
B = np.array([[1, 0], [0, 2], [0, 3]])
C = np.array([[2, 0, 2], [0, 2, 2], [0, 2, 3]])
rankA = np.linalg.matrix_rank(A)
rankB = np.linalg.matrix_rank(B)
rankC = np.linalg.matrix_rank(C)

# 2
v = np.array([4, 1]).reshape(-1, 1)
w = np.array([-2, 2]).reshape(-1, 1)
vplusw = v + w
vmiusw = v - w

# 3
A = np.array([[1, 1], [1, -1]])
B = np.array([[5, 1], [1, 5]])
V = B @ np.linalg.inv(A)

# 4
A = np.array([[2, 1], [1, 2]])
c = 3
d = 1
v = np.array([c, d]).reshape(-1, 1)
B = A @ v

# 6
A = np.array([[1, -2, 1], [0, 1, -1]]).T
c, d = 7, 6
v = np.array([c, d]).reshape(-1, 1)
o1 = np.sum(A @ v)

# 7
A = np.array([[2, 0], [1, 1]])
B = np.array([[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]])
o = A @ B

# 8

# 9

# 10
i = np.array([1,0,0]).reshape(-1,1)
j = np.array([0,1,0]).reshape(-1,1)
iplusj = i + j
print(iplusj)
k = np.array([0,0,1]).reshape(-1,1)
sumijk = i + j + k
print(sumijk)

# 11
centerpoint = np.array([1,1,1]).reshape(-1,1)/2
print(centerpoint)

# 12

# 13
print(0)
v = np.array([np.cos(np.pi/6), np.sin(np.pi/6)]).reshape(-1,1)
print(v)

# 14

# 15

# 16

# 17

# 18

# 19

# 20

# 21

# 22

# 23
print('a plane')

# 24

# 25

# 26
A = np.array([[1,3],[2,1]])
b = np.array([14,8]).reshape(-1,1)
x = np.linalg.inv(A) @ b
print(x)

# 27
