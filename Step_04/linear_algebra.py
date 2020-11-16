# linear algebra
import tensorflow as tf
# 1 
a = tf.constant([1,2,3], dtype = tf.float32)
print('norm(a) = {}'.format(tf.norm(a)))

# 2
A = tf.constant([[1,2],[3,4]], dtype = tf.float32)
B = tf.linalg.matrix_transpose(A)

print('matrix A : {}'.format(A))
print("A' = {}".format(B))

# 3
C = tf.linalg.det(A)
print('Determinant of A: {}'.format(C))

D = tf.linalg.inv(A)
print('Inverse matrix of A: {}'.format(D))

E = tf.matmul(A, B) # E = tf.linalg.matmul(A, B)
print('Inner production of A and B: {}'.format(E))

# 4
A = tf.constant([[1,4,1],
                 [1,6,-1],
                 [2,-1,2]], dtype = tf.float32)
b = tf.constant([[7],
                 [13],
                 [5]], dtype = tf.float32)

# 5
'''
solve linear equation
Problem
 x1 + 4x2 +  x3 =  7
 x1 + 6x2 -  x3 = 13
2x1 -  x2 + 2x3 =  5

| 1  4  1 | |x1| = |  7 |
| 1  6 -1 | |x2| = | 13 |
| 2 -1  2 | |x3| = |  5 |

   Ax = b
   if det(A) is not equal to 0, there exist inv(A) and a unique solution.
   
step 1: A = PLU
step 2: Ly = P'b
step 3: Ux = y

* if a square matrix (n x n) has a unique solution, rank(A) = n (If rank(A) = n, n column vector in each row are independent.)
* The number of nonzero diagonal elements of U is qual to n.
* det(n) != 0
    Ax = b
    PLUx = b    : A = PLU
    LUx = P'b   : inv(P) = P'
    Ly = P'b    : forward substitution
    Ux = y      : backward substitution
    
L_U = |   2    -1    1 |  p = [2, 1, 0] 
      | 0.5   6.5   -2 |
      | 0.5  0.69 2.38 |
      
      
P = | 0 0 1 |  L = |   1    0    1 |   U = |  2    -1    1 |
    | 0 1 0 |      | 0.5    1    0 |       |  0   6.5   -2 |
    | 1 0 0 |      | 0.5 0.69    0 |       |  0     0 1.38 |
'''
print('A : {}'.format(A))
print('det(A) : {}'.format(tf.linalg.det(A))) # if det(A) is not equal to 0, 

x = tf.matmul(tf.linalg.inv(A), b) # solution
print('solution : {}'.format(x))

# 6
def all_close(x,y,tol = 1e-5):
    #return tf.reduce_sum(tf.abs(x-y)) < tol
    return tf.reduce_sum(tf.square(x-y)) < tol

print('is x a correct solution? : {}'.format(all_close(tf.matmul(A,x), b)))

# 7
x2 = tf.linalg.solve(A, b)
print('solution : {}'.format(x2))

print('is x2 a correct solution? : {}'.format(all_close(tf.matmul(A, x), b)))


# 8 
'''
Solve linear equation using LU-decomposition
'''
L_U, p = tf.linalg.lu(A)
print('L_U : \n{}'.format(L_U))
print('p : \n{}'.format(p))

# make P, L, U
U = tf.linalg.band_part(L_U, 0, -1) # upper triangluar matrix
print('U : \n{}'.format(U))

L = tf.linalg.band_part(L_U, -1, 0) # lower triangular matrix
print('L : \n{}'.format(L))

L = tf.linalg.set_diag(L, [1,1,1]) # strictly lower triangular part of LU
print('strictly lower triangular matrix : \n{}'.format(L))

P = tf.gather(tf.eye(3), p)
print('P : \n{}'.format(P))

# 9 chk A = PLU
# 9-1
A_reconstructed = tf.linalg.lu_reconstruct(L_U, p)
print('reconstructed A : \n{}'.format(A_reconstructed))

# 9-2 calculate directly the same as # 3-1
A_chk = tf.matmul(P, tf.matmul(L, U)) # tf.gather(tf.matmul(L, U), p)
print('Check reconstructed A : \n{}'.format(A_chk))

# 10 : solve AX = b using PLUx = b
# 10-1
sol_1 = tf.linalg.lu_solve(L_U, p, b)
print('Solution : \n{}'.format(sol_1))

# 10-2: calculate directly the same as
y = tf.linalg.triangular_solve(L, tf.matmul(tf.transpose(P), b))
x = tf.linalg.triangular_solve(U, y, lower = False)
print('Check solution : \n{}'.format(x))

# 11 : stuff: pivots, calculate det(A, rank(A))
D = tf.linalg.diag_part(L_U) # tf.linalg.diag_part(U) # diagonal elements of L_U
print('Diagonal elemets of A : \n{}'.format(D))

rank = tf.math.count_nonzero(D) # count amount of nonzero elements

det_U = tf.reduce_prod(tf.linalg.diag_part(U)) # tf.linalg.det(U)
print('det(U) : \n{}'.format(det_U))

det_L = tf.reduce_prod(tf.linalg.diag_part(L)) # tf.linalg.det(L)
print('det(L) : \n{}'.format(det_L))

det_P = tf.linalg.det(P)
print('det(P) : \n{}'.format(det_P))
 
det_A = det_P * det_L * det_U # tf.linalg.det(A)
print('det(A) : \n{}'.format(det_A))
