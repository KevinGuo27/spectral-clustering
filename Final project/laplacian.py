import numpy as np
L = np.array([[2,-1,-1,0,0,0,0],[-1,3,-1,-1,0,0,0],[-1,-1,2,0,0,0,0],[0,-1,0,3,-1,-1,0],[0,0,0,-1,2,-1,0],[0,0,0,0,-1,2,-1],[0,0,0,-1,0,-1,2]])
D = np.diag(np.diag(L))
D = np.sqrt(np.linalg.inv(D))
L = D @ L @ D
# preserve L to 3 digits
# preserve U,S V to 3 digits
np.set_printoptions(precision = 3, suppress=True)
L = np.around(L,3)
print(L)
U,S,V = np.linalg.svd(L)

print(f"Eigenvectors of L")
print(U)
print(f"eigenvalues of L")
print(S)

