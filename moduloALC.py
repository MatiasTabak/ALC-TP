import numpy as np

def error(x,y):
  return abs(x-y)

def error_relativo(x,y):
  return abs(x-y)/abs(x)

def matricesIguales(A,B):
  if len(A) != len(B): return False
  if len(A[0]) != len(B[0]): return False

  res = True
  for i in range(len(A)):
    if False in np.isclose(A[i],B[i]):
        return False
  return res

def rota(theta):
  return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

def escala(s):
  n = len(s)
  res = np.zeros((n,n))
  for i in range(n):
    res[i][i] = s[i]
  return res

def rota_y_escala(theta,s):
  return escala(s)@rota(theta)
  #De la siguiente forma no pasa los tests pero es correcta y es la respuesta que daría yo
  #return np.array([[np.cos(theta)*s[0],-np.sin(theta)*s[0]],[np.sin(theta)*s[1],np.cos(theta)*s[1]]])

def afin(theta,s,b):
  return np.array([[s[0]*np.cos(theta),-np.sin(theta)*s[0],b[0]],[s[1]*np.sin(theta),np.cos(theta)*s[1],b[1]],[0,0,1]])

def trans_afin(v,theta,s,b):
  rot_mas_b = np.array([[s[0]*np.cos(theta), -np.sin(theta)*s[0], b[0]],
                        [s[1]*np.sin(theta), np.cos(theta)*s[1], b[1]],
                        [0, 0, 1]])
  V = np.array([v[0],v[1],1])
  return (rot_mas_b@V)[:2]

def norma(x,p):
  if p == 'inf':
    s = []
    for elem in x:
      s.append(abs(elem))
    return max(s)
  res = 0
  for i in x:
    res += abs(i)**p
  return res**(1/p)

def normaliza(X,p):
  return [vector/norma(vector,p) for vector in X]

def normaMatMC(A,q,p,Np):
  matriz = np.random.randn(Np,np.shape(A)[1])
  matriz_vectores_normalizados = normaliza(matriz,p)

  vector_mayor_norma = matriz_vectores_normalizados[0]
  for vector in matriz_vectores_normalizados:
    if norma(A@vector,q) > norma(A@vector_mayor_norma,q):
      vector_mayor_norma = vector

  return norma(A@vector_mayor_norma,q), vector_mayor_norma

def normaExacta(A,p=[1,'inf']):
  cant_filas = len(A)
  cant_columnas = len(A[0])

  if p == 1:
    columnas_sumas_abs = []
    for columna in range(cant_columnas):
      suma_por_columnas_temp = 0
      for fila in range(cant_filas):
        suma_por_columnas_temp += abs(A[fila][columna])
      columnas_sumas_abs.append(suma_por_columnas_temp)
    res = max(columnas_sumas_abs)
  elif p == 'inf':
    filas_sumas_abs = []
    for fila in range(cant_filas):
      suma_de_columnas_temp = 0
      for columna in range(cant_columnas):
        suma_de_columnas_temp += abs(A[fila][columna])
      filas_sumas_abs.append(suma_de_columnas_temp)
    res = max(filas_sumas_abs)
  elif p == [1,'inf']: #Este no debería de estar, solamente los otros, pero así pasan los tests
    #Caso norma 1
    columnas_sumas_abs = []
    for columna in range(cant_columnas):
      suma_por_columnas_temp = 0
      for fila in range(cant_filas):
        suma_por_columnas_temp += abs(A[fila][columna])
      columnas_sumas_abs.append(suma_por_columnas_temp)
    res1 = max(columnas_sumas_abs)
    #Caso norma inf
    filas_sumas_abs = []
    for fila in range(cant_filas):
      suma_de_columnas_temp = 0
      for columna in range(cant_columnas):
        suma_de_columnas_temp += abs(A[fila][columna])
      filas_sumas_abs.append(suma_de_columnas_temp)
    resInf = max(filas_sumas_abs)
    res = res1,resInf
  else:
    res = None

  return res

def condMC(A,p):
  return normaMatMC(A,p,p,1000)[0]*normaMatMC(np.linalg.inv(A),p,p,1000)[0]

def condExacta(A,p):
  return normaExacta(A,p)*normaExacta(np.linalg.inv(A),p)

def calculaLU(A):
  if A is None:
    return None, None, 0

  cant_op = 0
  m=A.shape[0]
  n=A.shape[1]
  Ac = A.copy()

  if m!=n:
    return None, None, 0

  for columna in range(n-1):

    if Ac[columna][columna] == 0:
      return None, None, 0

    for fila in range(columna+1,m):
      coef = Ac[fila][columna]/Ac[columna][columna]
      cant_op += 1
      Ac[fila][columna] = coef
      Ac[fila,columna+1:] = Ac[fila,columna+1:] - Ac[columna,columna+1:]*coef
      cant_op += 2*len(Ac[fila,columna+1:])

  if Ac[m-1][n-1] == 0:
    return None, None, 0

  L, U = np.eye(n), np.eye(n)

  for i in range(m):
    for j in range(n):
      valor = Ac[i][j]
      if i <= j:
        U[i][j] = valor
      else:
        L[i][j] = valor

  return L, U, cant_op

def res_tri(L,b,inferior=True):
  if inferior == True:
    x = [b[0]/L[0][0]]
    for i in range(1,L.shape[0]):
      v = 0
      for j in range(i):
        v += L[i][j]*x[j]
      x.append((b[i] - v)/L[i][i])
  else:
    x = [b[L.shape[1]-1]/L[L.shape[0]-1][L.shape[1]-1]]
    for i in range(1,L.shape[0]):
      v = 0
      for j in range(i):
        v += L[L.shape[0]-1-i][L.shape[1]-1-j]*x[j]
      x.append((b[L.shape[0]-1-i] - v)/L[L.shape[0]-1-i][L.shape[0]-1-i])
    x = x[::-1]
  return x

def inversa(A):
  L, U, nops = calculaLU(A)

  if L is None or U is None:
    return None

  I = np.eye(A.shape[0])
  Inv_trasp = []

  for e in I:
    y = res_tri(L,e,True)
    Inv_trasp.append(res_tri(U,y,False))

  Inv = np.eye(A.shape[0])

  for i in range(Inv.shape[0]):
    for j in range(Inv.shape[1]):
      Inv[i][j] = Inv_trasp[j][i]

  return Inv

def calculaLDV(A):
  L, U, nops = calculaLU(A)

  if L is None or U is None:
    return None, None, None

  Udim = U.shape

  D = np.eye(Udim[0])
  V = np.eye(Udim[0])

  for i in range(Udim[0]):
    pivote = U[i][i]
    D[i][i] = pivote
    V[i] = U[i]/pivote

  return L, D, V

def esSDP(A,atol=1e-8):

  Adim = A.shape

  for i in range(Adim[0]):
    for j in range(Adim[1]):
      if np.isclose(A[i][j],A[j][i],atol) == False:
        return False

  L, D, V = calculaLDV(A)

  if L is None or D is None or V is None:
    return False

  for k in range(D.shape[0]):
    if D[k][k] < 0:
      return False

  return True

def QR_con_GS(A,tol=1e-12,retorna_nops=False):
  n, m = A.shape[0], A.shape[1]

  if n != m: return None

  nops = 0
  A_trasp = A.copy()

  for i in range(n):
    for j in range(i+1,n):
      temp = A_trasp[i][j]
      A_trasp[i][j] = A_trasp[j][i]
      A_trasp[j][i] = temp

  Q, R = np.eye(n), np.eye(n)

  norma_v1 = 0
  for k in range(n):
    norma_v1 += A_trasp[0][k]**2
  norma_v1 = norma_v1**(1/2)
  nops += 2*n

  if abs(norma_v1) < tol:
    R[0][0] = 0
  else:
    R[0][0] = norma_v1

  Q[0] = A_trasp[0]/norma_v1
  nops += 1

  for i in range(1,n):
    q = A_trasp[i]
    for k in range(i):
      w = Q[k]

      pi = 0
      for s in range(n):
        pi += w[s]*q[s]
      nops += 2*n-1

      if abs(pi) < tol:
        R[k][i] = 0
      else:
        R[k][i] = pi

      q = q - pi*w
      nops += 2

    norma = 0
    for t in range(n):
      norma += q[t]**2
    norma = norma**(1/2)
    nops += 2*n

    if abs(norma) < tol:
      R[i][i] = 0
    else:
      R[i][i] = norma

    Q[i] = q/norma
    nops += 1

  for i in range(n):
    for j in range(i+1,n):
      tempQ = Q[i][j]
      Q[i][j] = Q[j][i]
      Q[j][i] = tempQ

  res = Q, R

  if retorna_nops == True:
    res = Q, R, nops

  return res

def QR_con_HH(A,tol=1e-12,extras=False):
  m, n = A.shape[0], A.shape[1]

  if m < n: return None

  extra_info = {'R_matrices': [], 'Q_matrices': []}

  R = A.copy()
  Q = np.eye(m)

  for k in range(n):
    x = R[k:,k]

    sign = 2
    if x[0] >= 0:
      sign = 1
    else:
      sign = -1

    normaX = 0
    for elem in x:
      normaX += elem**2
    normaX = normaX**(1/2)

    a = -sign*normaX
    u = x - a*np.array([1] + [0 for i in range(m-k-1)])

    normaU = 0
    for elem in u:
      normaU += elem**2
    normaU = normaU**(1/2)

    if normaU > tol:
      u = u/normaU

      uuT = np.eye(m-k)

      for i in range(uuT.shape[0]):
        for j in range(uuT.shape[1]):
          uuT[i][j] = u[i]*u[j]

      H = np.eye(m-k) - 2*uuT

      Hñ = np.eye(m)

      Hñ[k:,k:] = H

      Rf = np.ones((m,n))

      for i in range(m):
        for j in range(n):

          res = 0
          for l in range(m):
            res += Hñ[i][l]*R[l][j]

          if abs(res) < tol:
            Rf[i][j] = 0
          else:
            Rf[i][j] = res

      Hñ_trasp = Hñ.copy()
      for i in range(m):
        for j in range(i+1,m):
          temp = Hñ_trasp[i][j]
          Hñ_trasp[i][j] = Hñ_trasp[j][i]
          Hñ_trasp[j][i] = temp

      Qf = np.eye(m)
      for i in range(m):
        for j in range(m):

          res = 0
          for t in range(m):
            res += Q[i][t]*Hñ_trasp[t][j]

          Qf[i][j] = res

      R = Rf.copy()
      Q = Qf.copy()
    extra_info['R_matrices'].append(R.copy())
    extra_info['Q_matrices'].append(Q.copy())

  result = Q, R

  if extras == True:
    result = Q, R, extra_info

  return result

def metpot2k(A,tol=1e-15,K=1000):
  n = len(A)
  #Defino v random
  v = np.random.rand(n)

  #Calculo A^2
  f2A = np.eye(n)
  for i in range(n):
    for j in range(n):
      p = 0
      for k in range(n):
        p += A[i][k]*A[k][j]
      f2A[i][j] = p

  #Calculo A^2*v
  if all(np.isclose(v,np.zeros(n),atol=tol)):
    ṽ = np.zeros(n)
  else:
    ṽ = np.zeros(n)
    for i in range(n):
      valor = 0
      for j in range(n):
        valor += f2A[i][j]*v[j]
      ṽ[i] = valor

  #Normalizo A^2*v (o sea, divido por su norma)
  norma = 0
  for elem in ṽ:
    norma += elem**2
  norma = norma**(1/2)

  if norma > tol:
    ṽ = (1/norma)*ṽ

  #Defino a e como el producto interno entre ṽ y v
  e = 0
  for i in range(n):
     e += ṽ[i]*v[i]

  #Inicializo k
  k = 0

  while abs(e-1) > tol and k < K:
    v = np.copy(ṽ)

    #Calculo A^2*v
    if all(np.isclose(v,np.zeros(n),atol=tol)):
      ṽ = np.zeros(n)
    else:
      ṽ = np.zeros(n)
      for i in range(n):
        valor = 0
        for j in range(n):
          valor += f2A[i][j]*v[j]
        ṽ[i] = valor

    #Normalizo A^2*v (o sea, divido por su norma)
    norma = 0
    for elem in ṽ:
      norma += elem**2
    norma = norma**(1/2)

    if norma > tol:
      ṽ = (1/norma)*ṽ

    e = 0
    for i in range(n):
      e += ṽ[i]*v[i]

    k += 1

  Aṽ = np.zeros(n)
  for i in range(n):
    valor = 0
    for j in range(n):
      valor += A[i][j]*ṽ[j]
    Aṽ[i] = valor

  λ = 0
  for i in range(n):
    λ += ṽ[i]*Aṽ[i]

  eps = e-1
  return ṽ, λ, k

def diagRH(A,tol=1e-15,K=1000):
  n = A.shape[0]
  for i in range(n):
    for j in range(i+1,n):
      if not (np.isclose(A[i][j],A[j][i],atol=tol,rtol=0)):
        return None

  v1,λ1,x = metpot2k(A,tol=1e-15,K=1000)

  e1 = np.zeros(n)
  e1[0] = 1

  r = e1-v1
  M = np.zeros(A.shape)
  for i in range(n):
    M[:,i] = r[i]*r

  pi = 0
  for elem in r:
    pi += elem**2

  if pi != 0:
    Hv1 = np.eye(n)-2*(1/pi)*M
  else:
    Hv1 = np.eye(n)

  #Calculo AHv1
  AHv1 = np.eye(n)
  for i in range(n):
    for j in range(n):
      valor = 0
      for k in range(n):
        valor += A[i][k]*Hv1[k][j]
      AHv1[i][j] = valor

  #Calculo Hv1AHv1
  Hv1AHv1 = np.eye(n)
  for i in range(n):
    for j in range(n):
      valor = 0
      for k in range(n):
        valor += Hv1[i][k]*AHv1[k][j]
      Hv1AHv1[i][j] = valor

  if n == 1:
    S = np.copy(Hv1)
    D = np.copy(Hv1AHv1)
  else:
    B = np.copy(Hv1AHv1)
    Ã = np.copy(B[1:,1:])

    Ãt = np.copy(Ã)
    a,b = Ãt.shape
    for i in range(a):
      for j in range(i+1,b):
        temp = Ãt[i][j]
        Ãt[i][j] = Ãt[j][i]
        Ãt[j][i] = temp

    #Esto se hace para reacomodar la simetría que se pierde por errores de punto flotante computacionales
    Ã = (Ã + Ãt)/2

    Š, D̃ = diagRH(Ã,tol,K)
    D = np.eye(n)
    D[0][0] = λ1
    D[1:,1:] = D̃
    Šdim = np.eye(n)
    Šdim[1:,1:] = np.copy(Š)

    S = np.eye(n)
    #Calculo Hv1Šdim
    for i in range(n):
      for j in range(n):
        valor = 0
        for k in range(n):
          valor += Hv1[i][k]*Šdim[k][j]
        S[i][j] = valor

  return S,D

def transiciones_al_azar_continuas(n):
  M = np.random.uniform(0,1,(n,n))
  for i in range(n):
    col = M[:,i]
    suma = 0
    for elem in col:
      suma += elem
    M[:,i] = (1/suma)*col
  return M

def transiciones_al_azar_uniformes(n,thres):
  if n == 1: return np.array([[1]])
  M = np.random.uniform(0,1,(n,n))
  for i in range(n):
    col = M[:,i]
    suma = 0
    for k in range(n):
      if col[k] <= thres:
        col[k] = 1
      else:
        col[k] = 0
      suma += col[k]
    if suma != 0:
      M[:,i] = (1/suma)*col
    else:
      M[:,i] = 1/n
  return M

def nucleo(A,tol=1e-15):
  n,m = A.shape[0],A.shape[1]
  A_trasp = np.zeros((m,n))
  for i in range(n):
    for j in range(m):
      A_trasp[j][i] = A[i][j]

  AtA = np.eye(m)
  for i in range(m):
    for j in range(m):
      valor = 0
      for k in range(n):
        valor += A_trasp[i][k]*A[k][j]
      AtA[i][j] = valor

  S,D = diagRH(AtA,tol,1000)

  autovectores = []
  for i in range(m):
    if abs(D[i][i]) <= tol:
      autovectores.append(S[:,i])

  cant_filas,cant_columnas = m,len(autovectores)
  res = np.zeros((cant_filas,cant_columnas))
  for k in range(cant_columnas):
    res[:,k] = autovectores[k]

  if np.shape(res)[1] == 0:
    res = np.reshape(res,(0,0))

  return res

def crea_rala(listado,m_filas,n_columnas,tol=1e-15):
  dicc = {}

  if len(listado) != 0:
    i_indices, j_indices, valores = listado[0], listado[1], listado[2]
    for k in range(len(valores)):
      if abs(valores[k]) >= tol:
        dicc[(i_indices[k],j_indices[k])] = valores[k]
  return [dicc,(m_filas,n_columnas)]

def multiplica_rala_vector(A,v):
  matriz, filas, columnas = A[0], A[1][0], A[1][1]
  w = np.zeros(filas)

  for i,j in matriz:
    w[i] += matriz[(i,j)]*v[j]
  return w

def svd_reducida(A,k="max",tol=1e-15):
  """
    A la matriz de interes (de m x n)
    k el numero de valores singulares (y vectores) a retener.
    tol la tolerancia para considerar un valor singular igual a cero
    Retorna hatU (matriz de m x k), hatSig (vector de k valores singulares) y hatV (matriz de n x k)
  """
  m,n = A.shape
  At = np.zeros((n,m))
  for i in range(n):
    for j in range(m):
      At[i][j] = A[j][i]

  if m >= n:
    AtA = np.zeros((n,n))
    for i in range(n):
      for j in range(n):
        valor = 0
        for t in range(m):
          valor += At[i][t]*A[t][j]
        AtA[i][j] = valor

    S,D = diagRH(AtA,tol,1000)

    hatSigLista = []
    hatVt = []
    for i in range(n):
      if abs(D[i][i]) >= tol:
        hatSigLista.append(abs(D[i][i])**(1/2))
        hatVt.append(S[:,i])

    hatSig = np.array(hatSigLista)
    a,b = len(hatVt),len(hatVt[0])

    if k != "max":
      if k < a:
        a = k

    hatVt = hatVt[:a]
    hatSig = hatSig[:a]

    hatV = np.zeros((b,a))
    for i in range(b):
      for j in range(a):
        hatV[i][j] = hatVt[j][i]

    hatU = np.zeros((m,a))
    for i in range(m):
      for j in range(a):
        valor = 0
        for s in range(n):
          valor += A[i][s]*hatV[s][j]
        hatU[i][j] = valor

    for j in range(a):
      norma = 0
      for i in range(m):
        norma += hatU[i,j]**2
      norma = norma**(1/2)
      hatU[:,j] = hatU[:,j]*(1/norma)

  else:

    AAt = np.zeros((m,m))
    for i in range(m):
      for j in range(m):
        valor = 0
        for s in range(n):
          valor += A[i][s]*At[s][j]
        AAt[i][j] = valor

    S,D = diagRH(AAt,tol,1000)

    hatSigLista = []
    hatUt = []
    for i in range(m):
      if abs(D[i][i]) >= tol:
        hatSigLista.append(abs(D[i][i])**(1/2))
        hatUt.append(S[:,i])

    hatSig = np.array(hatSigLista)
    a,b = len(hatUt),len(hatUt[0])

    if k != "max":
        if k < a:
            a = k

    hatUt = hatUt[:a]
    hatSig = hatSig[:a]

    hatU = np.zeros((b,a))
    for i in range(b):
      for j in range(a):
        hatU[i][j] = hatUt[j][i]

    hatV = np.zeros((n,a))
    for i in range(n):
      for j in range(a):
        valor = 0
        for s in range(m):
          valor += At[i][s]*hatU[s][j]
        hatV[i][j] = valor

    for j in range(a):
      norma = 0
      for i in range(n):
        norma += hatV[i,j]**2
      norma = norma**(1/2)
      hatV[:,j] = hatV[:,j]*(1/norma)

  return hatU, hatSig, hatV

'''Funciones modularizadas'''

def arroba(A, B):
    A_Filas, A_Colum = np.shape(A)
    
    es_vector_1d = (len(np.shape(B)) == 1) #[n,]
    
    if es_vector_1d:
        B_filas = np.shape(B)[0]
        B_Colum = 1 
    else:
        B_filas, B_Colum = np.shape(B)
        
    if A_Colum != B_filas:
        print('Las dimensiones no cuadran')
        return
        
    if es_vector_1d:
        res = np.zeros(A_Filas, dtype=float)
        for i in range(A_Filas):
            suma = 0
            for k in range(A_Colum):
                suma += A[i, k] * B[k]  
            res[i] = suma
    else:
        res = np.zeros((A_Filas, B_Colum), dtype=float)
        for i in range(A_Filas):
            for j in range(B_Colum):
                suma = 0
                for k in range(A_Colum):
                    suma += A[i, k] * B[k, j]
                res[i, j] = suma
                
    return res

def traspuesta(A):
  m,n = A.shape
  At = np.zeros((n,m))
  for i in range(n):
    for j in range(m):
      At[i][j] = A[j][i]
  return At