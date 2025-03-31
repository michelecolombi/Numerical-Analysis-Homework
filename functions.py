import numpy as np
from scipy.linalg import solve
from numpy.linalg import norm


################################ Metodi espliciti 

def explicit_euler(f, u0, t):
    """
    Metodo di Eulero esplicito per ODE

    Input:
    - f: funzione del problema (f(t, y))
    - u0: condizione iniziale (array 1D)
    - t: array degli istanti temporali

    Output:
    - u: array con la soluzione approssimata in ciascun istante
    - f_eval: numero di valutazioni della funzione f
    """
    N = len(t)
    d = u0.shape[0]
    u = np.zeros((d, N))
    u[:, 0] = u0
    f_eval = 0
    for i in range(1, N):
        h = t[i] - t[i - 1]
        u[:, i] = u[:, i - 1] + h * f(t[i - 1], u[:, i - 1])
        f_eval += 1
    return u, f_eval

def heun(f, u0, t):
    """
    Metodo di Heun di ordine 2 (RK2)

    Input:
    - f: funzione del problema (f(t, y))
    - u0: condizione iniziale (array 1D)
    - t: array degli istanti temporali

    Output:
    - u: array con la soluzione approssimata in ciascun istante
    - f_eval: numero di valutazioni della funzione f
    """
    N = len(t)
    d = u0.shape[0]
    u = np.zeros((d, N))
    u[:, 0] = u0
    f_eval = 0
    for i in range(1, N):
        h = t[i] - t[i - 1]
        k1 = f(t[i - 1], u[:, i - 1])
        k2 = f(t[i - 1] + h, u[:, i - 1] + h * k1)
        u[:, i] = u[:, i - 1] + h * 0.5 * (k1 + k2)
        f_eval += 2
    return u, f_eval

def heun3(f, u0, t):
    """
    Metodo di Heun di ordine 3

    Input:
    - f: funzione del problema (f(t, y))
    - u0: condizione iniziale (array 1D)
    - t: array degli istanti temporali

    Output:
    - u: array con la soluzione approssimata in ciascun istante
    - f_eval: numero di valutazioni della funzione f
    """
    N = len(t)
    d = u0.shape[0]
    u = np.zeros((d, N))
    u[:, 0] = u0
    f_eval = 0
    for i in range(1, N):
        h = t[i] - t[i - 1]
        k1 = f(t[i - 1], u[:, i - 1])
        k2 = f(t[i - 1] + h / 3, u[:, i - 1] + h * k1 / 3)
        k3 = f(t[i - 1] + 2 * h / 3, u[:, i - 1] + 2 * h * k2 / 3)
        u[:, i] = u[:, i - 1] + h * (k1 + 3 * k3) / 4
        f_eval += 3
    return u, f_eval

def midpoint(f, u0, t):
    """
    Metodo del punto medio (RK2)

    Input:
    - f: funzione del problema (f(t, y))
    - u0: condizione iniziale (array 1D)
    - t: array degli istanti temporali

    Output:
    - u: array con la soluzione approssimata in ciascun istante
    - f_eval: numero di valutazioni della funzione f
    """
    N = len(t)
    d = u0.shape[0]
    u = np.zeros((d, N))
    u[:, 0] = u0
    f_eval = 0
    for i in range(1, N):
        h = t[i] - t[i - 1]
        k1 = f(t[i - 1], u[:, i - 1])
        k2 = f(t[i - 1] + h/2, u[:, i - 1] + h/2 * k1)
        u[:, i] = u[:, i - 1] + h * k2
        f_eval += 2
    return u, f_eval

def erk_solver(f, u0, t, A, b, c):
    """
    Risolutore Runge-Kutta esplicito generico

    Input:
    - f: funzione del problema (f(t, y))
    - u0: condizione iniziale (array 1D)
    - t: array degli istanti temporali
    - A, b, c: coefficienti del tableau di Butcher 

    Output:
    - u: soluzione approssimata per ogni tempo
    - f_eval: numero di valutazioni di f
    """
    s = A.shape[1]
    d = u0.shape[0]
    N = len(t)
    u = np.zeros((d, N))
    u[:, 0] = u0
    f_eval = 0
    for i in range(1, N):
        h = t[i] - t[i - 1]
        K = np.zeros((s, d))
        K[0, :] = f(t[i - 1], u[:, i - 1])
        for j in range(1, s):
            K[j, :] = f(t[i - 1] + c[j - 1] * h, u[:, i - 1] + h * np.einsum("j,jd->d", A[j, :j], K[:j, :]))
        u[:, i] = u[:, i - 1] + h * np.einsum("i,ij->j", b, K)
        f_eval += s
    return u, f_eval


def ab(f, t, u0, k):
    """
    Metodo di Adams-Bashforth esplicito di ordine 2, 3 o 4.

    Input:
    - f: funzione del problema (f(t, y))
    - t: array dei tempi
    - u0: condizione iniziale (array 1D)
    - k: ordine del metodo (2, 3 o 4)

    Output:
    - u: soluzione approssimata in ciascun istante t
    - n_eval: numero totale di valutazioni di f
    """
    
    if k == 2:
        beta = 0.5 * np.array([3, -1])
        A = np.array([[0, 0], [1, 0]])
        b = np.array([0.5, 0.5])
        c = np.array([0, 1])
    elif k == 3:
        beta = (1/12) * np.array([23, -16, 5])
        A = np.array([[0, 0, 0], [1/3, 0, 0], [0, 2/3, 0]])
        b = np.array([0.25, 0.0, 0.75])
        c = np.array([0, 1/3, 2/3])
    elif k == 4:
        beta = (1/24) * np.array([55, -59, 37, -9])
        A = np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]])
        b = np.array([1/6, 1/3, 1/3, 1/6])
        c = np.array([0, 0.5, 0.5, 1])

    N = len(t)
    d = len(u0)
    u = np.zeros((d, N))
    stor = np.zeros((d, k))
    n_eval = 0

    # Calcolo primi k valori con RK esplicito
    u[:, :k], f_eval = erk_solver(f, u0, t[:k], A, b, c)
    n_eval += f_eval

    for i in range(k):
        stor[:, i] = f(t[i], u[:, i])
    n_eval += k

    for i in range(k, N):
        h = t[i] - t[i - 1]
        u[:, i] = u[:, i - 1]
        for j in range(k):
            u[:, i] += h * beta[j] * stor[:, k - j - 1]  
        new_f = f(t[i], u[:, i]).reshape(-1, 1)
        stor = np.hstack((stor[:, 1:], new_f))
        n_eval += 1

    return u, n_eval



################################ Metodi impliciti

def newton_method_for_bdf(u, w, f, beta0, t, h, Jf, tol=1e-5, max_iter=100000):
    """
    Metodo di Newton per la risoluzione del passo implicito BDF.

    Input:
    - u: stima iniziale
    - w: termine noto del metodo BDF
    - f: funzione del problema (f(t, y))
    - beta0: coefficiente BDF
    - t: tempo attuale
    - h: passo temporale
    - Jf: funzione Jacobiana df/dy
    - tol: tolleranza sulla convergenza
    - max_iter: numero massimo di iterazioni

    Output:
    - y: soluzione al passo
    - f_eval: numero di valutazioni di f/Jf
    - flag: True se non converge
    """
    m = len(u)
    I = np.eye(m)
    y = u
    f_eval = 0
    rel_err = 1
    i = 0
    while rel_err > tol and i < max_iter:
        J = I - h * beta0 * Jf(t, y)
        s = solve(J, -y + h * beta0 * f(t, y) - w) # linalg.solve(A,B) solves the linear system Ax=B for x
        y = y + s
        rel_err = norm(s)
        f_eval += 2
        i += 1
    flag = (i == max_iter)
    return y, f_eval, flag


def bdf(f, Jf, t, y0):
    """
    Metodo BDF di ordine 3 con bootstrap iniziale tramite Heun3

    Input:
    - f: funzione del problema (f(t, y))
    - Jf: Jacobiana di f rispetto a y
    - t: array degli istanti temporali
    - y0: condizione iniziale (array 1D)

    Output:
    - y: array con la soluzione approssimata in ciascun istante
    - n_eval: numero totale di valutazioni della funzione f
    """
    alpha = np.array([1, -18/11, 9/11, -2/11])
    beta0 = 6/11

    m = len(y0)
    n = len(t)
    y = np.zeros((m, n))
    n_eval = 0

    t_prov = np.linspace(t[0], t[2], 3)
    y[:, :3], f_eval = heun3(f, y0, t_prov)
    n_eval += f_eval

    stor = y[:, :3]
    for i in range(3, n):
        h = t[i] - t[i - 1]
        w = np.zeros(m)
        for j in range(3):
            w += alpha[j + 1] * stor[:, 2 - j]
        y[:, i], eval_i, flag = newton_method_for_bdf(y[:, i - 1], w, f, beta0, t[i], h, Jf)
        if flag:
            print("Numero massimo di iterazioni raggiunto")
        stor = np.hstack((stor[:, 1:], y[:, i].reshape(-1, 1)))
        n_eval += eval_i

    return y, n_eval



################################ IRK per funzioni scalari

def newton_method_for_irk(f, u, t_start, t_end, A, c, J):
    """
    Risolve il sistema implicito per i metodi IRK tramite il metodo di Newton

    Input:
    - f: funzione del problema (f(t, y))
    - u: valore iniziale (scalare)
    - t_start, t_end: estremi temporali 
    - A, c: coefficienti di Butcher del metodo IRK
    - J: funzione Jacobiana df/dy

    Output:
    - K: stima dei valori intermedi degli stadi
    - n_itr: numero di valutazioni effettuate
    - flag: True se il metodo non converge entro max_it
    """
    h = t_end - t_start
    max_it = min(10_000, 100 * np.size(u))
    abstoll = 1e-5
    q = len(c)
    tau = t_start + h * c

    K = np.zeros(q)
    F = np.zeros(q)
    M = np.zeros((q, q))
    I = np.eye(q)

    rel_err = 1
    n_itr = 0
    it = 0

    while rel_err > abstoll and it < max_it:
        w = u + h * np.einsum('ij,j->i', A, K)
        for i in range(q):
            F[i] = f(tau[i], w[i])
            M[i, :] = I[i, :] - h * J(tau[i], w[i]) * A[i, :]
        s = solve(M, -K + F)
        K += s
        rel_err = norm(s)
        n_itr += 2 * q
        it += 1

    flag = (it == max_it)
    return K, n_itr, flag


def irk_scal_solver(f, u0, t, A, b, c, J):
    """
    Risolutore Runge-Kutta implicito per problemi scalari

    Input:
    - f: funzione del problema (f(t, y))
    - u0: valore iniziale (scalare)
    - t: array dei tempi
    - A, b, c: coefficienti del metodo IRK
    - J: funzione Jacobiana df/dy

    Output:
    - t: array dei tempi (inalterato)
    - u: array con la soluzione approssimata in ciascun tempo
    - f_eval: numero di valutazioni della funzione f
    """
    N = len(t)
    u = np.zeros(N)
    u[0] = u0
    f_eval = 0

    for k in range(1, N):
        K, n_itr, flag = newton_method_for_irk(f, u[k - 1], t[k - 1], t[k], A, c, J)
        if flag:
            print("Numero massimo di iterazioni raggiunte")
        Z = np.dot(b, K)  # somma pesata degli stadi
        u[k] = u[k - 1] + (t[k] - t[k - 1]) * Z
        f_eval += n_itr

    return u, f_eval


################################ IRK per sistemi di equazioni

def newton_method_for_irk_systems(f, u, t_start, t_end, A, c, J):
    """
    Metodo di Newton per sistemi IRK multidimensionali.

    Input:
    - f: funzione del problema (f(t, y))
    - u: valore iniziale (array 1D)
    - t_start, t_end: estremi temporali
    - A, c: coefficienti del metodo IRK
    - J: matrice Jacobiana df/dy valutata in (t_start, u)

    Output:
    - K: vettore concatenato degli stadi
    - n_itr: numero di valutazioni effettuate
    - flag: True se non converge
    """
    h = t_end - t_start
    m = len(u)
    q = len(c)
    tau = t_start + h * c

    max_it = min(int(1e4), 100 * m)
    abstoll = 1e-5

    K = np.zeros(m * q)
    M = np.eye(m * q) - h * np.kron(A, J)

    rel_err = 1
    n_itr = 0
    it = 0

    while rel_err > abstoll and it < max_it:
        K_mat = K.reshape((m, q))
        w = np.zeros((m, q))
        F = np.zeros((m, q))
        for i in range(q):
            w[:, i] = u + h * np.einsum("mj,j->m", K_mat, A[i, :])
            F[:, i] = f(tau[i], w[:, i])
        F_flat = F.reshape(m * q)
        s = solve(M, -K + F_flat)
        K += s
        rel_err = norm(s)
        n_itr += q
        it += 1

    flag = (it == max_it)
    return K, n_itr, flag


def irk_solver(f, u0, t, A, b, c, J):
    """
    Risolutore IRK per sistemi vettoriali di ODE.

    Input:
    - f: funzione del problema (f(t, y))
    - u0: condizione iniziale (array 1D)
    - t: array dei tempi
    - A, b, c: coefficienti del metodo IRK
    - J: funzione Jacobiana df/dy (t, y)

    Output:
    - t: array dei tempi
    - u: soluzione approssimata per ogni tempo
    - f_eval: numero di valutazioni della funzione f
    """

    N = len(t)
    d = len(u0)
    u = np.zeros((d, N))
    u[:, 0] = u0
    f_eval = 0

    for i in range(1, N):
        Jacobian = J(t[i - 1], u[:, i - 1])
        K, n_itr, flag = newton_method_for_irk_systems(f, u[:, i - 1], t[i - 1], t[i], A, c, Jacobian)
        if flag:
            print("Numero massimo di iterazioni raggiunto")
        K_matrix = K.reshape(d, len(c))
        u[:, i] = u[:, i - 1] + (t[i] - t[i - 1]) * np.einsum("ij,j->i", K_matrix, b)
        f_eval += n_itr

    return t, u, f_eval


################################ Eulero simplettico 

def harmonic_symplectic_euler(omega, y0, t):
    """
    Metodo di Eulero simplettico per il sistema armonico lineare.

    Input:
    - omega: frequenza del sistema armonico
    - y0: condizioni iniziali ([u0, v0])
    - t: array degli istanti temporali

    Output:
    - u: array 2xN con le soluzioni (posizione e velocità) in ogni istante
    - f_eval: numero di valutazioni di f
    """
    N = len(t)
    u = np.zeros((2, N))
    u[:, 0] = y0
    f_eval = 0
    for i in range(1, N):
        h = t[i] - t[i - 1]
        A = np.array([
            [1 - h**2 * omega**2, -h * omega**2],
            [h, 1]])
        u[:, i] = np.einsum("ij,j->i", A, u[:, i - 1])
        f_eval += 2
    return u, f_eval
