#pj_Library.py 
#Name: Prajwal Pravin Sahare Roll no.: 2411016

import math
import matplotlib.pyplot as plt

class MyComplex:
    def __init__(self, real=0.0, imag=0.0):
        self.r = float(real)
        self.i = float(imag)

    # Operator overloading
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return MyComplex(self.r + other, self.i)
        return MyComplex(self.r + other.r, self.i + other.i)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return MyComplex(self.r - other, self.i)
        return MyComplex(self.r - other.r, self.i - other.i)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return MyComplex(other - self.r, -self.i)
        return MyComplex(other.r - self.r, other.i - self.i)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return MyComplex(self.r * other, self.i * other)
        real = self.r * other.r - self.i * other.i
        imag = self.r * other.i + self.i * other.r
        return MyComplex(real, imag)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return MyComplex(self.r / other, self.i / other)
        denom = other.r**2 + other.i**2
        if denom == 0.0:
            raise ZeroDivisionError("Complex division by zero.")
        real = (self.r * other.r + self.i * other.i) / denom
        imag = (self.i * other.r - self.r * other.i) / denom
        return MyComplex(real, imag)

    # Modulus / Absolute value
    def __abs__(self):
        return math.hypot(self.r, self.i)

    def conjugate(self):
        return MyComplex(self.r, -self.i)

    # String representations
    def __repr__(self):
        sign = "+" if self.i >= 0 else "-"
        return f"MyComplex({self.r}, {self.i})"

    def __str__(self):
        sign = "+" if self.i >= 0 else "-"
        return f"{self.r} {sign} {abs(self.i)}i"


#LCG random number generator 
def pj_LCG(x_i, a, c, m, N):
  random_numbers_lcg = []
  for _ in range(N):
    x_i = (a * x_i + c) % m
    random_numbers_lcg.append(x_i / m)
  return random_numbers_lcg

#LCG random number generator plot correlation 
def plot_correlation_pj_lcg(random_numbers_lcg, p):
  x_i = random_numbers_lcg[:-p]
  x_i_plus_p = random_numbers_lcg[p:]

  plt.scatter(x_i, x_i_plus_p)
  plt.xlabel('x_i')
  plt.ylabel('x_i+' + str(p))
  plt.title('Correlation between x_i and x_i+' + str(p))
  plt.show()


#LCG random number generator(Only one at a time)
def myrand(low=0, high=1, seed=None):
 if not hasattr(myrand, "x0"):
   myrand.x0 = 0
 if seed is not None:
   myrand.x0 = seed  
 a = 1103515245
 c = 12345
 m = 2**31
 x1 = ( a * myrand . x0 + c ) % m
 myrand . x0 = x1
 rn = x1 / m # converting to [0.0 , 1.0]
 return low + (high - low) * rn


#Gauss-Jordan Elimination using partial pivoting
def gje_16(A,B):

  aug_matrix = []
  n = len(A)

  #augmented matrix creation
  for i in range(n):
   aug_matrix.append(A[i]+[B[i]])

  #partial pivoting 
  for i in range(n):
    max_row = i
    for j in range(i+1, n):
      if abs(aug_matrix[j][i]) > abs(aug_matrix[max_row][i]):
        max_row = j
    aug_matrix[i], aug_matrix[max_row] = aug_matrix[max_row], aug_matrix[i]

    pivot = aug_matrix[i][i]
    for j in range(i, n+1):
     aug_matrix[i][j] /= pivot

    for k in range(n):
     if k != i:
       factor = aug_matrix[k][i]
       for j in range(i, n+1):
        aug_matrix[k][j] -= factor * aug_matrix[i][j]

  solution = [aug_matrix[row][n] for row in range(n)]
  return solution


#LU Decomposition of a matrix
def LU_doolittle(A):

 for i in range(len(A)):
        for j in range(i, len(A)):
            A[i][j] -= sum(A[i][k] * A[k][j] for k in range(i))

        # 2. Compute elements of L in column i (for j > i)
        for j in range(i + 1, len(A)):
            A[j][i] = (A[j][i] - sum(A[j][k] * A[k][i] for k in range(i))) / A[i][i]

 return A


#Verification of symmetric matrix and Cholesky decomposition of a matrix
def ch_decomp(A):
    
    # Checking if the matrix is symmetric
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i][j] != A[j][i]:
                raise ValueError("Matrix is not symmetric")

    # Cholesky decomposition
    for i in range(len(A)):
        for j in range(i + 1):
            sum_ = sum(A[i][k] * A[j][k] for k in range(j))
            if i == j:
                 val = A[i][i] - sum_
                 if val <= 1e-12:
                    raise ValueError("Matrix is not positive-definite")
                 A[i][j] = val**0.5               
            else:
                A[i][j] = (A[i][j] - sum_) / A[j][j]
        for j in range(i + 1, len(A)):
            A[i][j] = 0.0        
    return A

#froward-backward substitution    
def for_back_sub(LU, b,method="choleskey"):
    n = len(LU)
    y = [0] * n
    x = [0] * n

    # Forward substitution to solve Ly = b
    for i in range(n):
       sum_l = sum(LU[i][j] * y[j] for j in range(i))
       if method.upper() == "LU":
        y[i] = (b[i] - sum_l) 
       else:
        y[i] = (b[i] - sum_l) / LU[i][i]

    #backward substitution to solve Ux = y
    for i in range(n - 1, -1, -1):
        if method.upper() == "LU":
         sum_u = sum(LU[i][j] * x[j] for j in range(i + 1, n))
        else:
         sum_u = sum(LU[j][i] * x[j] for j in range(i + 1, n)) 
        val = (y[i] - sum_u)/ LU[i][i]
        x[i] = 0.0 if abs(val) < 1e-12 else val 
    return x          


#Jacobi iterative method
def jacobi_pj(A, b, tol = 1e-6, max_iterations = 500):
   n=len(A)

  #check if diagonal elements are non-zero and swap rows if necessary
   for i in range(n):
    if A[i][i] == 0:
      for j in range(i + 1, n):
        if A[j][i] != 0:
         A[i], A[j] = A[j], A[i]
         b[i], b[j] = b[j], b[i]
        break
    else:
      continue  
  #initial guess and main function
   x = [0] * n
   for k in range(max_iterations):
    x_new = [0] * n
    for i in range(n):
      sum_ax = sum(A[i][j] * x[j] for j in range(n) if j != i)
      x_new[i] = (b[i] - sum_ax) / A[i][i]
    if max(abs(x_new[i] - x[i]) for i in range(n)) < tol:
      return x_new, k + 1
    x = x_new   
   return x, max_iterations     
    


#Gauss-seidel iterative method with successive over-relaxation factor
def gauss_seidel_pj(A, b, tol=1e-6, max_iterations=500, omega=1.0):
    n = len(A)
    x = [0] * n  # Initial guess

    for k in range(max_iterations):
        x_new = x.copy()
        for i in range(n):
            sum1 = sum(A[i][j] * x_new[j] for j in range(i))
            sum2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] =(1-omega) * x[i] + omega * (b[i] - sum1 - sum2) / A[i][i]
        if max(abs(x_new[i] - x[i]) for i in range(n)) < tol:
            return x_new, k + 1
        x = x_new.copy()
    return x, max_iterations
    


#Bisection method for root finding
def bisection_pj(func, a, b, tol=1e-6, max_iterations=100):
    while func(a) * func(b) >= 0:
        if abs(func(a))< abs(func(b)):
            a = a - (b - a) / 3 
        else:
            b = b + (b - a) / 3
              
    for iteration in range(max_iterations):
         c = (a + b) / 2
         if abs(func(c)) < tol or (b - a) / 2 < tol:
            return c, iteration + 1
         
         if func(c) * func(a) < 0:
            b = c
         else:
            a = c
    return (a + b) / 2, max_iterations
   

#Regula Falsi method for root finding
def regula_falsi_pj(func, a, b, tol=1e-6, max_iterations=100):
    while func(a) * func(b) >= 0:
        if abs(func(a)) < abs(func(b)):
            a = a - (b - a) / 3
        else:
            b = b + (b - a) / 3
    interval = (a, b)        

    for iteration in range(max_iterations):
        c = (a * func(b) - b * func(a)) / (func(b) - func(a))
        if abs(func(c)) < tol or abs(b - a) < tol:
            return c, iteration + 1, interval

        if func(c) * func(a) < 0:
            b = c
        else:
            a = c
    return (a * func(b) - b * func(a)) / (func(b) - func(a)), max_iterations, interval    

   
#Newton-Raphson method for root finding
def newton_raphson_pj(func, d_func, x0, tol=1e-6, max_iterations=100):
    x = x0
    for iteration in range(max_iterations):
        f_x = func(x)
        df_x = d_func(x)
        x_new = x - f_x / df_x
        if abs(x_new - x) < tol:
            return x_new, iteration + 1
        x = x_new
    return x, max_iterations

#Fixed point method for root finding
def fixed_point_pj(func, x0, tol=1e-6, max_iterations=100):
    x = x0
    for iteration in range(max_iterations):
        x_new = func(x)
        if abs(x_new - x) < tol:
            return x_new, iteration + 1
        x = x_new
    return x, max_iterations

#defining polynomial and its derivatives for laguerre's method
def polynomial_pj(coeffs, x):
    n = len(coeffs) - 1
    return sum(c * (x ** (n - i)) for i, c in enumerate(coeffs))

def d_polynomial_pj(coeffs,x):
   n = len(coeffs) - 1
   return sum(c * (n - i) * (x ** (n - i - 1)) for i, c in enumerate(coeffs) if (n - i) > 0)

def d2_polynomial_pj(coeffs,x):
    n = len(coeffs) - 1
    return sum(c * (n - i) * (n - i - 1) * (x ** (n - i - 2)) for i, c in enumerate(coeffs) if (n - i) > 1)

#laguerre's method for root finding
def laguerre(coeffs, guess=0.0, epsilon=1e-7, max_iter=100):
    n = len(coeffs) - 1
    x = guess

    for _ in range(max_iter):
        f_x = polynomial_pj(coeffs, x)
        if abs(f_x) < epsilon:
            return x

        f_prime_x = d_polynomial_pj(coeffs, x)
        f_double_prime_x = d2_polynomial_pj(coeffs, x)

        G = f_prime_x / f_x
        H = G * G - f_double_prime_x / f_x
        denom1 = G + (((n - 1) * (n * H - G * G)) ** 0.5)
        denom2 = G - (((n - 1) * (n * H - G * G)) ** 0.5)

        if abs(denom1) > abs(denom2):
            a_n = n / denom1
        else:
            a_n = n / denom2

        x -= a_n

    return x, max_iter

#synthetic division for deflation of polynomial
def deflate(coeffs, root):
    new_coeffs = [coeffs[0]]
    for i in range(1, len(coeffs) - 1):
        new_coeffs.append(coeffs[i] + root * new_coeffs[-1])
    return new_coeffs

#integration by midpoint rule
def midpoint_pj(func, a, b, n):
    h = (b - a) / n
    total = 0.0
    for i in range(1, n + 1):
        mid = ((a + (i-1)*h) + (a + i*h)) / 2
        total += func(mid)
    return total * h

#integration by trapezoidal rule
def trapezoidal_pj(func, a, b, n):
    h = (b - a) / n
    total = 0.5 * (func(a) + func(b))
    for i in range(1, n):
        total += func(a + i * h)
    return total * h

#integration by simpson's rule
def simpson_pj(func, a, b, n):
   h = (b - a) / n
   total = func(a) + func(b)
   for i in range(1, n):
     x_i = a + i * h 
     if i % 2 == 0:
         total += 2 * func(x_i)
     else:
        total += 4 * func(x_i)
   return total * h / 3         

#integration by monte carlo method
def monte_carlo_pj(func, a, b, n, seed=None):
    sum_f = 0.0
    sum_f_sq = 0.0
    for i in range(n):
        s = seed if i == 0 else None
        x_i = myrand(a, b, seed=s)
        f_x = func(x_i)
        sum_f += f_x
        sum_f_sq += f_x ** 2
    mean_f = (sum_f * (b - a)) / n
    sigma_f_sq = (sum_f_sq / n) - (sum_f / n)**2
    return mean_f, (sigma_f_sq)**0.5

#Tables for Gauss quadrature
#Laguerre quadrature nodes and weights
def gauss_laguerre_nodes_weights(N):
    if N == 1:
        return [1.0], [1.0]
    elif N == 2:
        return [0.585786438, 3.414213562], [0.8535, 0.1464]
    elif N == 3:
        return [0.41577456, 2.294280360, 6.289945082], [0.7111, 0.2785, 0.0104]
    elif N == 4:
        return [0.322547689, 1.745761101, 4.536620297, 9.395070912], [0.6032, 0.3574, 0.0389, 0.0005]
    elif N == 5:
        return [0.263560320, 1.413403059, 3.596425771, 7.085810006, 12.640800844], [0.5217, 0.3987, 0.0759, 0.0036, 0.00002]
    elif N == 6:
        return [0.222846604, 1.188932102, 2.992736326, 5.775143569, 9.837467418, 15.982873981], [0.4590, 0.4170, 0.1134, 0.0104, 0.0003, 0.000002]
    else:
        raise ValueError("N must be between 1 and 6")

#Legendre quadrature nodes and weights
def gauss_legendre_nodes_weights(N):
   if N == 1:
      return [0.0], [2.0]
   elif N == 2:
      return [-0.577350269, 0.577350269], [1.0, 1.0]
   elif N == 3:
      return [-0.774596669, 0.0, 0.774596669], [0.5556, 0.8889, 0.5556]
   elif N == 4:
      return [-0.861136312, -0.339981044, 0.339981044, 0.861136312], [0.3478, 0.6521, 0.6521, 0.3478]
   elif N == 5:
      return [-0.906179846, -0.538469310, 0.0, 0.538469310, 0.906179846], [0.236, 0.4786, 0.5689, 0.4786, 0.2369]
   elif N == 6:
      return [-0.932469514, -0.661209386, -0.238619186, 0.238619186, 0.661209386, 0.932469514], [0.1713, 0.3608, 0.4679, 0.4679, 0.3608, 0.1713]
   else:
      raise ValueError("N must be between 1 and 6")



#integration by gauss quadrature method
def gauss_quadrature_pj(func, a = None, b = None, N = None, quadrature = "gauss_legendre"):
   if quadrature == "gauss_legendre":
     nodes, weights = gauss_legendre_nodes_weights(N)
     integral = 0.0
     for i in range(N):
        x_i = 0.5*(b-a)*nodes[i] + 0.5*(b+a)
        integral += 0.5*(b-a)*weights[i] * func(x_i)
     return integral
   elif quadrature == "gauss_laguerre":
     nodes, weights = gauss_laguerre_nodes_weights(N)
     integral = 0.0
     for i in range(N):
        integral += weights[i] * func(nodes[i])
     return integral


#eigen value problem using power iteration method
def power_iteration_pj(A, num_iterations=1000, tol=1e-6, b_init=None):
    if len(A) != len(A[0]):
        raise ValueError("Matrix must be square")

    n = len(A)
    if b_init is not None:
        b_k = list(b_init)
    else:
        b_k = [float(i + 1) for i in range(n)]

    iterations_taken = 0

    for iteration in range(1, num_iterations + 1):
        iterations_taken = iteration

        # Matrix-vector multiplication A * b_k
        b_k1 = [0.0] * n
        for i in range(n):
            for j in range(n):
                b_k1[i] += A[i][j] * b_k[j]

        # Euclidean norm
        norm_b_k1 = math.sqrt(sum(x ** 2 for x in b_k1))
        if norm_b_k1 == 0:
            break
        b_k1 = [x / norm_b_k1 for x in b_k1]

        # Check convergence on vector differences (accounting for sign flips)
        diff_pos = all(abs(b_k1[i] - b_k[i]) < tol for i in range(n))
        diff_neg = all(abs(b_k1[i] + b_k[i]) < tol for i in range(n))
        if diff_pos or diff_neg:
            b_k = b_k1
            break

        b_k = b_k1

    # Rayleigh quotient
    eigenvalue = sum(b_k[i] * sum(A[i][j] * b_k[j] for j in range(n)) for i in range(n))
    return eigenvalue, b_k, iterations_taken

#Lagrange's interpolation method 
def lagrange_interpolation_pj(x_values, y_values, x):
    n = len(x_values) - 1
    result = 0.0
    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if j != i:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        result += term
    return result  

#Least squares fitting method with pearson correlation coefficient
def least_squares_fit_pj(x_values, y_values, sigma_values=None):
    n = len(x_values)
    if n != len(y_values):
        raise ValueError("x_values and y_values must have the same length")
    if sigma_values is None:
        sigma_values = [1.0] * n
    error = [1.0 / (sigma ** 2) for sigma in sigma_values]
    sum_error = sum(error)    
    sum_x = sum(x*i for x, i in zip(x_values, error))
    sum_y = sum(y*i for y, i in zip(y_values, error))
    sum_xy = sum(x * y * i for x, y, i in zip(x_values, y_values, error))
    sum_xx = sum((x ** 2)*i for x, i in zip(x_values, error))
    sum_yy = sum((y ** 2)*i for y, i in zip(y_values, error))

    denominator = sum_error * sum_xx - sum_x ** 2
    if denominator == 0:
        raise ValueError("Denominator in slope calculation is zero")
    
    #Parameters of the best fit line y = a1 + a2*x
    a2 = (sum_error * sum_xy - sum_x * sum_y) / denominator
    a1 = (sum_xx*sum_y - sum_x*sum_xy) / denominator

    #Error in the parameters
    sigma2_a1 = sum_xx / denominator
    sigma2_a2 = sum_error / denominator

    # Pearson correlation coefficient
    r_numerator = sum_xy ** 2
    r_denominator = sum_xx * sum_yy
    if r_denominator == 0:
        raise ValueError("Denominator in correlation coefficient calculation is zero")
    
    r = r_numerator / r_denominator

    return a1, a2, sigma2_a1, sigma2_a2, r  