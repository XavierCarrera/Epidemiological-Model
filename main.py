import numpy as np
import matplotlib.pylab as plt
from scipy.special import erf
import pandas as pd

def rk4vec(t0, y0, dt, f):
    k1 = f(t0, y0)
    k2 = f(t0 + dt/2.0, y0 + dt * k1 / 2.0)
    k3 = f(t0 + dt/2.0, y0 + dt * k2 / 2.0)
    k4 = f(t0 + dt, y0 + dt * k3)
    y = y0 + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
    return y

class SIR_model():

  def __init__(self, init_condition, tmin = 0., tmax = 50., n = 10000, **params):
    self.tmin = tmin
    self.tmax = tmax
    self.n = n
    self.t = np.linspace(self.tmin, self.tmax, self.n)
    self.dt = self.t[1] - self.t[0]
    self.S = np.zeros([self.n])
    self.I = np.zeros([self.n])
    self.R = np.zeros([self.n])
    self.set_params(**params)
    self.init_condition = init_condition 

  def set_params(self, beta, miu):
    # Parameters definition for the model
    self.beta = beta
    self.miu =  miu

  def func(self, t, u):
    # ODE system
    self.uprime = np.zeros_like(u)
    self.uprime[0] = -self.beta*u[0]*u[1]
    self.uprime[1] = self.beta*u[0]*u[1]-self.miu*u[1]
    self.uprime[2] = self.miu*u[1]
    return self.uprime

  def run_solver(self):
      # Execution of the RK4 solution
      self.u0 = np.array(self.init_condition)
      self.u1 = np.zeros_like(self.u0)
      for i in range(self.n):
        self.S[i] = self.u0[0]
        self.I[i] = self.u0[1]
        self.R[i] = self.u0[2]
        self.u1 = rk4vec(self.t[i], self.u0, self.dt, self.func)
        self.u0 = np.copy(self.u1)

params = {'beta': 0.0005, 'miu': 0.1}
init_condition = [1000, 1, 0]
model = SIR_model(init_condition=init_condition, tmax=80, n = 50000, **params)
model.run_solver()
plt.figure(figsize=(7,7))
plt.plot(model.t, model.S, label = 'susceptibles', color = 'b')
plt.plot(model.t, model.I, label = 'infected', color = 'r')
plt.plot(model.t, model.R, label = 'recovered', color = 'g')
plt.fill_between(model.t, 0, model.I, color = 'r', alpha = 0.4)
plt.grid(True)
plt.legend(fontsize = 20)
plt.show()

plt.figure(figsize=(7,7))

def plot_solution(params, color = 'red', style = '-'):
    init_condition = [1000, 1, 0]
    model = SIR_model(init_condition=init_condition, tmax=80, n = 50000, **params)
    model.run_solver()
    plt.plot(model.t, model.I, linestyle= style, color = color, 
             label = r'$\beta =$ {}'.format(model.beta))
    plt.fill_between(model.t, 0, model.I, color = color, alpha = 0.5)

plot_solution(params = {'beta': 0.00055, 'miu': 0.1}, style = '--')
plt.plot(np.arange(0,80), 400*np.ones(len(np.arange(0,80))), '-k')
plt.text(x= 25, y = 410, s = 'Capacity of the Health system', fontsize = 15)
plot_solution(params = {'beta': 0.00035, 'miu': 0.1}, color = 'b')
plt.grid(True)
plt.legend()
plt.show()