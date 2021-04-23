#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy

class DMP(ABC):
	
	def __init__(self, start=0, goal=1, τ=1, size=(2,10), dt=1/240, αz=50, αx=4, αyx=200, basis="gaussian", phase="exponential", phaseStopping=False):
		"""
		Parameters
		----------
		start, goal: scalar or 1D array
			starting and goal position, if they are both arrays, they must have the same length
		τ : scalar
			time scaling representing the duration of the motion
		size: tuple or ndarray
			if 2-element tuple, must contain positive integers (dimension, nb basis functions>1)
			if ndarray, ndim must be equal to 2 and shape[1]>1, represents the weights
		dt: float
			time step for integration, default is equal to that of pybullet
		αz, αx, αyx: scalar
			gains
		basis: str
			basis function to choose either 'gaussian' or 'mollifier'
		phase: str
			phase decay, either 'exponential' or 'linear'
		"""
		if basis not in {"gaussian", "mollifier"}:
			raise ValueError("basis must be a str, either 'gaussian', 'mollifier'")
		elif phase not in {"exponential", "linear"}:
			raise ValueError("phase must be a str, either 'exponential', 'linear'")
		elif (np.ndim(size)==1 and (len(size)!=2 or size[1]<2)) or (np.ndim(size)==2 and np.shape(size)[1]<2) or np.ndim(size) not in {1,2}:
			raise ValueError("size must be a positive int 2-element array (dimension, nb basis>1) or an array of weights")
		elif (isinstance(start, (list, tuple, np.ndarray)) and isinstance(start, (list, tuple, np.ndarray)) and np.shape(start)!=np.shape(goal)):
			raise ValueError("start and goal have inconsistent dimensions")
		
		self.start, self.goal, self.τ, self.dt, self.αz, self.αx, self.αyx, self.phaseStopping = start, goal, τ, dt, αz, αx, αyx, phaseStopping
		self.basis = getattr(self, basis)
		self.phase = getattr(self, phase)
		self.w = np.zeros(size) if isinstance(size,tuple) else np.copy(size) # weights
		self.βz = αz/4
		d, N = self.w.shape # nb basis functions
		self.position = np.ones(d)*start if np.isscalar(start) else deepcopy(start)
		self.velocity = np.zeros(self.w.shape[0])
		self.lastPosition = deepcopy(self.position)
		
		self.x = 1 # phase variable
		self.c = np.exp(-self.αx*np.arange(N)/(N-1)) # centers
		self.h = 1/np.diff(self.c) # widths
		if basis=="gaussian":
			self.h = np.square(np.hstack([self.h, self.h[-1]])) # repeat the last
		elif basis=="mollifier":
			self.h = np.hstack([self.h[0], self.h]) # repeat the first
		self.distance = lambda a,b : np.linalg.norm(a-b)
		
		
	def run(self, nstep):
		return np.vstack([np.hstack(self.step()) for i in range(nstep)])
		
	def exponential(self, x=None, ỹ=None, y=None, dt=None):
		x, ỹ, y, dt = self.x if x is None else x, ỹ or self.lastPosition if ỹ is None else ỹ, self.position if y is None else y, self.dt if dt is None else dt
		if np.isscalar(x): # return the value of the phase variable at the next time step
			return x*np.exp(-self.αx/self.τ*dt/(1+self.αyx*self.distance(self.position, ỹ)))
		# otherwise we suppose time steps are give with np.arange, we return the phase variable array without phase stopping
		return np.exp(-self.αx/self.τ*dt*x) # consider x as t
		
	def linear(self, x=None, ỹ=None, y=None, dt=None):
		x, ỹ, y, dt = self.x if x is None else x, ỹ or self.lastPosition if ỹ is None else ỹ, self.position if y is None else y, self.dt if dt is None else dt
		if np.isscalar(x):
			return np.max(0, self.x-dt/(self.τ*(1+self.αyx*self.distance(self.position, ỹ))))
		return np.maximum(1-(x*dt)/self.τ, 0) # consider x as t
	
	def gaussian(self, x):
		return np.exp(-self.h*np.square(x-self.c))
	
	def mollifier(self, x):
		v = np.abs(self.h*(x-self.c))
		return np.where(v<1, np.exp(-1/(1-np.square(v))), 0)
		
	def forcing(self, x):
		φ = self.basis(x)
		return np.sum(self.w*φ, axis=1)/np.sum(φ)*x if np.max(φ)>0 else 0
	

	
				
class MultiDimensionalDMP(DMP):
	def __init__(self, start=0, goal=1, τ=1, size=(2,10), dt=1/240, αz=50, αx=4, αyx=200, basis="gaussian", phase="exponential", phaseStopping=False):
		super().__init__(start=start, goal=goal, τ=τ, size=size, dt=dt, αz=αz, αx=αx, αyx=αyx, basis=basis, phase=phase, phaseStopping=phaseStopping)
		
	def step(self, ỹ=None):
		if ỹ is None:
			ỹ = self.lastPosition if self.phaseStopping else self.position
		"""# RK4 with second derivative https://fr.wikipedia.org/wiki/Méthodes_de_Runge-Kutta
		y, ẏ = self.position, self.velocity#self.trajectory[self.t,:,0], self.trajectory[self.t,:,1] # τẏ=z
		dt, dt2 = self.dt, self.dt/2
		k1 = self.ÿ(self.x,                       y,   ẏ)
		ŷ = y+dt2*ẏ
		k2 = self.ÿ(self.phase(ỹ=ỹ, y=ŷ, dt=dt2), ŷ,   ẏ+dt2*k1)
		ŷ = y+dt2*ẏ+dt2*dt2*k1
		k3 = self.ÿ(self.phase(ỹ=ỹ, y=ŷ, dt=dt2), ŷ,   ẏ+dt2*k2)
		ŷ = y+dt*(ẏ+dt2*k2)
		k4 = self.ÿ(self.phase(ỹ=ỹ, y=ŷ, dt=dt),  ŷ,   ẏ+dt*k3)
		self.position += dt*self.velocity + dt*dt*(k1+k1+k3)/6
		self.velocity += dt*(k1+2*(k2+k3)+k4)/6"""
		self.velocity += self.ÿ(self.x, self.position, self.velocity)*self.dt
		self.position += self.velocity*self.dt
		self.lastPosition = ỹ
		self.x = self.phase(ỹ=ỹ)
		return self.position, self.velocity
		
	def ÿ(self, x, y, ẏ):
		# divide by τ² because there is a double integration with τ -> the output is unscaled
		# (1-x) avoids high acceleration at the beginning
		αz, βz, τ, g, s, f = self.αz, self.βz, self.τ, self.goal, self.start, self.forcing(x)
		return (αz*(βz*(g-y-(g-s)*x+f)-ẏ*τ)) / (τ*τ)
		#return (αz*(βz*(g-y)-ẏ*τ)+f) / (τ*τ)
		
	def batchRegresion(self, trajectory):
		"""
		Learn a trajectory with batch regression, but doesn't work well
		The trajectory is supposed to be τ seconds long and the step time can be deduced from the length
		An incremental regression with forgetting factor exists as well
		"""
		dt = self.τ/len(trajectory)
		t = np.arange(trajectory.shape[0])+1 # time (in time step) plus an ε to avoid zero division
		αz, βz, τ, g, s, x = self.αz, self.βz, self.τ, self.goal, self.start, self.phase(t, dt=dt)
		Γ = self.basis(x[:,None]).T # (nb basis functions, times)
		ẏ = np.diff(trajectory, axis=0)/dt
		ẏ = np.vstack([ẏ, ẏ[-1]]) # (times, dimension)
		ÿ = np.diff(ẏ, axis=0)/dt
		ÿ = np.vstack([ÿ, ÿ[-1]])
		#f = (self.τ*self.τ*ÿ - self.αz*(self.βz*(self.goal - trajectory) - self.τ*ẏ)) # (times, dimension)
		f = (ÿ*τ*τ/αz + ẏ*τ)/βz - (g-trajectory-(g-s)*x[:,None])
		self.w = np.dot(Γ*x, f).T / np.dot(Γ*x, x) # (dimensions, nb basis function)
		""" simpler equivalent but slower
		for i, fd in enumerate(f.T):
			for j, γ in enumerate(Γ):
				self.w[i,j] = (s.T @ np.diag(γ) @ fd) / (s.T @ np.diag(γ) @ s)"""
				
from pyquaternion import Quaternion
class quaternionDMP(DMP):
	
	def __init__(self, start=Quaternion(1,0,0,0), goal=Quaternion(0,1,0,0), τ=1, weights=10, dt=1/240, αz=50, αx=4, αyx=200, basis="gaussian", phase="exponential", phaseStopping=False):
		if np.isscalar(weights):
			weights = (3,weights)
		elif weights.shape[0]==3:
			weights = weights.copy()
		else:
			raise ValueError("the weights must be a positive int or or ndarray such as weights.shape[0]==3")
		super().__init__(start=Quaternion(start), goal=Quaternion(goal), τ=τ, size=weights, dt=dt, αz=αz, αx=αx, αyx=αyx, basis=basis, phase=phase, phaseStopping=phaseStopping)
		self.distance = Quaternion.distance
		
		
	def step(self, ỹ=None, x=None):
		if ỹ is None:
			ỹ = self.lastPosition if self.phaseStopping else self.position
		αz, βz, τ, g, s, x, f, q, η = self.αz, self.βz, self.τ, self.goal, self.start, self.x if x is None else x, self.forcing(self.x), self.position, self.velocity*self.τ
		#self.position.integrate(self.velocity + self.dt*(k1+k1+k3)/6, self.dt)
		self.position = Quaternion.exp(self.dt/2*Quaternion(vector=η)/self.τ)*q
		self.velocity += (αz*(βz*(2*Quaternion.log(g*q.conjugate).vector-2*Quaternion.log(g*s.conjugate).vector*x+f)-η)) / (τ*τ) * self.dt # basic euler
		self.x = self.phase(self.x if x is None else x, ỹ=ỹ)
		self.lastPosition = ỹ
		return self.position.elements, self.velocity
		
class PoseDMP(MultiDimensionalDMP):
	"""3D+quaternion"""
	def __init__(self, start=0, goal=1, weights=10, startQuaternion=Quaternion(1,0,0,0), goalQuaternion=Quaternion(0,1,0,0), weightsQuaternion=10, τ=1, size=10, dt=1/240, αz=50, αx=4, αyx=200, basis="gaussian", phase="exponential", phaseStopping=False):
		if np.isscalar(weights): size = (3,weights)
		elif weights.shape[0]==3: size = weights.copy()
		else: raise ValueError(f"the weights must be a positive int or or ndarray such as weights.shape[0]==3, weights={weights}")
		if np.isscalar(weights): sizeQuaternion = weightsQuaternion
		elif weights.shape[0]==3: sizeQuaternion = weights.copy()
		else: raise ValueError(f"the weightsQuaternion must be a positive int or or ndarray such as weights.shape[0]==3, weightsQuaternion={weightsQuaternion}")

		super().__init__(start=start, goal=goal, τ=τ, size=size, dt=dt, αz=αz, αx=αx, αyx=αyx, basis=basis, phase=phase, phaseStopping=phaseStopping)
		self.qdmp = quaternionDMP(start=startQuaternion, goal=goalQuaternion, τ=τ, weights=sizeQuaternion, dt=dt, αz=αz, αx=αx, αyx=αyx, basis=basis, phase=phase, phaseStopping=phaseStopping)
		
	def step(self, ỹ=None):
		if ỹ is not None:
			assert len(ỹ)==7, "ỹ must be a concatenation of 3D position and quaternion"
			current, currentQuaternion = ỹ[:3], ỹ[3:]
		else: current, currentQuaternion = None, None
		position, velocity = super().step(current)
		positionQuaternion, velocityQuaternion = self.qdmp.step(currentQuaternion, self.x) # the phase stopping depends on the 3D position error only, not the quaternion
		return position, positionQuaternion, velocity, velocityQuaternion

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	T = 240
	#dmp = MultiDimensionalDMP(goal=0, τ=1, basis="gaussian", phase="exponential")#, size=np.random.rand(2,20)*1e1)
	#dmp = quaternionDMP(τ=1, weights=10, basis="gaussian", phase="exponential")
	dmp = PoseDMP(goal=1, τ=1, weights=np.random.rand(3,10))#, phaseStopping=True)
	
	targetTrajectory = np.exp(-(np.square(np.linspace([-1,-1], [1,2], T))-0)*100)
	#dmp.batchRegresion(targetTrajectory)
	trajectory = dmp.run(T)
	print(trajectory.shape, dmp.x)
	
	fig, ax = plt.subplots(2)
	ax[0].plot(trajectory[:,0], label="x")
	ax[0].plot(trajectory[:,1], label="y")
	#ax[0].plot(targetTrajectory, label="targets")
	ax[1].plot(trajectory[:,2])
	ax[1].plot(trajectory[:,3])
	ax[1].set_xlabel("t")
	ax[0].legend()
	fig.suptitle("DMP")
	plt.show()
