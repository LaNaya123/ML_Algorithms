# -*- coding: utf-8 -*-
import torch
import torch.linalg as linalg
from torch.distributions import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt

def negative_log_grad(x, mean, cov):
    x = torch.from_numpy(x)
    mean = torch.from_numpy(mean)
    cov = torch.from_numpy(cov)
    cov_inv = linalg.inv(cov)

    grad = torch.matmul(cov_inv, (x - mean).unsqueeze(1)).squeeze(1).numpy()
    return grad

def negative_log_prob(x, mean, cov):
    x = torch.from_numpy(x)
    mean = torch.from_numpy(mean)
    cov = torch.from_numpy(cov)
    
    dist = MultivariateNormal(mean, cov)
    
    logp = dist.log_prob(x).item()
    
    return -logp
    
class MHSampler():
    def __init__(self, mean, cov, step_size=5, seed=None):
        self.mean = mean
        self.cov = cov
        self.step_size = step_size
        
        if seed:
            np.random.seed(seed)
    
    def sample(self, num_samples, x0):
        samples = np.zeros((num_samples,) + x0.shape)
       
        x = x0
        i = 0

        while i < num_samples:
            x_ = np.random.uniform(low = x - self.step_size,
                                   high = x + self.step_size,
                                   size = x.shape)
            
            log_prob1 = -negative_log_prob(x, self.mean, self.cov)
            log_prob2 = -negative_log_prob(x_, self.mean, self.cov)
            accept_rate = min(1., np.exp(log_prob2 - log_prob1))
            
            if np.random.random() > accept_rate:
                continue
            else:
                samples[i] = x_
                x = x_
                i += 1
                
        return samples
    
class GibbsSampler():
    def __init__(self, mean, cov, seed=None):
        self.mean = mean
        self.cov = cov
        
        if seed:
            np.random.seed(seed)
            
    def sample(self, num_samples, x0):
        samples = np.zeros((num_samples,) + x0.shape)
        
        x = x0
        
        for i in range(num_samples):
            for sampling_index in range(len(x0)):
                
                conditioned_index = 1 - sampling_index
                
                a = self.cov[sampling_index, sampling_index]
                b = self.cov[sampling_index, conditioned_index]
                c = self.cov[conditioned_index, conditioned_index]
                
                mu = self.mean[sampling_index] + b / c * (x[conditioned_index] - self.mean[conditioned_index])
                sigma = a - b**2 / c
                
                sample = np.random.randn() * sigma + mu
                
                x[sampling_index] = sample
                
            samples[i] = x
            
        return samples
   
class HMCSampler():
    def __init__(self, mean, cov, step_size=0.1, path_len=10, seed=None):
        self.mean = mean
        self.cov = cov
        self.step_size = step_size
        self.path_len = path_len
        
        self.num_vars = len(mean)
        self.mean_p = np.zeros(self.num_vars)
        self.cov_p = np.eye(self.num_vars)
        
        if seed:
            np.random.seed(seed)
    
    def sample(self, num_samples, x0):
        samples = np.zeros((num_samples,) + x0.shape)
        
        x = x0
        
        for i in range(num_samples):
            p = np.array([np.random.randn()] * self.num_vars)
            
            x_, p_ = self._leapfrog(x, p)
            
            h1 = negative_log_prob(x_, self.mean, self.cov) + negative_log_prob(p_, self.mean_p, self.cov_p)
                 
            h2 = negative_log_prob(x_, self.mean, self.cov) + negative_log_prob(p_, self.mean_p, self.cov_p)
            
            acc = min(1, np.exp(h1 - h2))
            
            if np.random.uniform(0, 1) <= acc:
                samples[i] = x_
                x = x_
            else:
                samples[i] = x
                
        return samples
            
    
    def _leapfrog(self, x, p):
        x, p = np.copy(x), np.copy(p)
        
        p -= self.step_size * negative_log_grad(x, self.mean, self.cov) / 2
        
        for _ in range(self.path_len-1):
            x += self.step_size * negative_log_grad(p, self.mean_p, self.cov_p)

            p -= self.step_size * negative_log_grad(x, self.mean, self.cov)
            
        x += self.step_size * negative_log_grad(p, self.mean_p, self.cov_p)
        
        p -= self.step_size * negative_log_grad(x, self.mean, self.cov) / 2
        
        return x, p 
        
            
if __name__ == "__main__":
    mean = np.array([5., 15.])
    cov = np.array([[5., 3.], [3., 5.]])
    
    x0 = np.array([0, 0], dtype=np.float64)
    
    mh = MHSampler(mean, cov)
    samples = mh.sample(5000, x0)
    plt.hist(samples[:, 0], bins=20)
    
    sampler = GibbsSampler(mean, cov)
    samples = sampler.sample(5000, x0)
    plt.hist(samples[:, 0], bins=30)
    
    hmc = HMCSampler(mean, cov)
    samples = hmc.sample(5000, x0)
    plt.hist(samples[:, 0], bins=30)