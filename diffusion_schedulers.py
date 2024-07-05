"""Schedulers for Denoising Diffusion Probabilistic Models"""

import math

import numpy as np
import torch
"""定义扩散模型的调度器,扩散调度器负责控制在扩散过程中的噪声添加和去除步骤。
  高斯扩散 (GaussianDiffusion)：实现了基于高斯噪声的扩散过程。
  分类扩散 (CategoricalDiffusion)：实现了基于分类噪声的扩散过程。
  推理调度 (InferenceSchedule)：用于在推理阶段控制扩散步骤的调度。
"""


class GaussianDiffusion(object):
  """Gaussian Diffusion process with linear beta scheduling"""

  def __init__(self, T, schedule):
    # Diffusion steps
    self.T = T

    # Noise schedule
    if schedule == 'linear':
      b0 = 1e-4
      bT = 2e-2
      self.beta = np.linspace(b0, bT, T)
    elif schedule == 'cosine':
      self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(
          0)  # Generate an extra alpha for bT
      self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)

    self.betabar = np.cumprod(self.beta)
    self.alpha = np.concatenate((np.array([1.0]), 1 - self.beta))
    self.alphabar = np.cumprod(self.alpha)
    
    print(f"Initialized GaussianDiffusion with T={self.T}, schedule={schedule}")
    print(f"betabar shape: {self.betabar.shape}")
    print(f"alpha shape: {self.alpha.shape}")
    print(f"alphabar shape: {self.alphabar.shape}")


  def __cos_noise(self, t):
    offset = 0.008
    return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

  def sample(self, x0, t):
    # Select noise scales
    noise_dims = (x0.shape[0],) + tuple((1 for _ in x0.shape[1:]))
    t_cpu = t.cpu().numpy().astype(np.int32)  # 确保 t 是整数
    atbar = torch.from_numpy(self.alphabar[t_cpu]).view(noise_dims).to(x0.device)
    assert len(atbar.shape) == len(x0.shape), 'Shape mismatch'

    # Sample noise and add to x0
    epsilon = torch.randn_like(x0)
    xt = torch.sqrt(atbar) * x0 + torch.sqrt(1.0 - atbar) * epsilon
    
    # 打印调试信息
    print(f"sample - x0 shape: {x0.shape}, t shape: {t.shape}, atbar shape: {atbar.shape}, xt shape: {xt.shape}, epsilon shape: {epsilon.shape}")

    return xt, epsilon
  
  def q_sample(self, x, t):
    t_cpu = t.cpu().numpy().astype(np.int32)  # 确保 t 是整数
    noise_dims = (x.shape[0],) + tuple((1 for _ in x.shape[1:]))
    atbar = torch.from_numpy(self.alphabar[t_cpu]).view(noise_dims).to(x.device)
    epsilon = torch.randn_like(x)
    xt = torch.sqrt(atbar) * x + torch.sqrt(1.0 - atbar) * epsilon
    print(f"q_sample - x shape: {x.shape}, t shape: {t.shape}, atbar shape: {atbar.shape}, xt shape: {xt.shape}")
    return xt, epsilon


class CategoricalDiffusion(object):
  """Gaussian Diffusion process with linear beta scheduling"""

  def __init__(self, T, schedule):
    # Diffusion steps
    self.T = T

    # Noise schedule
    if schedule == 'linear':
      b0 = 1e-4
      bT = 2e-2
      self.beta = np.linspace(b0, bT, T)
    elif schedule == 'cosine':
      self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(
          0)  # Generate an extra alpha for bT
      self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)

    beta = self.beta.reshape((-1, 1, 1))
    eye = np.eye(2).reshape((1, 2, 2))
    ones = np.ones((2, 2)).reshape((1, 2, 2))

    self.Qs = (1 - beta) * eye + (beta / 2) * ones

    Q_bar = [np.eye(2)]
    for Q in self.Qs:
      Q_bar.append(Q_bar[-1] @ Q)
    self.Q_bar = np.stack(Q_bar, axis=0)
    
    print(f"Initialized CategoricalDiffusion with T={self.T}, schedule={schedule}")
    print(f"Qs shape: {self.Qs.shape}")
    print(f"Q_bar shape: {self.Q_bar.shape}")


  def __cos_noise(self, t):
    offset = 0.008
    return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

  # def sample(self, x0_onehot, t):
  #   # Select noise scales
  #   Q_bar = torch.from_numpy(self.Q_bar[t]).float().to(x0_onehot.device)
  #   xt = torch.matmul(x0_onehot, Q_bar.reshape((Q_bar.shape[0], 1, 2, 2)))
    
  #    # 打印调试信息
  #   print(f"sample - x0_onehot shape: {x0_onehot.shape}, t shape: {t.shape}, Q_bar shape: {Q_bar.shape}, xt shape: {xt.shape}")

  #   return torch.bernoulli(xt[..., 1].clamp(0, 1))
  
  def sample(self, x0_onehot, t):
    # 确保x0_onehot需要梯度
    x0_onehot = x0_onehot.detach().requires_grad_(True)
    
    # Select noise scales
    Q_bar = torch.from_numpy(self.Q_bar[t.cpu().numpy()]).float().to(x0_onehot.device)
    Q_bar = Q_bar.unsqueeze(0)  # Adding batch dimension
    
    # Print debug information
    print(f"sample - x0_onehot shape: {x0_onehot.shape}, t shape: {t.shape}, Q_bar initial shape: {Q_bar.shape}")
    
    try:
        # Perform the matrix multiplication
        xt = torch.matmul(x0_onehot.unsqueeze(1), Q_bar)  # [batch_size, num_edges, 2] x [1, num_edges, 2, 2] -> [batch_size, num_edges, 2, 2]
        xt = xt.squeeze(1)  # Removing the extra dimension added for matmul

        # Print debug information
        print(f"sample - x0_onehot shape after unsqueeze: {x0_onehot.shape}, Q_bar shape: {Q_bar.shape}, xt shape: {xt.shape}")

        return torch.bernoulli(xt[..., 1].clamp(0, 1))
    except RuntimeError as e:
        print(f"Error during matmul: {e}")
        print(f"x0_onehot shape: {x0_onehot.shape}, Q_bar shape: {Q_bar.shape}")
        return None



class InferenceSchedule(object):
  def __init__(self, inference_schedule="linear", T=1000, inference_T=1000):
    self.inference_schedule = inference_schedule
    self.T = T
    self.inference_T = inference_T
    
    # print(f"Initialized InferenceSchedule with schedule={inference_schedule}, T={self.T}, inference_T={self.inference_T}")

  def __call__(self, i):
    assert 0 <= i < self.inference_T

    if self.inference_schedule == "linear":
      t1 = self.T - int((float(i) / self.inference_T) * self.T)
      t1 = np.clip(t1, 1, self.T)

      t2 = self.T - int((float(i + 1) / self.inference_T) * self.T)
      t2 = np.clip(t2, 0, self.T - 1)
      
      # # 打印调试信息
      # print(f"InferenceSchedule call - i: {i}, t1: {t1}, t2: {t2}")

      return t1, t2
    
    elif self.inference_schedule == "cosine":
      t1 = self.T - int(
          np.sin((float(i) / self.inference_T) * np.pi / 2) * self.T)
      t1 = np.clip(t1, 1, self.T)

      t2 = self.T - int(
          np.sin((float(i + 1) / self.inference_T) * np.pi / 2) * self.T)
      t2 = np.clip(t2, 0, self.T - 1)
      
      # # 打印调试信息
      # print(f"InferenceSchedule call - i: {i}, t1: {t1}, t2: {t2}")

      return t1, t2
    else:
      raise ValueError("Unknown inference schedule: {}".format(self.inference_schedule))
