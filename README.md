# FFJORD: Free-Form Continuous Normalizing Flow from Scratch

An implementation of **FFJORD (Free-Form Jacobian of Reversible Dynamics)** built entirely from mathematical primitives and foundational PyTorch deep learning blocks. This project bypasses high-level wrappers to construct Continuous Normalizing Flows (CNFs) via Neural Ordinary Differential Equations (Neural ODEs), optimizing generative density estimation on the MNIST dataset.

Project scope was an implementation of the **Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2018). FFJORD: Free-Form Continuous Dynamics for Scalable Reversible Generative Models.** research paper. 

## 🚀 Key Milestones & Architecture

* **From-Scratch Foundations:** Developed custom explicit Runge-Kutta 4th Order (**RK4**) and adaptive **dopri5** (Dormand-Prince) Ordinary Differential Equation solvers to handle continuous time state transformations.
* **Efficient Trace Estimation:** Implemented the **Hutchinson Trace Estimator** using Rademacher/Gaussian noise to accurately approximate the divergence of the vector field, avoiding the costly $O(D^3)$ exact Jacobian calculation.
* **Proven Convergence:** Achieved an outstanding **99.95% reduction in Negative Log-Likelihood (NLL) loss** over 100 training epochs on MNIST.
* **Robust Numerical Engineering:** Successfully resolved complex training hurdles including structural ODE instabilities, GPU memory allocation overflows, and gradient leaks during adjoint state propagation.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.x
* **Core Framework:** PyTorch
* **Differential Equations:** `torchdiffeq` (for benchmarking/integration validation)
* **Environment:** Designed for seamless deployment on Google Colab / GPU runtimes

---

## 📐 Core Methodology

Continuous Normalizing Flows map a simple base distribution (e.g., standard Gaussian) to a complex data distribution by integrating a vector field defined by a neural network:

$$\frac{\partial z(t)}{\partial t} = f(z(t), t, \theta)$$

The change in log-density follows the instantaneous change of variables formula:

$$\log p(z(t_1)) = \log p(z(t_0)) - \int_{t_0}^{t_1} \text{Tr}\left( \frac{\partial f}{\partial z(t)} \right) dt$$

Using **Hutchinson's trace trick**, we rewrite the trace as an expectation over a noise distribution $\epsilon \sim p(\epsilon)$:

$$\text{Tr}\left( \frac{\partial f}{\partial z} \right) = \mathbb{E}_{\epsilon} \left[ \epsilon^T \frac{\partial f}{\partial z} \epsilon \right]$$

This reduces the computational complexity of the generative step drastically, allowing for free-form neural network architectures.

---

## 💻 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/lindashalash/ffjord-model-implementation.git](https://github.com/lindashalash/ffjord-model-implementation.git)
cd ffjord-model-implementation
```

### 2. Clone the Repository
For the full line-by-line implementation and project report, refer to the **Project_Report.pdf** file in this repository!