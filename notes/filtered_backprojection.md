**Fourier Slice Theorem**

To my understanding, given a 2D image $f(x,y)=f(\mathbf{x})$, the Radon transform $f(x,y) \mapsto g(\theta, s)$ is basically the collection of a wheel of 1D projection $p_\theta(s)=\int_{l_{(\theta, s)}}f\,dl$ where $g(\theta, s)=p_\theta(s)$.

  

We denote the Fourier transform of $f(\mathbf{x})$ as $\hat{f}(\mathbf{k})$ and of $p_\theta(s)$ as $\hat{p}_\theta(\sigma)$. The Fouriere slice theorem states that the 1D Fourier transform of a projection equals a central slice of the 2D Fourier transform of the image take along the same direction. Precisely, $$\hat{p}_\theta(\sigma) = \hat{f}(\sigma\cos\theta,\ \sigma\sin\theta) = \hat{f}(\sigma\,\hat{\mathbf{n}}_\theta)$$where $\hat{\mathbf{n}}_\theta = (\cos\theta,\sin\theta)$ is the unit vector along the projection direction, and $\sigma$ is the 1D frequency conjugate to $s$.

Proof:
1. $p_\theta(s)=\int_{l_{(\theta, s)}}f\,dl = \iint f(\mathbf{x})\delta(\mathbf{x}\cdot\hat{\mathbf{n}}_\theta - s)\,d\mathbf{x}$

2. Take the 1D FT in $s$: $$\hat{p}_\theta(\sigma)=\int p_\theta(s)\,e^{-i\sigma s}\,ds = \iint f(\mathbf{x})\left[\int \delta(\mathbf{x}\cdot\hat{\mathbf{n}}_\theta - s)\,e^{-i\sigma s}\,ds\right]d\mathbf{x}.$$
3. The inner integral collapses the delta: $\int\delta(\cdots - s)e^{-i\sigma s}ds = e^{-i\sigma(\mathbf{x}\cdot\hat{\mathbf{n}}_\theta)}$.

4. So $\hat{p}_\theta(\sigma)=\iint f(\mathbf{x})\,e^{-i(\sigma\hat{\mathbf{n}}_\theta)\cdot\mathbf{x}}\,d\mathbf{x} = \hat{f}(\sigma\hat{\mathbf{n}}_\theta)$



**Backprojection**

If we denote the Radon transform operator as $A$, it is clear that $$Af=\iint f(\mathbf{x})\delta(\mathbf{x}\cdot\hat{\mathbf{n}}_\theta - s)\,d\mathbf{x}=g'(\theta, s).$$ We can see that $\forall g$,  $$\langle Af, g\rangle = \int_0^{\pi}\int_{\mathbb R} \left[\iint f(\mathbf x)\,\delta(\mathbf x\cdot\hat{\mathbf n}_\theta - s)\,d\mathbf x\right] g(\theta,s)\, ds\, d\theta = \iint f(\mathbf x)\left[\int_0^{\pi} g(\theta,\ \mathbf x\cdot\hat{\mathbf n}_\theta)\, d\theta\right] d\mathbf x.$$
Observe that $\iint f(\mathbf x)\left[\int_0^{\pi} g(\theta,\ \mathbf x\cdot\hat{\mathbf n}_\theta)\, d\theta\right] d\mathbf x=\langle f, A^Tg\rangle$. Therefore, the adjoint $A^T$ satisfies:
$$\boxed{(A^T g)(\mathbf x) = \int_0^\pi g\big(\theta,\ \mathbf x\cdot\hat{\mathbf n}_\theta\big)\, d\theta}$$

Geometrically, to find the value of $A^T g$ at a point $\mathbf x$, for each angle $\theta$ pick out the projection value of the line that passes through $\mathbf x$, then sum (integrate) over all angles. That is exactly backprojection: smear each projection back along its lines and add up the angles.

**Filtered backprojection**

Leaving the proof aside, we get a clean result: $$(A^T A f)(\mathbf{x}) = (f * h)(\mathbf{x}), \qquad \hat{h}(\mathbf{k}) = \frac{2\pi}{|\mathbf{k}|}$$
$A^T A$ is a convolution whose Fourier multiplier is $1/|\mathbf{k}|$ (up to the $2\pi$ from my unnormalized FT convention). It is a radially-symmetric low-pass-ish smear: it over-weights low frequencies by exactly $1/|\mathbf{k}|$. Reconstructing the 2D inverse FT requires the area element $d\mathbf{k} = |\sigma|,d\sigma,d\theta$, but backprojection supplies only $d\sigma,d\theta$. The missing Jacobian factor $|\sigma| = |\mathbf{k}|$ is exactly what's _under_-counted, so $A^TA$ acquires the multiplier $1/|\mathbf{k}|$.

To invert, premultiply by the ramp filter $|\mathbf{k}|$ in Fourier space before backprojection. That "filter then backproject" is filtered backprojection

Intuitively, backprojection smears each line's value across the whole plane, dumping energy near the origin in Fourier space (every slice passes through $\mathbf{k}=0$). Low frequencies get counted by every angle; high frequencies by few. The ramp reweights to undo this $1/|\mathbf{k}|$ angular crowding.