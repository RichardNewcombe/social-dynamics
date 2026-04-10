# Spatial Memory Field

## Notation

| Symbol | Definition |
|--------|-----------|
| $N$ | Number of particles |
| $K$ | Number of preference dimensions per particle |
| $G$ | Grid resolution ($G \times G$ cells covering the domain) |
| $L$ | Domain size (toroidal, default $L = 1.0$) |
| $i$ | Particle index, $i \in \{1, \ldots, N\}$ |
| $d$ | Preference dimension index, $d \in \{1, \ldots, K\}$ |
| $x_i \in [0, L)^2$ | Position of particle $i$ |
| $p_i \in [-1, 1]^K$ | Preference (signal) vector of particle $i$ |
| $\tilde{p}_i$ | Effective preference after field modulation |
| $M(c, d)$ | Memory field value at grid cell $c$, dimension $d$ |
| $\text{cell}(x)$ | Grid cell containing position $x$: $\text{cell}(x) = (\lfloor x \cdot G/L \rfloor) \mod G$ |
| $\alpha$ | Field strength (how strongly the field modulates preferences) |
| $\beta$ | Write rate (how fast particles deposit into the field) |
| $\gamma$ | Decay factor per step ($0 < \gamma \leq 1$) |
| $\sigma$ | Gaussian blur standard deviation (in grid cells) |
| $s$ | Social learning rate (positive = conformity, negative = differentiation) |

## Overview

A persistent grid field $M(x, d)$ of shape $(G \times G \times K)$ that creates a feedback loop between particle dynamics and the environment. Particles both **read** from and **write** to the field as they move through space, allowing the system to develop location-dependent behavioral biases based on interaction history.

## Mechanism

### Step 1: Read (before physics)

Each particle's preferences are modulated by the field at its grid cell:

$$\tilde{p}_i^{(d)} = p_i^{(d)} \cdot \left(1 + \alpha \cdot M(\text{cell}(x_i), d)\right)$$

where:
- $p_i^{(d)}$ is particle $i$'s preference in dimension $d$
- $\tilde{p}_i^{(d)}$ is the effective (modulated) preference used for physics
- $\alpha$ is the **field strength** parameter
- $M(\text{cell}(x_i), d)$ is the memory field value at the particle's grid cell

The modulated preferences are clipped to $[-1, 1]$.

**Effect**: Positive field values amplify the matching preference dimension. Negative values dampen it. A region where $M(x, 0) > 0$ will cause particles with $p^{(0)} > 0$ to have stronger attraction in dimension 0, while particles with $p^{(0)} < 0$ will have weaker (dampened) repulsion.

### Step 2: Physics

The simulation runs normally using the modulated preferences $\tilde{p}_i$ for neighbor selection and compatibility weighting. The original (unmodulated) preferences $p_i$ are preserved.

Any social learning delta $\Delta p$ from the physics kernel is captured and applied to the **original** preferences:

$$p_i \leftarrow \text{clip}\left(p_i + \Delta p, -1, 1\right)$$

This ensures the field only affects interaction forces, not the stored identity of the particles.

### Step 3: Write (after physics)

Each particle deposits its raw preferences into the field at its **new** position:

$$M(\text{cell}(x_i), d) \leftarrow M(\text{cell}(x_i), d) + \beta \cdot p_i^{(d)}$$

where $\beta$ is the **write rate**. Multiple particles in the same cell accumulate via `np.add.at`.

### Step 4: Decay

The entire field decays exponentially:

$$M \leftarrow \gamma \cdot M$$

where $\gamma$ is the **decay** parameter. The half-life in steps is $t_{1/2} = \frac{\ln 2}{\ln(1/\gamma)}$.

### Step 5: Blur (optional)

A Gaussian blur with standard deviation $\sigma$ (in grid cells) is applied to each channel independently, with periodic boundary wrapping:

$$M^{(d)} \leftarrow M^{(d)} * \mathcal{G}(\sigma)$$

This diffuses the memory spatially, causing particle traces to spread into neighboring cells.

## Parameters

| Parameter | Symbol | Range | Default | Description |
|-----------|--------|-------|---------|-------------|
| Write Rate | $\beta$ | 0.0 -- 1.0 | 0.01 | Deposition rate per particle per step |
| Field Strength | $\alpha$ | 0.0 -- 5.0 | 0.5 | Modulation strength on preferences |
| Field Decay | $\gamma$ | 0.9 -- 1.0 | 0.999 | Exponential decay per step |
| Blur Sigma | $\sigma$ | 0.1 -- 10.0 | 1.0 | Gaussian diffusion radius (grid cells) |

### Decay half-lives

| $\gamma$ | Half-life (steps) | Character |
|----------|-------------------|-----------|
| 0.99 | 69 | Short-term memory |
| 0.995 | 139 | Medium-term |
| 0.999 | 693 | Long-term memory |
| 0.9999 | 6931 | Near-permanent |
| 1.0 | $\infty$ | Permanent (grows without bound) |

## Expected Behaviors

### Regime 1: Faint trails (low $\beta$, low $\alpha$)

Particles leave barely perceptible traces. The Memory view shows thin paths where particles have traveled. Minimal feedback on dynamics — the system behaves almost identically to no memory.

**Settings**: $\beta = 0.001$, $\alpha = 0.1$, $\gamma = 0.999$, blur off.

### Regime 2: Territory formation (high $\beta$, high $\alpha$, no blur)

Rapid spatial polarization. The first particles to visit a region "claim" it:

1. Particle with $p^{(0)} = +1$ passes through cell $(x, y)$
2. Field accumulates $M(x, y, 0) > 0$
3. Future particles at $(x, y)$ get $\tilde{p}^{(0)}$ amplified if positive, dampened if negative
4. Positive-$p^{(0)}$ particles are attracted more strongly $\rightarrow$ they stay $\rightarrow$ more deposition

This positive feedback creates sharp boundaries between regions of different preference types.

**Settings**: $\beta = 0.05$, $\alpha = 1.0$, $\gamma = 0.999$, blur off.

### Regime 3: Pheromone landscapes (medium $\beta$, medium $\alpha$, with blur)

Smooth preference gradients emerge instead of sharp territories. The blur diffuses deposits into broad regional biases. Particles drift along these gradients, reinforcing them.

This is closest to **ant colony pheromone dynamics** — particles leave chemical-like traces that guide future behavior, with spatial diffusion and temporal decay.

**Settings**: $\beta = 0.01$, $\alpha = 0.5$, $\gamma = 0.999$, blur on, $\sigma = 2.0$.

### Regime 4: Memory + differentiation (high $\alpha$ + negative social)

The most interesting regime. **Social learning** ($s < 0$) pushes particles to differentiate from their neighbors, while the **memory field** pushes them to conform to local history.

- Social force: "be different from who's near you now"
- Memory force: "be similar to who was here before"

The tension between these opposing pressures can create:
- **Oscillations**: particles differentiate, leave, field decays, new particles arrive and conformto old memory, then differentiate again
- **Traveling waves**: regions of preference bias propagate as particles carry information
- **Persistent structures**: stable spatial patterns where the field and particle distribution reach equilibrium

**Settings**: $\beta = 0.01$, $\alpha = 1.0$, $\gamma = 0.999$, $s = -0.003$, blur on, $\sigma = 1.0$.

### Regime 5: Colony ghosts (high $\alpha$, medium $\gamma$)

When particles cluster and deposit strongly, the field at those cells grows, amplifying preferences, increasing attraction, tightening the cluster. This is a **positive feedback loop** that creates stable "colonies."

If the cluster eventually disperses (due to repulsion or social differentiation), the field retains a "ghost" of the colony — a region of strong field values with no particles. This ghost can:
- Attract future particles of the same type back to the abandoned site
- Create "territorial memory" that persists after the original occupants leave
- Generate cyclic colonization-abandonment patterns

**Settings**: $\beta = 0.02$, $\alpha = 1.5$, $\gamma = 0.998$, blur on, $\sigma = 1.0$.

## Visualization

The **Memory** right-panel view shows $M(x, d)$ as RGB:
- Red channel: $M(x, 0)$
- Green channel: $M(x, 1)$  
- Blue channel: $M(x, 2)$

Each channel is independently normalized by its maximum absolute value, mapped from $[-\max, +\max]$ to $[0, 1]$:
- Gray (0.5, 0.5, 0.5) = neutral (no memory)
- Bright red = strong positive $M^{(0)}$ ("dimension 0 positive particles dominated here")
- Dark red (near 0) = strong negative $M^{(0)}$ ("dimension 0 negative particles dominated")
- White = all dimensions strongly positive
- Colored regions = spatially varying preference bias

## Mathematical Properties

### Fixed point analysis

At steady state ($\dot{M} = 0$), the field satisfies:

$$M^* = \frac{\beta}{\gamma^{-1} - 1} \cdot \langle p \rangle_{\text{local}}$$

where $\langle p \rangle_{\text{local}}$ is the time-averaged preference of particles passing through each cell. For $\gamma = 0.999$ and $\beta = 0.01$:

$$M^* \approx 10 \cdot \langle p \rangle_{\text{local}}$$

So the steady-state field is approximately 10$\times$ the local average preference.

### Feedback gain

The effective feedback gain is $\alpha \cdot |M^*|$. For the modulation to be significant but not overwhelming:

$$\alpha \cdot |M^*| \sim O(1)$$

With $M^* \approx 10 \cdot \langle p \rangle$ and $|\langle p \rangle| \sim 0.5$:

$$\alpha \cdot 5 \sim 1 \implies \alpha \sim 0.2$$

This suggests $\alpha \approx 0.2$ is the transition point between weak and strong feedback.

### Diffusion length

With blur sigma $\sigma$ applied every step with decay $\gamma$, the effective diffusion length (how far a deposit spreads before decaying to $1/e$) is:

$$L_{\text{diff}} = \sigma \cdot \sqrt{\frac{2}{1 - \gamma}} \cdot \frac{L_{\text{space}}}{G}$$

For $\sigma = 2$, $\gamma = 0.999$, $G = 256$:

$$L_{\text{diff}} \approx 2 \cdot \sqrt{2000} \cdot \frac{1}{256} \approx 0.35$$

This means a deposit diffuses across about 35% of the domain before decaying. Higher sigma or higher decay extends this.
