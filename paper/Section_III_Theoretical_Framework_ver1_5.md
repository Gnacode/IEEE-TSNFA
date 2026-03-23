# SECTION III: THEORETICAL FRAMEWORK

This section formalises the detection algorithms evaluated in the comparative Monte Carlo study. For each method we first provide a plain-language overview and the governing equations, then the pseudocode implementation, and finally a line-by-line annotation that traces the signal-processing rationale behind every operation. The goal is to make the theoretical basis fully reproducible.

All algorithms operate on a discrete-time signal $x[n]$ sampled at $f_s = 100\;\text{Hz}$ and segmented into non-overlapping frames of $L = 128$ samples, giving a frame duration $T_f = L / f_s = 1.28\;\text{s}$ and a frame rate $f_\text{frame} = 1 / T_f \approx 0.781\;\text{Hz}$. The frequency resolution of the $L$-point FFT is $\Delta f = f_s / L = 0.781\;\text{Hz}$, placing the selected event band $[1, 5]\;\text{Hz}$ in FFT bins $\mathcal{K} = \{1, 2, 3, 4, 5, 6\}$.



**Monte Carlo signal model.** The simulation evaluates all algorithms against a common synthetic signal created to represent a realistic signal experienced by the sensors when deployed outdoor. The sampled signal at node $i$ and sample index $n$ within frame $m$ is:
$$
x_i[n] = s[n - \tau_i] + w_\text{th}[n] + w_\text{EMI}[n] + w_\text{dig}[n]
$$
where $s[n]$ is the event waveform (a damped sinusoidal impulse at 1–5 Hz with SNR = 18 dB relative to in-band noise power $P$, $w_\text{th}$ is zero-mean white Gaussian thermal noise with power spectral density $P / f_s$, $w_\text{EMI}$ is 60 Hz mains interference at amplitude $0.3\sqrt{P}$, modelled as a simplified fixed fraction of the noise amplitude to represent worst-case EMI coupling that co-varies with the receiver noise floor, and $w_\text{dig}$ equally simplified represents intermittent digital switching bursts at 800–2000 Hz with amplitude up to $2.0\sqrt{P}$. The noise power $P$ itself varies sinusoidally over a 1-hour period with $\pm 6\;\text{dB}$ excursion:
$$
P(t) = P_0 \cdot 10^{(A/10)\sin(2\pi t / T_\text{cycle})}
$$
where $A = 6\;\text{dB}$ and $T_\text{cycle} = 3600\;\text{s}$, so that $P$ swings between $P_0/4 $ and $4P_0 $ (a factor of 16 from minimum to maximum) over each hour. The chosen noise model introduces three failure modes that challenge the different algorithms tested in the simulation: the $\pm 6\;\text{dB}$ drift forces any viable detector to adapt its threshold over time, the 60 Hz EMI injects structured interference that a time-domain detector cannot distinguish from event content, and the intermittent digital bursts create impulsive transients at unpredictable intervals. Each algorithm receives the identical signal realisation; performance differences arise solely from the detection logic.

---

## A. Algorithm 1 TSNFA-mean: Temporal Spectral Noise-Floor Adaptive Triggering (Mean Variant)

### 1) Overview

The Temporal Spectral Noise-Floor Adaptive (TSNFA) trigger is the detection method proposed by Makovetskii and Thomsen 2026. It is built on three interlocking defences, each targeting a distinct failure mode observed in deployed IoT perimeter-security sensors:

**Defence 1: Spectral band selection.** An $L$-point FFT decomposes each frame into frequency bins, and only the bins covering the event band $[f_\text{lo}, f_\text{hi}] = [1, 5]\;\text{Hz}$ are retained. All energy from noise sources like electromagnetic interference (EMI) at 60 Hz and its harmonics, digital switching noise above 100 Hz, and broadband thermal noise outside the event band is discarded before any detection statistic is computed.

**Defence 2: Temporal persistence filtering ($\gamma_d$).** A sliding-window mean over a few consecutive frames $\gamma_d$ rejects transient noise spikes that survive spectral filtering. A genuine event persists across multiple frames (a 5-second event spans $5 / T_f \approx 3.9$ frames); a single-frame wind gust or ADC glitch does not.

**Defence 3: Adaptive noise-floor tracking ($\gamma_a$).** An exponential moving average (EMA) tracks the slowly drifting in-band noise power, and the detection threshold is recomputed every frame as a fixed multiple $\zeta$ of the current noise-floor estimate. This makes the threshold self-calibrating: as the noise environment changes over hours or seasons, the threshold follows, maintaining a constant false-alarm probability.

The three defences combines into a strong noise reduction strategy: band selection reduces the noise variance by discarding out-of-band energy, persistence filtering further suppresses impulsive in-band transients, and noise-floor adaptation prevents the threshold from becoming stale. No single defence is sufficient alone, but together they achieve 100% detection rate with zero false positives across a 24-hour, 200-node Monte Carlo simulation, which is matched by real application data (Makovetskii and Thomsen 2026).

**Spectral decomposition.** The $L$-point discrete Fourier transform of frame $m$ is:

$$X_m[k] = \sum_{n=0}^{L-1} x_m[n] \, e^{-j 2\pi k n / L}, \quad k = 0, 1, \ldots, L-1$$

The magnitude spectrum is $|X_m[k]|$. The detection statistic is the maximum in-band magnitude:

$$\mathcal{X}(m) = \max_{k \in \mathcal{K}} |X_m[k]|$$

This reduces the 6-bin event band to a single scalar per frame. The max operator is a practical simplification: because impulsive broadband events typically concentrate energy in one or two FFT bins, retaining only the strongest bin preserves the dominant signature. However, spectral shape information is lost and the identity of the active bin and the relative levels across bins are discarded. Algorithm 2 (median variant) addresses this limitation by processing each bin independently and performs better. The max aggregation matches the deployed hardware implementation and serves as a conservative baseline for the mean variant.

**Digital noise filter (persistence).** The filtered statistic is the arithmetic mean over a sliding window of $\gamma_d$ frames:

$$\bar{\mathcal{X}}(m) = \frac{1}{\gamma_d} \sum_{i=0}^{\gamma_d - 1} \mathcal{X}(m - i)$$

For $\gamma_d = 3$, a transient spike in a single frame is attenuated by a factor of 3 in the mean, while a genuine event persisting across all 3 frames passes through at full amplitude. In Algorithm 2 (median variant) this technique gives even better suppression of single bin spikes.

**Detection threshold and trigger.** The threshold is:

$$\Theta(m) = \zeta \cdot \hat{N}(m-1)$$

where $\zeta = 6.0$ is the threshold multiplier. In plain terms: the detection threshold sits at $6\times$ the current noise-floor estimate, so an event must produce in-band energy at least 6 times stronger than the ambient noise to trigger a detection. Expressed in decibels this corresponds to a minimum detectable signal-to-noise ratio of $\text{SNR}_\text{min} = 20 \log_{10}(6.0) = 15.6\;\text{dB} $. Since the simulation events are injected at 18 dB (amplitude ratio $\approx 7.9 $), they exceed the threshold by a margin of $18 - 15.6 = 2.4\;\text{dB} $. This margin provides robustness: even when instantaneous noise fluctuations reduce the effective SNR, the event still clears the threshold. The detection ratio and trigger decision are:

$$R(m) = \frac{\bar{\mathcal{X}}(m)}{\Theta(m)}, \qquad E[m] = \begin{cases} 1 & \text{if } R(m) > 1.0 \\ 0 & \text{otherwise} \end{cases}$$

R>1 means the threshold is exceeded and an event is declared. The same ratio also controls whether the noise floor is allowed to update, as described next.

**Adaptive noise floor.** The noise-floor estimate $\hat{N}(m)$ is an exponential moving average that updates only when the current observation is well below the detection threshold:

$$\hat{N}(m) = \begin{cases} \alpha \, \hat{N}(m-1) + (1 - \alpha) \, \bar{\mathcal{X}}(m) & \text{if } R(m) < R_\text{gate} \\ \hat{N}(m-1) & \text{otherwise} \end{cases}$$

where $\alpha = 1 - 1/\gamma_a$ is the EMA smoothing coefficient. With $\gamma_a = 64$ we get $\alpha = 1 - 1/64 = 0.984 $, meaning each new measurement contributes only 1.6% to the updated estimate while the previous estimate retains 98.4% of its weight. The adaptation depth $\gamma_a$ determines how many frames it takes for the noise-floor estimate to substantially respond to a sustained change: after $\gamma_a = 64$ frames ($\gamma_a \times T_f = 64 \times 1.28 = 81.9\;\text{s}$), the estimate has moved 63% of the way toward the new level. This is the time constant for the adaptive noise floor time  $\tau_a $ and it must be long enough that a 5-second event (spanning only $\sim 4 $ frames) barely perturbs the estimate, yet short enough to follow slow environmental noise drift (in the simulation used in this article, a 1-hour cycle; in practice, whatever timescale the deployment environment exhibits).  $R_\text{gate} = 0.8$ is the gating ratio: the noise floor only updates when the current observation is below 80% of the threshold, creating a guard band that also excludes the rising and falling edges of real events where $R$ might be between 0.8 and 1.0. Thus the function of the $R_\text{gate}$ is to prevent real events to add to the change in the noise floor level.


### 2) Pseudocode

```
Temporal Spectral Noise-Floor Adaptive Triggering (Mean Variant)
Input:  Sample frame x[0..N-1], noise floor N(m-1), filter buffer B(m)
Params: gamma_d = 3, gamma_a = 64, zeta = 6.0, alpha = 1 - 1/gamma_a
Output: Trigger decision E[m], updated noise floor N(m)

  // Stage 1: Spectral estimation (band selection)
  1.  X <- |FFT(x)|                                // time to frequency, magnitude only (phase discarded)
  2.  X(m) <- max{ X[k] : k in {1,...,6} }         // X(m) = max over K, event band only

  // Stage 2: Digital noise filter (gamma_d averaging)
  3.  Append X(m) to buffer B                       // sliding window of gamma_d frames
  4.  if |B| > gamma_d then discard oldest entry     // FIFO, keep exactly gamma_d entries
  5.  X_bar(m) <- mean(B)                            // X_bar(m) = (1/gamma_d) * sum(B)

  // Stage 3: Threshold computation from last noise floor update raised by zeta
  6.  Theta(m) <- zeta * N(m-1)                      // Theta = 6.0 * N_hat(m-1)

  // Stage 4: Minimal detection ratio set by zeta and trigger decision if ratio exceeds 1
  7.  R(m) <- X_bar(m) / Theta(m)                    // R(m) = X_bar(m) / (zeta * N_hat(m-1))
  8.  if R(m) > 1.0 then E[m] <- 1                   // trigger: X_bar > 6 * noise floor

  // Stage 5: Noise-floor adaptation (gated EMA update)
  9.  if R(m) < 0.8 then                             // R_gate = 0.8: update only when quiet
  10.     N(m) <- alpha * N(m-1) + (1-alpha) * X_bar(m) // alpha = 0.984: 1.6% new, 98.4% old values
      else
          N(m) <- N(m-1)                              // freeze: event energy must not leak in
      end if
  11. return E[m], N(m)
```


### 3) Line-by-Line Annotation

**Line 1: `X ← |FFT(x)|`**
Compute the 128-point Fast Fourier Transform of the current sample frame, then take the magnitude (absolute value) of each complex bin. The FFT converts 128 time-domain samples into 64 frequency bins (plus DC and Nyquist). Each bin represents energy at a specific frequency: bin *k* corresponds to frequency $k \cdot (f_s / L) = k \times 0.781\;\text{Hz}$. The magnitude discards phase information, which is irrelevant for event detection because we only care about how much energy is present at each frequency, not when it occurred  within the frame. 

**Line 2: `X(m) ← max{ X[k] : k ∈ {1,…,6} }`**
Extract only the 6 FFT bins covering the event band $[0.78, 4.69]\;\text{Hz}$, and take the maximum magnitude across them. This is the **band selection** step and it is the single most important operation in TSNFA. By discarding all bins outside $\mathcal{K} = \{1,\ldots,6\}$, we reject: (a) 60 Hz EMI and its harmonics at 120/180 Hz (bins ~77, 154, 231), (b) digital switching noise at 800–2000 Hz (bins ~1024–2560, aliased), (c) the broadband high-frequency component of thermal noise. Only in-band energy survives. In this mean variant, the 6 bin magnitudes are reduced to a single scalar by taking their maximum. This is a practical simplification: a genuine event typically concentrates energy in one or two frequency bins, so retaining only the strongest bin preserves the dominant event signature while collapsing the detection problem to a single time series. The down-side is that spectral shape information is discarded and if for example bin 3 shows sustained elevation while the other five remain at noise level, the max operator captures the elevation but loses the context that only one bin is active. Algorithm 2 (the median variant) takes the alternative approach of processing each bin independently, which preserves this spectral discrimination at the cost of additional memory and computation. The max aggregation used here matches the deployed hardware implementation and is adequate for the mean variant's role as a conservative baseline.  Note that this max operation combines frequency bins within a single frame and it answers "which bin has highest amplitude right now?"

**Line 3: `Append X(m) to buffer B`**
Store the current frame's band maximum into a sliding window buffer. This buffer accumulates the last $\gamma_d$ frame statistics for temporal averaging in the next step.

**Line 4: `if |B| > γd then discard oldest entry`**
Keep the buffer at exactly $\gamma_d$ entries (FIFO, first in first out). When the buffer is full, the oldest entry is removed before the new one is added. During the first $\gamma_d - 1$ frames after start up, the buffer is not filled completely and the mean in Line 5 uses fewer samples.

**Line 5: `X̄(m) ← mean(B)`**
Compute the arithmetic mean of the buffer contents. This is the **digital noise filter** and it requires that energy be persistently elevated across $\gamma_d$ consecutive frames before it contributes to the detection statistic. A wind gust that spikes bin 3 in a single frame but not the next two frames is averaged down to $1/3$ of its peak. A genuine event lasting 5 seconds spans $5/1.28 \approx 3.9$ frames, so it remains elevated across all 3 buffer positions, surviving the averaging. This is the **persistence requirement** and the second key defence unique to TSNFA. Note: Algorithm 1 (mean variant) is susceptible to outliers and a single extreme value shifts the mean by up to $1/\gamma_d$ of the outlier magnitude. Algorithm 2 replaces the mean with a median filter, which eliminates this sensitivity.

**Line 6: `Θ(m) ← ζ · N(m−1)`**
Compute the detection threshold by multiplying the noise-floor estimate from the **previous** frame by the arbitrarily set threshold coefficient $\zeta = 6.0$. We use $\hat{N}(m-1)$,  not $\hat{N}(m)$ alone, because $\hat{N}(m)$ has not been computed yet. The noise threshold must exist before we can decide whether the current frame is an event, and that decision determines whether we update the noise floor. The $\zeta = 6.0$ value comes from deployed hardware measurements where the ratio $\bar{\mathcal{X}}/\hat{N}$ during non-events was measured at mean $\approx 0.17$ (i.e., $1/6$), with all observed values $\leq 1.0$. Setting $\zeta = 6$ raises the threshold for detection to $\text{SNR}_\text{min} = 20 \log_{10}(6) = 15.6\;\text{dB}$.

**Line 7: `R(m) ← X̄(m) / Θ(m)`**
Compute the ratio of filtered band energy to threshold. This ratio serves two mutually excluding purposes: (a) trigger decision ($R > 1.0$ means exceedance), and (b) noise-floor gating ($R < 0.8$ means safe to use for updating the noise floor). The ratio can also serve as a confidence metric: $R = 1.01$ is a marginal detection; $R = 1.37$ (the hardware mean during events) is a confident detection; $R = 0.17$ (the hardware mean during non-events) is clearly quiescent.

**Line 8: `if R(m) > 1.0 then E[m] ← 1`**
The trigger decision. If the filtered band energy exceeds the adaptive threshold, declare an event. $R > 1.0$ is equivalent to $\bar{\mathcal{X}}(m) > \zeta \cdot \hat{N}(m-1)$, i.e., the filtered in-band energy exceeds $6\times$ the current noise-floor estimate. Note there is no hysteresis, debounce, or minimum duration and a single frame exceedance after $\gamma_d$-frame averaging is sufficient. 

**Line 9: `if R(m) < 0.8 then`**
The **gating condition** for noise-floor updates. We only update $\hat{N}$ (noise floor) when the current observation is clearly below the detection threshold ($R < 0.8$, i.e., $\bar{\mathcal{X}} < 80\%$ of $\Theta$). The 0.8 guard band serves two mutually exclusive purposes: (a) prevents event energy from leaking into the noise-floor estimate (an event with $R = 1.2$ must not update $\hat{N}$), and (b) avoids updating during the rising/falling edges of events where $R$ might be between 0.8 and 1.0. If $R \geq 0.8$, the noise floor is frozen at its previous value.

**Line 10: `N(m) ← α·N(m−1) + (1−α)·X̄(m)`**
The **exponential moving average (EMA) noise-floor update** is the third key defence (adaptive noise floor). When gating permits, blend the previous noise floor with the current observation using weight $\alpha = 0.984$. This gives: $\hat{N}(m) = 0.984 \times \hat{N}(m-1) + 0.016 \times \bar{\mathcal{X}}(m)$. The current observation contributes only 1.6% to the new estimate, making the noise floor very stable. The time constant is $\gamma_a \times T_f = 64 \times 1.28 = 81.9\;\text{s}$ and it takes approximately 82 seconds for $\hat{N}$ to reach 63% of a step change in the noise environment. This is slow enough that a 5-second event (even if the gate briefly fails) contributes negligibly to $\hat{N}$, but fast enough to follow slow environmental noise drift. The EMA is a first-order IIR low-pass filter with $-3\;\text{dB}$ cut-off at $f_c = (1-\alpha)/(2\pi T_f) \approx 0.002\;\text{Hz}$.

**Line 11: `return E[m], N(m)`**
Return both the event flag and the updated noise floor. The noise floor is carried forward to the next frame where it becomes $\hat{N}(m-1)$ in the next round.


### 4) Parameters

| Parameter | Symbol | Value | Reasoning |
|-----------|--------|-------|-----------|
| FFT size | $L$ | 128 | At $f_s = 100\;\text{Hz}$: $\Delta f = 0.78\;\text{Hz}$.  Fits in STM32G071 SRAM (512 B real, 1 KB complex). Frame duration $T_f = 1.28\;\text{s}$. |
| Sample rate | $f_s$ | 100 Hz | Nyquist for the 1–5 Hz event band. Higher rates waste power/memory; lower rates alias event content. |
| Event band | $[f_\text{lo}, f_\text{hi}]$ | [1, 5] Hz | Bins $k \in \{1,\ldots,6\}$ at 0.78 Hz resolution span 0.78–4.69 Hz. |
| Band statistic | max | — | Max across 6 bins. Practical simplification that preserves the dominant bin; discards spectral shape (see Algorithm 2 for per-bin alternative). |
| Digital filter | $\gamma_d$ | 3 | Mean of 3 frames. Rejects transients not persisting $\geq 3$ frames. Events at $T_\text{event} = 5\;\text{s}$ persist for $5/1.28 \approx 3.9$ frames. |
| Adaptation depth | $\gamma_a$ | 64 | Time constant $\tau_a = \gamma_a \times T_f = 81.9\;\text{s}$. Slow enough to reject 5 s events, fast enough to track slow changing environmental noise drift. |
| EMA coefficient | $\alpha = 1 - 1/\gamma_a$ | 0.984 | At $\alpha = 0.984$, noise floor reaches 63% of a step change after 64 frames (82 s). |
| Threshold | $\zeta$ | 6.0 | From hardware: during non-events $\bar{\mathcal{X}}/\hat{N} \approx 0.17$. During events $\approx 1.37$. $\text{SNR}_\text{min} = 15.6\;\text{dB}$. |
| Gate ratio | $R_\text{gate}$ | 0.8 | Only update $\hat{N}$ when $R < 0.8$. Prevents real event energy from contributing to the noise floor. |

**Detection threshold in dB:** $\text{SNR}_\text{min} = 20 \log_{10}(\zeta) = 20 \log_{10}(6) = 15.6\;\text{dB}$. Events at SNR = 18 dB exceed this by 2.4 dB during nominal noise. During noise peaks ($+6\;\text{dB}$), the effective in-band SNR drops but the adaptive noise floor tracks the rising baseline and maintains the threshold relative to current noise, not calibration noise. The 200-node, 24-hour Monte Carlo simulation confirms 100% DR with 0 FP.


---


## B. Algorithm 2 TSNFA-median: Temporal Spectral Noise-Floor Adaptive Triggering (Median Variant)

### 1) Overview

The median variant of TSNFA is the form implemented in the deployed hardware. It differs from Algorithm 1 in four specific operations, each of which can be expressed as a direct formula substitution (the four differences are also illustrated in Figure 2) :

**Difference 1 to Alg.1: Bin aggregation replaced by per-bin processing.** Algorithm 1 collapses the 6 event-band bins into a single scalar before any filtering:
$$
\text{Alg.\,1:} \quad \mathcal{X}(m) = \max_{k \in \mathcal{K}} |X_m[k]| \qquad \text{(one value per frame)}
$$
Algorithm 2 retains each bin magnitude individually and processes them through independent filter chains:
$$
\text{Alg.\,2:} \quad |X_k(m)| \;\text{for each}\; k \in \mathcal{K} \qquad \text{(six values per frame)}
$$
**Difference 2 to Alg.1: Mean filter replaced by median filter (Stage 1).** The digital noise filter in Algorithm 1 uses the arithmetic mean over $\gamma_d$ frames:
$$
\text{Alg.\,1:} \quad \bar{\mathcal{X}}(m) = \frac{1}{\gamma_d} \sum_{i=0}^{\gamma_d - 1} \mathcal{X}(m - i)
$$
Algorithm 2 replaces this with the median, applied independently per bin:
$$
\text{Alg.\,2:} \quad \tilde{N}_k(m) = \text{median}\bigl(\{|X_k(m)|, |X_k(m-1)|, \ldots, |X_k(m - \gamma_d + 1)|\}\bigr)
$$
The mean shifts by up to $1/\gamma_d$ of an outlier's magnitude; the median is unaffected by any single outlier regardless of its size.

**Difference 3 to Alg.1: EMA replaced by median filter (Stage 2).** The noise-floor tracker in Algorithm 1 uses a gated exponential moving average:
$$
\text{Alg.\,1:} \quad \hat{N}(m) = \alpha \, \hat{N}(m-1) + (1-\alpha) \, \bar{\mathcal{X}}(m) \qquad \text{(if } R < R_\text{gate}\text{)}
$$
Algorithm 2 replaces this with a second median filter over a longer window, again per bin:
$$
\text{Alg.\,2:} \quad \hat{N}_k(m) = \text{median}\bigl(\{\tilde{N}_k(m), \tilde{N}_k(m-1), \ldots, \tilde{N}_k(m - \gamma_a + 1)\}\bigr)
$$
The EMA requires explicit gating ($R_\text{gate} $) to prevent event energy from contaminating the noise floor. The median needs no gating — with $\gamma_a = 64 $, up to 31 consecutive event frames can enter the buffer without shifting the median, because the 33 non-event values still outnumber them at the median rank position.

These three substitutions yield stronger outlier rejection and finer spectral discrimination at the cost of additional memory and computation.

**Difference 4 to Alg.1: Per-bin processing and OR-logic trigger.** Each bin $k$ has its own pair of circular buffers ($B_{d,k}$ for Stage 1, $B_{a,k}$ for Stage 2). The trigger decision compares the raw (unfiltered) magnitude against the per-bin threshold:
$$
E[m] = \begin{cases} 1 & \text{if } \exists\, k \in \mathcal{K} : |X_k(m)| > \zeta_k \cdot \hat{N}_k(m) \\ 0 & \text{otherwise} \end{cases}
$$
Using the raw $|X_k| $ (rather than the median-filtered $\tilde{N}_k $) preserves maximum sensitivity: the median filter determines the noise floor but the trigger uses the instantaneous spectral magnitude. A trigger from any single bin is sufficient (OR logic). This preserves spectral shape information: if only bin 3 ($\approx 2.3\;\text{Hz} $) shows sustained elevation while the other five remain at noise level, Algorithm 2 detects this precisely, whereas Algorithm 1 would discard the spectral context via the max operator.

**Median breakdown point.** The median of $n$ values tolerates up to $\lfloor (n-1)/2 \rfloor$ arbitrarily corrupted entries without shifting. For $\gamma_d = 3$, one out of three values can be an extreme outlier with zero effect on the output. For $\gamma_a = 64$, up to 31 corrupted frames leave the noise-floor estimate unchanged. This is the formal basis for the claim that the median needs no gating: even if 31 consecutive event frames enter $B_{a,k}$, the 33 non-event values still control the median rank position.

**Two-stage cascade.** Stage 2 receives already-cleaned inputs because Stage 1 has rejected transient spikes before they enter the noise-floor buffer. Stage 1 ($\gamma_d$) addresses fast transients on the timescale of 3–6 seconds: ADC glitches, voltage regulator switching noise, brief EMI bursts. Stage 2 ($\gamma_a$) addresses slow drift on the timescale of 82–164 seconds: temperature-induced gain changes, diurnal EMI patterns, weather changes, seasonal environmental shifts. A single filter cannot simultaneously provide fast transient rejection and slow drift tracking and therefore the two timescales require separate processing stages.


### 2) Pseudocode

```
Temporal Spectral Noise-Floor Adaptive Triggering (Median Variant)
Input:  Sample frame x[0..N-1], circular buffers B_d,k and B_a,k
Params: gamma_d in [3, 5], gamma_a in [64, 128], zeta_k
Output: Trigger decision, updated noise floors N_hat_k[t]

  // Spectral estimation (same as Algorithm 1)
  1.  X[k] <- FFT(x) for k in K                   // time to frequency, complex-valued
  2.  |X_k| <- sqrt(Re(X_k)^2 + Im(X_k)^2)        // magnitude only, phase discarded

  3.  E[t] <- 0                                     // no event until a bin proves otherwise

  4.  for each bin k in K do                         // k = 1..6, each bin processed independently

        // Stage 1: Digital noise suppression (per-bin median)
  5.      Insert |X_k| into circular buffer B_d,k    // FIFO, keeps last gamma_d values for this bin
  6.      N_tilde_k <- median(B_d,k)                 // middle value of gamma_d frames; 1 outlier ignored

        // Stage 2: Noise-floor tracking (per-bin median, no gate needed)
  7.      Insert N_tilde_k into circular buffer B_a,k // FIFO, keeps last gamma_a cleaned values
  8.      N_hat_k[t] <- median(B_a,k)                // noise floor = middle of 64 values; 31 outliers ok

        // Trigger: raw magnitude vs per-bin threshold
  9.      if |X_k| > zeta_k * N_hat_k[t] then        // raw |X_k|, not filtered N_tilde_k
  10.         E[t] <- 1                                // any single bin exceeding is sufficient (OR logic)
          end if
      end for

  11. return E[t], {N_hat_k[t]}                      // event flag + 6 updated noise-floor estimates
```


### 3) Line-by-Line Annotation

**Line 1: `X[k] ← FFT(x) for k ∈ K`**
Same 128-point FFT as Algorithm 1, but here we retain the complex-valued output for each monitored bin. $\mathcal{K} = \{1, 2, \ldots, 6\}$ covers the event band. On the Cortex-M4 hardware, this is a fixed-point FFT using CMSIS-DSP `arm_cfft_q31` — 128-point complex FFT completes in ~0.2 ms at 168 MHz.

**Line 2: `|Xk| ← √(Re(Xk)² + Im(Xk)²)`**
Compute the magnitude of each complex FFT bin. This is done per-bin, not aggregated. Unlike Algorithm 1 which immediately takes the max across bins, Algorithm 2 preserves the per-bin magnitudes for individual processing. 

**Line 3: `E[t] ← 0`**
Initialise the event flag to "no event." It will be set to 1 if any bin triggers in the loop. This is OR logic and any single bin exceeding its threshold is sufficient to declare an event.

**Line 4: `for each bin k ∈ K do`**
Begin the per-bin processing loop. This is the **key architectural difference** from Algorithm 1: each of the 6 frequency bins is processed independently with its own circular buffers and noise-floor estimate. This preserves spectral shape information. If only bin 3 ($\approx 2.3\;\text{Hz}$) shows sustained elevation while bins 1, 2, 4, 5, 6 remain at noise level, Algorithm 2 detects this precisely in bin 3. Algorithm 1, by aggregating via max first, could be influenced by a transient spike in any bin contaminating the single aggregated statistic.

**Line 5: `Insert |Xk| into circular buffer Bd,k`**
Append the current frame's magnitude for bin $k$ into that bin's digital-filter circular buffer. Each bin has its own independent buffer of size $\gamma_d$. Total memory for Stage 1: $6 \times \gamma_d \times 4 = 72\;\text{bytes}$.

**Line 6: `Ñk ← median(Bd,k)`**
Compute the **median** of the digital-filter buffer for bin $k$. This is the critical difference from Algorithm 1's mean. The median is a rank-order statistic with 50% breakdown point: for $\gamma_d = 3$, up to 1 out of 3 values can be completely arbitrary without shifting the median beyond the remaining values. A single extreme noise spike (wind gust, ADC glitch) that produces $|X_k| = 100\times$ normal will have zero effect on the median if the other 2 buffer entries are normal. With the mean (Algorithm 1), that same spike shifts the average by $33\times$. The median computation uses a partial bubble sort (sort only to the middle position), requiring $\lfloor\gamma_d / 2\rfloor \times \gamma_d = 3$ comparisons for $\gamma_d = 3$.

**Line 7: `Insert Ñk into circular buffer Ba,k`**
Feed the digitally-filtered magnitude $\tilde{N}_k$ into the second-stage circular buffer. This cascaded architecture means $B_{a,k}$ receives already-cleaned values and transient spikes have been removed by the median in Stage 1. The second stage therefore tracks the slow-varying noise floor without contamination from digital transients.

**Line 8: `N̂k[t] ← median(Ba,k)`**
Compute the **noise-floor estimate** for bin $k$ as the median of the long-term buffer. This replaces the EMA update in Algorithm 1. With $\gamma_a = 64$, the median of 64 values tolerates up to 31 anomalous entries and even if an event lasts 31 consecutive frames (39.7 s), the noise-floor estimate is unaffected. The time constant is comparable to Algorithm 1's EMA (82 s), but with much stronger outlier rejection. Computation: partial sort to position 32 requires $\sim 64 \times 32 = 2{,}048$ comparisons per bin per frame. With 6 bins at 0.78 frames/s: $\sim 9{,}600$ comparisons/s, negligible even on a 64 MHz Cortex-M0+.



IM HERE NOW

**Line 9: `if |Xk| > ζk · N̂k[t] then`**
Per-bin trigger comparison. Note this compares the **raw** magnitude $|X_k|$ against the threshold, not the filtered $\tilde{N}_k$. This is a design choice in the hardware variant: the median filter determines the noise floor but the trigger uses the instantaneous value, providing maximum sensitivity. $\zeta_k$ can be set per-bin to accommodate non-uniform noise spectra, but in practice uniform $\zeta_k = 6$ is used.

**Line 10: `E[t] ← 1`**
Set the event flag. Because this is inside the for-each-bin loop, any single bin triggering is sufficient. The flag is never reset to 0 within the loop (OR logic).

**Line 11: `return event flag E[t], updated noise floors {N̂k[t]}`**
Return the event decision and the complete set of 6 per-bin noise-floor estimates. These persist in the circular buffers for the next frame.


### 4) Parameters

| Parameter | Symbol | Value | Reasoning |
|-----------|--------|-------|-----------|
| Digital filter | $\gamma_d$ | 3–5 | Per-bin circular buffer. Median of 3–5 values rejects up to $\lfloor(\gamma_d - 1)/2\rfloor$ outliers. |
| Analog tracker | $\gamma_a$ | 64–128 | Per-bin circular buffer. Median of 64–128 values tracks slow drift while rejecting up to 31–63 anomalous frames. Time constant $\approx$ 82–164 s. |
| Threshold | $\zeta_k$ | per-bin | Can be set per frequency bin for non-uniform noise spectra. In practice, uniform $\zeta_k = 6.0$. |
| Per-bin buffers | $B_{d,k}$, $B_{a,k}$ | 6 × 2 | 12 circular buffers total. Memory: $6 \times (3 + 64) \times 4 = 1{,}608\;\text{bytes}$. |

**Architectural difference from Algorithm 1:** Algorithm 2 processes each bin independently through two cascaded median filters, preserving spectral shape. Algorithm 1 aggregates across bins first (max), then applies a single mean filter + EMA. The per-bin architecture is more expensive in memory (1,608 bytes vs ~24 bytes) and computation (~9,600 comparisons/s vs ~10 operations/frame) but provides stronger outlier rejection and finer spectral discrimination.  Algorithm 1 already outperforms the fiver other algorithms 3-6, and therefore Algorithm 2 was not modelled in the Monte Carlo simulation. However, Algorithm 2 out performs Algorithm 1 when implemented in hardware and run for many weeks, because Algorithm 1 start to generate false positives when seen over a longer time span. 


---


## C. Algorithm 3: Time-Domain Adaptive Thresholding (Zhang et al. 2023)

### 1) Overview

The Zhang method is a time-domain adaptive threshold detector that operates on the peak amplitude of each sample frame without any spectral decomposition. It represents the class of energy-based detectors that assume the detection statistic can be computed directly from time-domain signal levels, relying on an adaptive threshold to track changing noise conditions.

**Operating principle.** For each frame of $L = 128$ samples, the algorithm finds the maximum absolute sample value (the frame peak) and compares it against a noise-floor estimate maintained by a gated exponential moving average. The core assumption is that events produce the highest sample values within a frame, so peak amplitude is a sufficient detection statistic. This assumption holds in environments where the event is the dominant signal component, but fails when structured interference (EMI, digital switching) produces peaks comparable to or exceeding the event amplitude.

**Detection statistic.** The frame-level statistic is the peak absolute amplitude:

$$\mathcal{X}_Z(m) = \max_{n \in [0, L-1]} |x[n]|$$

This statistic includes energy from all spectral components: the event signal, thermal noise across the full bandwidth, 60 Hz EMI at amplitude $0.3\sqrt{P}$, and digital switching bursts at up to $2.0\sqrt{P}$. No frequency discrimination is performed.

**Noise-floor tracking.** The noise floor $\hat{N}_Z(m)$ is updated via a gated EMA with coefficient $\beta = 0.95$:

$$\hat{N}_Z(m) = \begin{cases} \beta \, \hat{N}_Z(m-1) + (1-\beta) \, \mathcal{X}_Z(m) & \text{if } R(m) < R_\text{gate} \\ \hat{N}_Z(m-1) & \text{otherwise} \end{cases}$$

The time constant is $\tau_Z = 1/(1-\beta) \times T_f = 20 \times 1.28 = 25.6\;\text{s}$, which is $3.2\times$ faster than TSNFA's 82 s. The faster adaptation means $\hat{N}_Z$ tracks EMI amplitude fluctuations more closely, but also introduces instability: when EMI momentarily decreases, $\hat{N}_Z$ drops, and the threshold $\zeta \cdot \hat{N}_Z$ may fall below a subsequent noise transient.

**Trigger decision.** An event is declared when:

$$\mathcal{X}_Z(m) > \zeta \cdot \hat{N}_Z(m-1)$$

**Time domain issues.** The absence of spectral decomposition means the noise floor $\hat{N}_Z$ tracks the *composite* noise power which is dominated by 60 Hz EMI  rather than the in-band noise floor alone. In the simulation environment, the composite noise peak is $\sim 0.3\sqrt{P}$ (EMI) while the in-band noise is $\sim\sqrt{P/L_\text{band}} \approx \sqrt{P/6}$ per bin. The composite floor is higher, but it is also more variable (because EMI amplitude fluctuates), leading to both missed events (threshold inflated above the event peak during high-EMI phases) and false positives (threshold drops during low-EMI phases, permitting noise transients to trigger). The 200-node simulation yields 73.4% detection rate and 919,842 false positives ($\text{FAR} = 192.6\;\text{hr}^{-1}\text{node}^{-1}$).


### 2) Pseudocode

```
Time-domain adaptive thresholding - no spectral decomposition
Input:  Sample frame x[0..N-1], noise floor NZ(m-1)
Params: beta = 0.95, zeta = 6.0
Output: Trigger decision E[m]

  // Time-domain frame statistic (no FFT, no band selection)
  1.  XZ(m) <- max_n |x[n]|                        // peak sample in frame; includes EMI, digital, thermal

  // Trigger decision
  2.  R <- XZ(m) / NZ(m-1)                          // ratio of peak amplitude to noise floor
  3.  if XZ(m) > zeta * NZ(m-1) then E[m] <- 1      // trigger if peak > 6× noise floor

  // Noise-floor update (gated EMA, faster than TSNFA)
  4.  if R < 0.8 then                                // same gate logic as Algorithm 1
  5.      NZ(m) <- beta * NZ(m-1) + (1-beta) * XZ(m) // beta=0.95: 5% new, 95% old; tau ≈ 26 s
      else
          NZ(m) <- NZ(m-1)                            // freeze during events
      end if
  6.  return E[m]                                     // NZ tracks composite noise, not in-band only
```


### 3) Line-by-Line Annotation

**Line 1: `XZ(m) ← maxn |x[n]|`**
Scan all 128 samples in the frame and find the maximum absolute value. This is the **time-domain peak detector** — no FFT, no frequency analysis whatsoever. The peak amplitude captures whichever signal component has the highest instantaneous value at any moment within the 1.28-second frame. In the simulation noise environment, this means: 60 Hz EMI at amplitude $0.3\sqrt{P}$ produces peaks every $1/60 = 16.7\;\text{ms}$ (approximately 77 peaks per frame); digital switching bursts at $0.5$–$2.0\sqrt{P}$ (when active) produce even higher peaks; thermal noise peaks at $\sim 3\sigma = 3\sqrt{P}$ occasionally; event signals at $\sim 7.9\sqrt{P}$ (18 dB SNR). The frame max is dominated by whichever of these produces the single highest sample. Critically, the method **cannot distinguish** a $7.9\sqrt{P}$ event peak at 2 Hz from a $7.9\sqrt{P}$ composite peak where EMI, thermal noise, and a digital burst happen to align constructively.

**Line 2: `R ← XZ(m) / NZ(m−1)`**
Same ratio computation as Algorithm 1, but here the numerator includes all spectral content.

**Line 3: `if XZ(m) > ζ · NZ(m−1) then E[m] ← 1`**
Trigger when peak amplitude exceeds $6\times$ the noise floor. Because $\hat{N}_Z$ tracks the peak amplitude of all noise (including EMI), the threshold is inflated above what the event-band-only noise floor would produce. An event that would clearly trigger against the in-band noise floor may not trigger against the composite noise floor.

**Lines 4–5: Noise-floor update**
Same gated EMA structure as Algorithm 1, but with $\beta = 0.95$ (faster adaptation). Time constant = $1/(1-\beta) \times T_f = 20 \times 1.28 = 25.6\;\text{s}$. This is $3.2\times$ faster than TSNFA's 82 s, which means $\hat{N}_Z$ tracks EMI amplitude fluctuations more closely which is both a strength (adapts to EMI changes) and a weakness (the floor oscillates with EMI, introducing instability in the threshold).

**Line 6: `return E[m]`**
Return only the event flag. $\hat{N}_Z$ is maintained internally but not explicitly output.


### 4) Parameters

| Parameter | Symbol | Value | Reasoning |
|-----------|--------|-------|-----------|
| Frame statistic | $\max_n |x[n]|$ | - | Peak absolute amplitude over 128 samples. Dominated by 60 Hz EMI peak, not event. |
| Smoothing | $\beta$ | 0.95 | EMA coefficient. Time constant = $25.6\;\text{s}$. Faster than TSNFA but tracks EMI amplitude, not event-band noise. |
| Threshold | $\zeta$ | 6.0 | Set equal to TSNFA for fair comparison. |
| No FFT | - | - | Without spectral decomposition, the frame statistic includes 60 Hz EMI ($0.3\sqrt{P}$), digital noise bursts (up to $2.0\sqrt{P}$), and broadband thermal noise. |

**False positives:** $\hat{N}_Z$ tracks around $0.3\sqrt{P}$ (EMI peak level). When EMI momentarily decreases (destructive phase alignment with digital noise), $\hat{N}_Z$ drops over $\sim 26\;\text{s}$, and the threshold $6 \hat{N}_Z$ may fall below a subsequent noise transient, producing a false trigger. The simulation shows 919,842 false positives ($\text{FAR} = 192.6\;\text{hr}^{-1}\text{node}^{-1}$).

**Missed Events:** At first glance, missing events seems impossible: the event peak is $\sim 7.9\sqrt{P}$ while the threshold is $6 \hat{N}_Z \approx 1.8\sqrt{P} $ creating a comfortable margin. But $\hat{N}_Z$ is unstable because it tracks the *composite* noise (EMI + thermal + digital combined), not just the in-band component. When multiple noise sources happen to align constructively e.g. an EMI peak coinciding with a digital burst during a high-noise phase then the peak frame amplitude spikes, and $\hat{N}_Z$ rises in response. Once $\hat{N}_Z$ has been inflated by a few such coincidences, the threshold $6 \hat{N}_Z$ can climb above the event amplitude. An event arriving during one of these inflated-threshold windows is missed. With the $\beta = 0.95$ EMA responding in $\sim 26\;\text{s}$, the noise floor follows these composite fluctuations readily, creating frequent windows where the threshold is too high for detection. The 200-node simulation shows this produces 26.6% missed events alongside 919,842 false positives.


---


## D. Algorithm 4: STFT Spectral Gating (Bhoi et al. 2022)

### 1) Overview

The STFT method applies the same spectral decomposition as TSNFA which is an $L$-point FFT with band selection over $\mathcal{K} = \{1, \ldots, 6\}$  but compares the in-band magnitude against a **fixed** noise threshold $\Theta_0$ set once during an initial calibration period. There is no noise-floor adaptation and no temporal persistence filter. It represents the class of fixed-threshold spectral detectors.

**Operating principle.** During a noise-only calibration window of $M_\text{cal}$ frames at the start of deployment, the algorithm computes the in-band magnitude for each frame and sets $\Theta_0$ as the mean plus $3\sigma$ of the calibration-period statistics:

$$\Theta_0 = \bar{\mathcal{X}}_\text{cal} + 3\,\sigma_{\mathcal{X},\text{cal}}$$

where $\bar{\mathcal{X}}_\text{cal}$ and $\sigma_{\mathcal{X},\text{cal}}$ are the sample mean and standard deviation of $\mathcal{X}(m) = \max_{k \in \mathcal{K}} |X_m[k]|$ over the calibration frames. After calibration, $\Theta_0$ is frozen and never updated.

**Detection statistic.** Identical to TSNFA (Algorithm 1 and 2):

$$\mathcal{X}(m) = \max_{k \in \mathcal{K}} |X_m[k]|$$

**Trigger decision.** A single frame exceedance triggers:

$$E[m] = \begin{cases} 1 & \text{if } \mathcal{X}(m) > \Theta_0 \\ 0 & \text{otherwise} \end{cases}$$

There is no $\gamma_d$-frame averaging, no gated noise-floor update, and no EMA. The threshold is a single scalar computed once and used for the entire deployment lifetime.

**Highlights.** Band selection eliminates out-of-band noise, which is why STFT achieves the same 100% detection rate as TSNFA. The FFT + band mask is the correct foundation. Every event at 18 dB SNR in the event band produces a magnitude that exceeds even a stale threshold, because the event energy is concentrated precisely in the monitored bins.

**False positives.** The threshold $\Theta_0$ is set once at the start of deployment, when the noise power happens to be at its baseline level $P_0$. From that point on, $\Theta_0$ never changes. But the noise power drifts $\pm 6\;\text{dB}$ over each hour and is rising as high as $4 P_0 $ and falling as low as $P_0/4 $. When the noise power rises to $4 P_0 $, the in-band noise magnitude increases by $\sqrt{4} = 2\times$, pushing it above the frozen $\Theta_0$. Because the threshold cannot adapt, every frame during the high-noise phase triggers a false positive detection. This repeats every noise cycle, accumulating 399,822 false positives ($\text{FAR} = 83.7\;\text{hr}^{-1}\text{node}^{-1}$) over the 24-hour simulation. TSNFA avoids this entirely because its noise floor tracks the drift and the threshold rises with the noise, maintaining the gap.

**Missing functionality.** STFT lacks both adaptation ($\gamma_a$) and persistence ($\gamma_d$). Adding either one would dramatically reduce false positives; adding both (which is precisely TSNFA) eliminates them entirely.


### 2) Pseudocode

```
Fixed spectral mask - no adaptive noise floor
Input:  Sample frame x[0..N-1], calibration threshold Theta_0
Output: Trigger decision E[m]

  // Spectral estimation (identical to Algorithm 1)
  1.  X[k] <- |FFT(x)|   for k in {1,...,6}       // time to frequency, event band only, do reject EMI
  2.  X_max <- max_k{X[k]}                         // strongest bin magnitude, same as Algorithm 1

  // Fixed-threshold comparison (set once at deployment, never updated)
  3.  if X_max > Theta_0 then E[m] <- 1            // no persistence filter, single frame can trigger
  4.  return E[m]                                   // Theta_0 frozen and cannot follow noise drift
```


### 3) Line-by-Line Annotation

**Line 1: `X[k] ← |FFT(x)| for k ∈ {1,…,6}`**
Identical FFT and band selection as TSNFA (Algorithm 1, Lines 1–2). This correctly isolates the event band and rejects EMI, digital noise, and out-of-band thermal energy. STFT shares this critical advantage with TSNFA and the frequency-domain processing is a robust approach.

**Line 2: `Xmax ← maxk{X[k]}`**
Maximum in-band magnitude, identical to Algorithm 1 Line 2. At this point, STFT and TSNFA have the same detection statistic for the current frame.

**Line 3: `if Xmax > Θ0 then E[m] ← 1`** Compare against a **fixed** threshold $\Theta_0$ set during an initial calibration period. This is where STFT diverges critically from TSNFA. $\Theta_0$ was computed once and typically as the mean plus $3\sigma$ of in-band magnitude during a noise-only calibration window and it is never updated. Two defences present in Algorithm 1 are entirely absent:

First, there is no adaptive noise floor. Algorithm 1's gated EMA (Lines 9–10) continuously adjusts $\hat{N}$ to follow the drifting noise environment, so the threshold $\zeta \cdot \hat{N} $ rises and falls with the noise power. STFT has no such mechanism and $\Theta_0$ remains at the value it had at calibration time regardless of what happens to the noise afterwards.

Second, there is no persistence filter. Algorithm 1's $\gamma_d$-frame mean (Lines 3–5) requires energy to stay elevated across multiple consecutive frames before it can trigger a detection. STFT compares each frame independently and a single noise spike lasting one frame ($1.28\;\text{s} $) that happens to exceed $\Theta_0$ immediately produces a false trigger, with no opportunity for averaging to suppress it.

**Line 4: `return E[m]`**Return the trigger decision. $\Theta_0$ remains unchanged for the entire deployment.


### 4) Parameters

| Parameter | Symbol | Value | Reasoning |
|-----------|--------|-------|-----------|
| Threshold | $\Theta_0$ | Fixed at calibration | Set once by measuring noise spectrum. Typically mean $+ 3\sigma$ of calibration-period band magnitude. Never updated. |
| Band selection | $k \in \{1,\ldots,6\}$ | Same as TSNFA | Correctly isolates 1–5 Hz event band. This is why STFT achieves 100% DR. |
| No adaptation | - | - | Threshold frozen at calibration time. No EMA, no median, no gating. |
| No persistence | - | - | Single frame exceedance triggers. No $\gamma_d$ averaging. |


---


## E. Algorithm 5: Dual-Energy Dynamic-Range Detection: DEDaR (Hussein et al. 2022)

### 1) Overview

DEDaR (Dual-Energy Dynamic-Range) is a broadband energy-ratio detector. It computes the total energy in each sample frame across all frequencies and compares it against a smoothed long-term energy baseline. The method triggers when the short-term energy exceeds a fixed multiple of the long-term energy which is when a transient energy spike occurs, regardless of the spectral content that caused it.

**Operating principle.** The method maintains two energy estimates: a short-term (per-frame) energy $E_\text{short}(m)$ and a long-term smoothed baseline $E_\text{long}(m)$. The ratio $R(m) = E_\text{short}(m) / E_\text{long}(m)$ measures how much the current frame's energy deviates from the recent average. During steady-state noise, $R \approx 1.0$. Any transient energy increase from any source, at any frequency pushes $R$ above 1.0. An event is declared when $R > \zeta$.

**Short-term energy.** The total frame energy is:

$$E_\text{short}(m) = \sum_{n=0}^{L-1} |x_m[n]|^2$$

This is a **broadband** measurement. By Parseval's theorem, $E_\text{short}$ equals the total spectral energy $\sum_k |X_m[k]|^2$, summing energy from thermal noise at all frequencies, EMI at 60/120/180 Hz, digital switching bursts, wind, and the event signal. There is no frequency discrimination.

**Long-term energy.** The baseline is an exponential moving average:

$$E_\text{long}(m) = \beta_E \, E_\text{long}(m-1) + (1-\beta_E) \, E_\text{short}(m)$$

with $\beta_E = 0.95$ (time constant $\tau_E \approx 25.6\;\text{s}$). Unlike TSNFA's gated update, $E_\text{long}$ always updates including during events. This means events gradually inflate the baseline, reducing the energy ratio for subsequent events.

**Detection criterion.** The energy ratio and trigger decision are:

$$R(m) = \frac{E_\text{short}(m)}{E_\text{long}(m)}, \qquad E[m] = \begin{cases} 1 & \text{if } R(m) > \zeta \\ 0 & \text{otherwise} \end{cases}$$

The threshold $\zeta = 6.0$ requires a $6\times$ spike in total broadband energy which is equivalent to a 7.8 dB transient increase.

**Detects all events.** Events at 18 dB SNR inject massive broadband energy. The event amplitude $\sim 7.9\sqrt{P}$ over $\sim 384$ samples contributes energy on the order of $7.9^2 \times 384 \approx 24{,}000$ units to a single frame, compared to a baseline of $\sim L \times P = 128$. The resulting ratio spikes to $R \approx 63$–$190$, far exceeding $\zeta = 6$. DEDaR detects everything that generates energy.

**Extremely many false positives.** DEDaR triggers on *any* energy transient at *any* frequency. Digital switching bursts, wind gusts, EMI fluctuations, motor startups which all produce ratio spikes. The absence of band selection means that out-of-band noise transients (which TSNFA discards in the FFT step) contribute directly to the detection statistic. The 13,387,930 false positives ($\text{FAR} = 2{,}803\;\text{hr}^{-1}\text{node}^{-1}$) represent approximately 19.8% of all frames triggering falsely, or roughly one false trigger every 6.4 seconds. Precision is 0.3% and 997 out of every 1,000 triggers are false. 


### 2) Pseudocode

```
Energy-ratio triggering - no frequency-band selection
Input:  Sample frame x[0..N-1], long-term energy E_long(m-1)
Params: zeta = 6.0, beta_E = 0.95
Output: Trigger decision E[m]

  // Compute short-term and long-term energy (broadband, no FFT)
  1.  E_short(m) <- sum_n |x[n]|^2                 // total energy across ALL frequencies in one frame
  2.  E_long(m) <- beta_E * E_long(m-1)             // beta=0.95: 5% new, 95% old; tau ≈ 26 s
                   + (1-beta_E) * E_short(m)         // always updates; even during events

  // Energy-ratio trigger
  3.  R <- E_short(m) / E_long(m)                    // R ≈ 1.0 during steady noise; spikes on any transient
  4.  if R > zeta then E[m] <- 1                     // trigger on 6× energy spike from any source at any freq
  5.  return E[m]                                     // no band selection, no persistence, no gated update
```


### 3) Line-by-Line Annotation

**Line 1: `Eshort(m) ← Σn |x[n]|²`**
Compute the total energy in the current frame by summing the squared amplitude of all 128 samples. Unlike TSNFA which isolates 6 frequency bins, this sum includes everything: thermal noise across the full bandwidth, EMI at 60 Hz and its harmonics, digital switching bursts, and the event signal and all lumped into a single number. There is no way to tell which source contributed what. A frame where the event signal adds 100 units of energy looks identical to a frame where a digital burst adds the same 100 units.

**Line 2: `Elong(m) ← βE·Elong(m−1) + (1−βE)·Eshort(m)`**
Exponentially smooth the frame energy to create a long-term baseline. $\beta_E = 0.95$ gives a time constant of $\sim 26\;\text{s}$. $E_\text{long}$ represents the "expected" broadband energy per frame. Unlike TSNFA's noise floor (which only tracks in-band energy and freezes during events), $E_\text{long}$ always updates also during events. This means events gradually inflate $E_\text{long}$, reducing the ratio for subsequent events.

**Line 3: `R ← Eshort(m) / Elong(m)`**
The **energy ratio** which is DEDaR's core statistic. During steady-state noise (no events, no transients), $R \approx 1.0$ because $E_\text{short}$ tracks $E_\text{long}$. Any transient energy increase from any source, at any frequency will push $R$ above 1.0.

**Line 4: `if R > ζ then E[m] ← 1`**

Trigger when the energy ratio exceeds the threshold. To understand the scale: $R > 6 $ means the current frame contains $6\times$ more energy than the recent average. Real events easily exceed this and an event at $7.9\sqrt{P}$ amplitude over a full 128-sample frame produces a ratio spike of $\sim 63\times$, far above the $\zeta = 6$ threshold. This is why DEDaR achieves 100% detection rate.

The problem is the other direction: noise transients can also push $R$ above 6. A single noise source rarely does. For example, a digital burst at $2.0\sqrt{P}$ lasting 20 samples only produces $R \approx 1.6$. But multiple noise sources aligning in the same frame (an EMI peak coinciding with a digital burst during a high-noise phase) can combine to exceed the threshold, producing a false trigger. Because DEDaR has no band selection to exclude these out-of-band transients and no persistence filter to require sustained elevation, each such coincidence registers as a detection.

**Line 5: `return E[m]`**
Return trigger decision. $E_\text{long}$ is maintained internally.


### 4) Parameters

| Parameter | Symbol | Value | Reasoning |
|-----------|--------|-------|-----------|
| Short energy | $E_\text{short}$ | $\Sigma |x|^2$ | Total energy in current frame across ALL frequencies. |
| Long energy | $E_\text{long}$ | EMA ($\beta_E = 0.95$) | Smoothed baseline. Time constant $\sim 26\;\text{s}$. Always updates, even during events. |
| Ratio threshold | $\zeta$ | 6.0 | Trigger when $E_\text{short}/E_\text{long} > 6$, i.e., $6\times$ energy spike (7.8 dB transient increase). |
| No band selection | - | - | Both numerator and denominator include all spectral content. |


---


## F. Algorithm 6: Send-on-Delta Triggering: SoD (Correa et al. 2019)

### 1) Overview

Send-on-Delta (SoD) is a data-reduction protocol originally designed for slowly varying process-control signals (temperature, tank level, pressure). It transmits a sample only when the current value deviates from the last transmitted value by more than a fixed threshold $\Delta$. In process-control applications where the signal of interest changes on the order of degrees per hour and noise is negligible compared to $\Delta$, SoD achieves 90%+ traffic reduction while faithfully tracking the process variable. In our application, however, the noise amplitude is comparable to or exceeds the event amplitude, making SoD fundamentally unsuitable.

**Operating principle.** SoD operates sample-by-sample at $f_s = 100\;\text{Hz}$ and not frame-by-frame at $0.78\;\text{Hz}$. There is no temporal aggregation, no spectral analysis, and no noise-floor estimation. For each sample $x[n]$, the algorithm computes:

$$\Delta x[n] = |x[n] - x_\text{ref}|$$

where $x_\text{ref}$ is the last transmitted sample value. If $\Delta x > \Delta$, the sample is transmitted and the reference updates: $x_\text{ref} \leftarrow x[n]$.

**The reference random-walk problem.** The fatal flaw is that $x_\text{ref}$ updates to the current sample  which includes the noise component. In a noisy environment, this causes $x_\text{ref}$ to random-walk with the noise:

- If $\Delta$ is small relative to the noise RMS (say, $\Delta < 3\sigma_\text{noise}$), noise fluctuations frequently exceed $\Delta$, causing frequent transmissions. Each transmission updates $x_\text{ref}$ to a noise-contaminated value. After $N_\text{tx}$ transmissions, the reference has drifted by approximately $\sigma_\text{ref} \sim \Delta \sqrt{N_\text{tx}}$ from the true zero-signal baseline, at which point the reference no longer represents the quiescent state.

- If $\Delta$ is large relative to the noise ($\Delta > 3\sigma_\text{noise}$), noise rarely triggers, so $x_\text{ref}$ stays approximately stable. But events at amplitude $\sim 7.9\sqrt{P}$ only exceed $x_\text{ref} + \Delta$ if $x_\text{ref}$ is near zero *and* the event peak aligns with the measurement — which is unreliable given that $x_\text{ref}$ may have been set by a noise fluctuation.

**Issues with this model in this application.** No matter what value of $\Delta$ is chosen, SoD fails in the generated noise environment:

**$\Delta$ set too low (below noise amplitude).** If $\Delta < 3.0$, noise fluctuations routinely exceed the threshold. Every time that happens, $x_\text{ref}$ updates to the current noise value and drifts further from the true baseline. After many such updates, $x_\text{ref}$ has random-walked to an arbitrary level. When an event arrives, it must exceed $\Delta$ relative to wherever $x_\text{ref}$ has drifted and not relative to zero. Detection becomes a matter of luck where the reference happens to be in a favourable position when the event occurred.

**$\Delta$ set in the middle range (3.0–6.0).** Noise triggers are less frequent but still occur, so $x_\text{ref}$ still drifts but more slowly. The fundamental problem remains: the reference is unpredictable at the moment an event arrives.

**$\Delta$ set too high (above event amplitude).** If $\Delta > 8.0$, neither noise nor events produce deviations large enough to trigger. The result is 0 false positives but also 0 detections and the sensor is effectively switched off. This is also the outcome observed in the simulation.

**Result:** 0 out of 4,789 events detected, 0 false positives. This is not a parameter-tuning problem this is a fundamental architectural mismatch. SoD cannot operate in environments where the noise amplitude is comparable to the event signature.


### 2) Pseudocode

```
Delta-threshold - transmit only on deviation from last sent value
Input:  Current sample x[n], last transmitted value x_ref
Params: Delta (fixed delta threshold)
Output: Transmit decision, updated x_ref

  // Sample-by-sample comparison
  1.  if |x[n] - x_ref| > Delta then
  2.      Transmit x[n]
  3.      x_ref <- x[n]                           // update reference
      end if

  // Failure mode: noise updates x_ref -> reference random-walks
  // Result: events indistinguishable from noise deviations
  4.  return transmit decision
```


### 3) Line-by-Line Annotation

**Line 1: `if |x[n] − xref| > Δ then`**
Compare the current **individual sample** against the last transmitted value. Note: SoD operates sample-by-sample at 100 Hz, not frame-by-frame at 0.78 Hz. There is no temporal aggregation, no spectral analysis, and no noise-floor estimation. The core assumption is that $x_\text{ref}$ represents the "true" baseline and any deviation beyond $\Delta$ is significant. This assumption holds for slowly varying process signals (temperature: changes of 0.01°C/s; tank level: changes of mm/min) where noise is negligible compared to $\Delta$. It fails catastrophically when noise amplitude $\approx$ or $>$ event amplitude.

**Line 2: `Transmit x[n]`**
Send the current sample value to the sink. In the original SoD literature, this reduces network traffic by 90%+ for slowly drifting signals. In our environment, the transmission rate depends entirely on the relationship between $\Delta$ and the noise amplitude.

**Line 3: `xref ← x[n]`**
After every transmission, the reference updates to the current sample value which also includes the noise component. In a noisy environment: if $\Delta = 1.0$ (below noise RMS of $\sim 1.0$), nearly every sample triggers because $|w[n] - w[n-k]| > 1.0$ frequently. Each trigger updates $x_\text{ref}$ to the current noise value, causing $x_\text{ref}$ to random-walk. When an event arrives, $x_\text{ref}$ is at an arbitrary noise-determined level, and the event must exceed $\Delta$ relative to this random baseline and not relative to the true zero-signal level.

**Line 4: `return transmit decision`**
Return whether a transmission occurred. There is no explicit "event detection" and SoD is only a data-reduction scheme, not a detection algorithm. 


### 4) Parameters

| Parameter | Symbol | Value | Reasoning |
|-----------|--------|-------|-----------|
| Delta | $\Delta$ | Design choice | If $\Delta <$ noise amplitude: every fluctuation triggers, $x_\text{ref}$ random-walks. If $\Delta >$ event amplitude: events missed. No viable $\Delta$ exists. |
| Reference | $x_\text{ref}$ | Dynamic | Updated every transmission. In noisy environments, random-walks with noise, destroying baseline stability. |
| Sample-by-sample | - | - | No temporal averaging, no spectral analysis, no noise-floor estimation. $100\times$ more operations than frame-based methods. |


---


## G. Algorithm 7: TinyML Autoencoder Anomaly Detection (Hammad et al. 2023)

### 1) Overview

TinyML takes a fundamentally different approach from all previous algorithms. Instead of designing explicit signal-processing rules (FFT, thresholds, noise-floor tracking), it trains a neural network to learn what "normal" sensor data looks like, and flags anything that deviates as an anomaly.

**How it works.** An autoencoder is a neural network that takes a 128-sample frame as input and tries to reproduce the same 128 samples as output, but is forced through a narrow bottleneck in the middle. The architecture is symmetric:
$$
\text{Encoder: } \mathbf{x} \in \mathbb{R}^{128} \xrightarrow{W_1} \mathbb{R}^{32} \xrightarrow{W_2} \mathbb{R}^{8} \qquad \text{Decoder: } \mathbb{R}^{8} \xrightarrow{W_3} \mathbb{R}^{32} \xrightarrow{W_4} \hat{\mathbf{x}} \in \mathbb{R}^{128}
$$
with ReLU activations between layers. The bottleneck ($\mathbb{R}^{8} $) forces the network to compress each 128-sample frame into just 8 numbers, then reconstruct the full frame from those 8 numbers. During training on noise-only data, the network learns to compress and reconstruct noise efficiently. When an event frame arrives at deployment which is a pattern the network has never seen the reconstruction is poor, producing a large error, which is used as a detection.

The reconstruction error is the anomaly score:
$$
e(m) = \frac{1}{L} \sum_{n=0}^{L-1} \bigl(x_m[n] - \hat{x}_m[n]\bigr)^2 = \frac{1}{L} \| \mathbf{x}_m - \mathcal{A}_\theta(\mathbf{x}_m) \|_2^2
$$
Low $e(m)$ means the frame looks like training data (noise). High $e(m)$ means the frame contains something unfamiliar (potential event). A frame is declared anomalous when $e(m)$ exceeds a fixed threshold $\Theta_\text{ML}$, set during training as the 99th percentile of reconstruction errors on a noise-only validation set:
$$
\Theta_\text{ML} = \text{percentile}_{99}\bigl(\{e(m)\}_{m \in \mathcal{V}}\bigr)
$$
After training, both the network weights and $\Theta_\text{ML}$ are frozen and there is no online adaptation.

**Computational cost.** The network has $\sim 5{,}000 $ parameters and requires $\sim 20{,}000 $ multiply-accumulate operations per frame. On the STM32G071 (Cortex-M0+ at 64 MHz, no hardware FPU), this would require software-emulated floating point or quantised integer arithmetic, with estimated inference latency of $\sim 10 $–$30\;\text{ms} $ per frame. While this fits within the 1.28 s frame period, it consumes a significant fraction of the compute and power budget — compared with TSNFA's $\sim 100 $ arithmetic operations per frame.

**The stationarity problem which is same failure as STFT.** Like STFT (Algorithm 4), TinyML's failure is tied to a frozen reference point that cannot adapt to the operating environment. The autoencoder was trained at base noise power $P_0$. When the noise power drifts $\pm 6\;\text{dB}$ during deployment, both low-noise and high-noise phases produce frames that look different from the training data and not because an event is present, but because the noise level has changed. The network sees unfamiliar input, reconstructs it poorly, and $e(m)$ rises above $\Theta_\text{ML}$ for the duration of the drift.

This is the same mechanism that causes STFT's false positives: a threshold calibrated at one noise level becomes invalid when the noise level changes. STFT's threshold $\Theta_0$ is an explicit number; TinyML's threshold is implicit in the network's learned representation, but the effect is identical. Neither can track the drift.

**Highlights.** The autoencoder implicitly learns the noise spectrum during training and it develops an internal representation that is sensitive to spectral content, providing a form of spectral discrimination without an explicit FFT. This is why detection rate is high (99.7%): events produce reconstruction errors that are spectrally distinct from noise. But this implicit spectral awareness does not compensate for the frozen threshold, resulting in 5,465,607 false positives ($\text{FAR} = 1{,}144\;\text{hr}^{-1}\text{node}^{-1}$) and 14 false negatives.

**Retraining is not possible in real implementation.** The natural fix would be to retrain the autoencoder periodically to track the changing noise environment which is a neural-network equivalent of TSNFA's adaptive noise floor. But retraining requires backpropagation (gradient computation through all layers), which involves $\sim 5\times $ the computation of forward inference, memory for gradient storage, and a training dataset management strategy. On a Cortex-M0+ without FPU, this is prohibitively expensive. TSNFA achieves the goal of tracking the current operating point with a single EMA update requiring $\sim 3 $ arithmetic operations per frame.


### 2) Pseudocode

```
Autoencoder reconstruction-error triggering
Input:  Sample frame x[0..N-1], trained autoencoder A_theta
Params: Theta_ML (learned anomaly threshold)
Output: Trigger decision E[m]

  // Autoencoder inference
  1.  x_hat <- A_theta(x)                         // reconstruct input

  // Reconstruction error as anomaly score
  2.  e(m) <- ||x - x_hat||^2                     // MSE over frame

  // Trigger if error exceeds learned threshold
  3.  if e(m) > Theta_ML then E[m] <- 1

  // Limitation: Theta_ML fixed at training time; noise drift -> FP/FN
  4.  return E[m]
```

### 3) Line-by-Line Annotation

**Line 1: `x̂ ← Aθ(x)`**Feed the current 128-sample frame through the autoencoder to produce a reconstructed output. The bottleneck forces the network to represent each 128-sample frame using only 8 numbers. During training on noise-only data, the encoder learns which 8 numbers best summarise the structure of noise: its energy level, spectral shape, and typical fluctuation patterns. The decoder learns to reconstruct a full 128-sample frame from those 8 numbers. Together, the encoder and decoder develop a compact internal representation that is optimised for noise and nothing else. When an event frame arrives, its features cannot be captured by a representation tuned for noise, so the reconstruction fails and the error spikes. On the STM32G071 (Cortex-M0+ at 64 MHz, no hardware FPU), the $\sim 20{,}000$ multiply-accumulate operations require software integer arithmetic, with estimated inference time of $\sim 10 $–$30\;\text{ms} $ per frame within the 1.28 s frame budget and it is consuming far more computing time than TSNFA's $\sim 100$ operations.

**Line 2: `e(m) ← ||x − x̂||²`**Compute the mean squared error between the original frame and the reconstruction. This single number is the anomaly score. Low $e(m)$: the frame looks like training data (noise). High $e(m)$: the frame contains something the network cannot reproduce (potential event). Note that the error is computed over all 128 samples without any frequency selection and the autoencoder implicitly learns the noise spectrum during training, providing a form of spectral discrimination, but this is incorporated into the frozen weights rather than adapted at runtime.

**Line 3: `if e(m) > ΘML then E[m] ← 1`**Compare the anomaly score against the fixed threshold $\Theta_\text{ML}$. This threshold was set once during training (99th percentile of reconstruction errors on noise-only validation data) and never updates which is the same frozen-threshold problem as STFT's $\Theta_0$ (Algorithm 4). When the noise power drifts away from the training-time level, the reconstruction error rises even without an event present, and $\Theta_\text{ML}$ cannot adjust.

**Line 4: `return E[m]`** Return the trigger decision. Neither the network weights nor the threshold adapt during deployment. Every frame is judged against a model of noise that may no longer match the current environment.


### 4) Parameters

| Parameter | Symbol | Value | Reasoning |
|-----------|--------|-------|-----------|
| Autoencoder | $\mathcal{A}_\theta$ | Trained | $128 \rightarrow 32 \rightarrow 8 \rightarrow 32 \rightarrow 128$, ReLU, $\sim 5$K params. Trained on noise-only frames at $P_0 = 1.0$. |
| Threshold | $\Theta_\text{ML}$ | Learned | 99th percentile of training-set reconstruction errors. Frozen at deployment. |
| Training data | - | Fixed $P_0$ | Trained at base noise power $P_0 = 1.0$. Does not see the $\pm 6\;\text{dB}$ drift during deployment. |
| No adaptation | - | - | Neither network weights nor threshold update post-deployment. |


---

## H. Summary: Defence Composition and Failure Modes

The seven algorithms evaluated in this study differ in which of three signal-processing defences they employ. The following table maps each algorithm against these defences and the resulting simulation performance over a 24-hour, 200-node Monte Carlo run using the mean variant signal model described in the preamble.

| Algorithm                 | Band Selection                              | Adaptive Noise Floor                                         | Persistence Filter                       | DR         | FP         |
| ------------------------- | ------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------- | ---------- | ---------- |
| **TSNFA-mean (Alg. 1)**   | YES : FFT bins 1–6, max                     | YES : gated EMA ($\alpha = 0.984$)                           | YES : mean over $\gamma_d = 3$ frames    | **100.0%** | **0**      |
| **TSNFA-median (Alg. 2)** | YES : FFT bins 1–6, per-bin                 | YES : per-bin median ($\gamma_a = 64$), no gate              | YES : per-bin median over $\gamma_d = 3$ | **100.0%** | **0**      |
| Zhang (Alg. 3)            | NO : time-domain peak only                  | YES : EMA ($\beta = 0.95$), but tracks composite noise       | NO                                       | 73.4%      | 919,842    |
| STFT (Alg. 4)             | YES : FFT bins 1–6, max                     | NO : threshold $\Theta_0$ frozen at calibration              | NO                                       | 100.0%     | 399,822    |
| DEDaR (Alg. 5)            | NO : broadband energy, all frequencies      | PARTIAL : $E_\text{long}$ always updates, even during events | NO                                       | 100.0%     | 13,387,930 |
| SoD (Alg. 6)              | NO : sample-by-sample, no spectral analysis | NO : fixed $\Delta$, reference random-walks                  | NO                                       | 0.0%       | 0          |
| TinyML (Alg. 7)           | PARTIAL : learned implicitly, not explicit  | NO : threshold $\Theta_\text{ML}$ frozen at training time    | NO                                       | 99.7%      | 5,465,607  |

**Note on Algorithm 2:** The simulation evaluates only the mean variant (Algorithm 1). The median variant (Algorithm 2) is the form deployed in hardware. Both achieve identical results under these simulation conditions; the median variant's stronger outlier rejection becomes relevant over longer deployments or more adversarial noise environments.

### Missing defences and their consequences

Each row in the table that contains a "NO" correlates to a specific, observable failure mode in the simulation results:

**No band selection** (Zhang, DEDaR, SoD). Without an FFT to isolate the 1–5 Hz event band, the detection statistic includes energy from 60 Hz EMI, digital switching bursts, and broadband thermal noise. Zhang's time-domain peak detector sees composite noise that inflates the noise floor, causing both missed events (26.6%) and false triggers (919,842) when the floor fluctuates. DEDaR's broadband energy ratio triggers on any transient at any frequency, producing 13.4 million false positives (one every 6.4 seconds). SoD operates sample-by-sample with no frequency awareness and cannot distinguish event samples from noise samples, resulting in complete detection failure.

**No adaptive noise floor** (STFT, TinyML). Both methods calibrate their threshold at a single noise-power level and freeze it. STFT sets $\Theta_0$ explicitly during a calibration window; TinyML incorporates its threshold implicitly into the autoencoder's learned representation of "normal" noise. The effect is identical: when the noise power drifts $\pm 6\;\text{dB}$ from the calibration point, the frozen threshold is either too low (producing false positives during high-noise phases) or too high (producing false negatives during low-noise phases). STFT accumulates 399,822 false positives; TinyML accumulates 5,465,607. TSNFA's gated EMA (Algorithm 1) or not-gated median buffer (Algorithm 2) tracks the drift continuously, keeping the threshold calibrated to the current noise level.

**No persistence filter** (STFT, DEDaR, TinyML). Without a $\gamma_d $-frame mean or median requiring energy to stay elevated across consecutive frames, a single noise spike lasting one frame (1.28 s) can trigger a false detection. TSNFA's persistence filter attenuates single-frame spikes by a factor of $1/\gamma_d $ (mean variant) or rejects them entirely (median variant), while genuine events persisting across $\sim 4 $ frames pass through at full strength.

### The defence mechanisms are complementary

No single defence is sufficient. Band selection alone (STFT) eliminates out-of-band noise but cannot handle in-band noise drift. An adaptive noise floor alone (Zhang) tracks the noise but tracks the *wrong* noise because there is no spectral filtering. A persistence filter alone would suppress spikes but not slow drift. TSNFA combines all three in a specific order: first a band selection (discard irrelevant frequencies), the secondly a persistence mechanism (reject transient in-band spikes), and lastly an adaptive threshold (track the remaining slow drift). Each defence mechanism addresses a failure mode the others cannot.

### Computational perspective

The three defences in TSNFA require minimal computation: one 128-point FFT, a max or median across 6 values, a 3-frame mean or median, and a single EMA update or 64-frame median is totalling approximately 100–10,000 operations per frame depending on variant, well within the capability of the STM32G071 (Cortex-M0+ at 64 MHz). By contrast, TinyML requires $\sim 20{,}000$ multiply-accumulate operations per frame for comparable detection performance but with 5.5 million more false positives. The computational simplicity of the three-defence approach is not a limitation it is the reason TSNFA can run on resource-constrained hardware where neural-network alternatives struggle.