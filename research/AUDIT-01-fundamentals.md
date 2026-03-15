# Audit: 01-quantization-fundamentals.md

Mathematical formula verification for MXQ Research Document 01.

Auditor: Claude Opus 4.6 (1M context)
Date: 2026-03-14

---

## 1. Quantization / Dequantization Formulas

### Section 1.1 -- Core Mapping Function

**Quantize formula (line 18):**
$$q = \text{clamp}\!\left(\left\lfloor \frac{w}{s} \right\rceil + z,\; 0,\; 2^N - 1\right)$$

[VERIFIED] Correct for asymmetric (unsigned) quantization. The rounding-then-offset-then-clamp order is standard.

**Dequantize formula (line 22):**
$$\hat{w} = s \cdot (q - z)$$

[VERIFIED] Correct inverse of the quantize formula.

**Combined Q(w) formula (line 35):**
$$Q(w) = s \cdot \left(\text{clamp}\!\left(\left\lfloor \frac{w}{s} \right\rceil + z,\; 0,\; 2^N - 1\right) - z\right)$$

[VERIFIED] Correct composition of quantize then dequantize.

### Section 1.2 -- Symmetric Quantization

**Scale formula (line 45):**
$$s = \frac{\alpha}{2^{N-1} - 1}$$

[VERIFIED] Correct. For signed symmetric quantization, the positive range is $[0, 2^{N-1}-1]$, so the scale maps $\alpha$ to the max positive integer. The signed range is $[-2^{N-1}, 2^{N-1}-1]$, and the scale is set by the positive side to keep the zero-point at zero.

**Quantize formula (line 47):**
$$q = \text{clamp}\!\left(\left\lfloor \frac{w}{s} \right\rceil,\; -2^{N-1},\; 2^{N-1} - 1\right)$$

[VERIFIED] Correct. Signed integer range $[-2^{N-1}, 2^{N-1}-1]$.

**N=4 example (line 53):** "the integer range is $[-8, +7]$, giving 16 representable values. The scale is $s = \alpha / 7$."

[VERIFIED] $2^3 - 1 = 7$. Range $[-8, 7]$ has $16 = 2^4$ values. All correct.

### Section 1.3 -- Asymmetric Quantization

**Scale formula (line 68):**
$$s = \frac{w_{\max} - w_{\min}}{2^N - 1}$$

[VERIFIED] Correct. The unsigned range $[0, 2^N-1]$ has $2^N - 1$ intervals.

**Zero-point formula (line 70):**
$$z = \left\lfloor -\frac{w_{\min}}{s} \right\rceil$$

[VERIFIED] Correct. This ensures $w_{\min}$ maps to $q = 0$: $q = \lfloor w_{\min}/s \rceil + z = \lfloor w_{\min}/s \rceil + \lfloor -w_{\min}/s \rceil \approx 0$.

**Worked example (lines 83-89):** $w \in [-0.1, +0.5]$, $N = 4$.
- $s = 0.6/15 = 0.04$ [VERIFIED]
- $z = \lfloor 0.1/0.04 \rceil = \lfloor 2.5 \rceil = 3$ [VERIFIED] (round-half-up gives 3)

[ERROR] **Line 87:** "$w = 0.5$ maps to $q = \lfloor 0.5/0.04 \rceil + 3 = 13 + 3 = 16$, clamped to $15$"

$0.5 / 0.04 = 12.5$. Rounding $12.5$ to nearest integer gives $12$ or $13$ depending on tie-breaking convention (round-half-up gives 13, round-half-to-even gives 12). The document says $13$, which is acceptable under round-half-up. However, $13 + 3 = 16$ is correct, and clamping to $15$ is correct.

Reconstruction: $\hat{w} = 0.04 \times (15 - 3) = 0.04 \times 12 = 0.48$. Error = $0.5 - 0.48 = 0.02$. [VERIFIED]

**On closer inspection:** The rounding and clamp behavior is self-consistent. The note about "error of 0.02 for the max value" is correct. Marking this as verified with a caveat about the tie-breaking convention being implicit.

[VERIFIED] (with caveat: round-half-up tie-breaking is assumed but not stated)

### Section 1.6 -- Quantization Error

**MSE for uniform quantization (line 153):**
$$\mathbb{E}[(w - Q(w))^2] = \frac{\Delta^2}{12}$$

[VERIFIED] This is the variance of a uniform distribution on $[-\Delta/2, +\Delta/2]$, which is $\Delta^2/12$. Standard result.

**Output error formula (line 162):**
$$E_{\text{output}} = \|XW - XQ(W)\|_F^2 = \|X(W - Q(W))\|_F^2$$

[VERIFIED] Correct by linearity of matrix multiplication.

**Trace expansion (line 166):**
$$E_{\text{output}} = \text{tr}\!\left((W - Q(W))^T X^T X (W - Q(W))\right)$$

[VERIFIED] This follows from $\|A\|_F^2 = \text{tr}(A^T A)$ applied to $A = X(W - Q(W))$:
$\text{tr}((X(W-Q(W)))^T \cdot X(W-Q(W))) = \text{tr}((W-Q(W))^T X^T X (W-Q(W)))$.

**Objective (line 174):**
$$\min_{Q} \; \text{tr}\!\left((W - Q(W))^T H (W - Q(W))\right)$$

[VERIFIED] Correct with $H = \mathbb{E}[X^T X]$.

---

## 2. Bit-Width Math

### Section 2.1 -- Representable Values Table (lines 186-194)

[VERIFIED] All values in the table are correct. Unsigned range $[0, 2^N-1]$, signed range $[-2^{N-1}, 2^{N-1}-1]$.

### Section 2.2 -- Step Size and Precision Table (lines 200-211)

**Step size formula:**
$$\Delta = \frac{R}{2^N - 1}$$

[VERIFIED] Correct for $R = w_{\max} - w_{\min}$.

**Table values (for $R = 1.0$):**
- $N=2$: $\Delta = 1/3 = 0.3333$ [VERIFIED]
- $N=3$: $\Delta = 1/7 = 0.1429$ [VERIFIED]
- $N=4$: $\Delta = 1/15 = 0.0667$ [VERIFIED]
- $N=5$: $\Delta = 1/31 = 0.0323$ [VERIFIED]
- $N=6$: $\Delta = 1/63 = 0.0159$ [VERIFIED]
- $N=8$: $\Delta = 1/255 = 0.00392$ [VERIFIED]

**Relative precision column:**
- $N=3$ vs $N=2$: $(1/3)/(1/7) = 7/3 = 2.33$ [VERIFIED] (document says 2.3x)
- $N=4$ vs $N=2$: $(1/3)/(1/15) = 15/3 = 5.0$ [VERIFIED]
- $N=5$ vs $N=2$: $(1/3)/(1/31) = 31/3 = 10.33$ [VERIFIED] (document says 10.3x)
- $N=6$ vs $N=2$: $(1/3)/(1/63) = 63/3 = 21.0$ [VERIFIED]
- $N=8$ vs $N=2$: $(1/3)/(1/255) = 255/3 = 85.0$ [VERIFIED]

### Section 2.3 -- SQNR Formula (lines 225-234)

**The SQNR formula presented (line 227):**
$$\text{SQNR}(N) \approx 6.02N + 4.35 - 10\log_{10}\!\left(\frac{12 \cdot \text{loading factor}}{(2^N - 1)^2}\right) \text{ dB}$$

[ERROR] This formula is garbled. It conflates two different standard forms and is not self-consistent.

**The standard SQNR formulas are:**

**(A) For a full-range sinusoidal input with uniform N-bit quantization:**
$$\text{SQNR} = 6.02N + 1.76 \text{ dB}$$

This is the classic ADC formula. The 1.76 dB comes from $10\log_{10}(3/2)$. This applies to a sinusoidal signal that spans the full quantizer range.

**(B) For an arbitrary signal with loading factor $L$ (ratio of signal RMS to full-scale RMS):**
$$\text{SQNR} = 6.02N + 1.76 + 20\log_{10}(L) \text{ dB}$$

where $L \leq 1$. For a Gaussian signal (which is the relevant case here), the loading factor penalty is significant because the signal rarely reaches full scale.

**(C) The general form for uniform quantization with step size $\Delta$ and signal variance $\sigma^2$:**
$$\text{SQNR} = 10\log_{10}\!\left(\frac{\sigma^2}{\Delta^2/12}\right) = 10\log_{10}\!\left(\frac{12\sigma^2}{\Delta^2}\right)$$

For a full-range signal with $\Delta = R/(2^N - 1)$ and $\sigma^2 = R^2/12$ (uniform distribution over the range), this gives $\text{SQNR} = 10\log_{10}((2^N-1)^2) \approx 6.02N$ dB for large $N$.

The formula as written in the document has `$6.02N + 4.35$` as a leading term. The constant $4.35$ dB does not correspond to any standard result. The standard constant for the sinusoidal case is $1.76$ dB. For a uniformly distributed signal it is $0$ dB. The $4.35$ appears to be incorrect.

Furthermore, the formula then subtracts $10\log_{10}(\frac{12 \cdot \text{loading factor}}{(2^N-1)^2})$, which double-counts the $6.02N$ term already present.

**Correction:** The document should either use:

For a sinusoidal input: $\text{SQNR} = 6.02N + 1.76$ dB

Or for a general input with loading factor adjustment:
$$\text{SQNR} = 10\log_{10}\!\left(\frac{12\sigma^2 (2^N - 1)^2}{R^2}\right) \text{ dB}$$

where $R$ is the quantizer full-scale range and $\sigma^2$ is the signal variance.

**Approximate SQNR values (lines 230-232):**

The document claims:
- 2-bit: ~13.6 dB
- 4-bit: ~25.8 dB
- 8-bit: ~50.2 dB

For a Gaussian source optimally clipped, reasonable SQNR values are approximately:
- 2-bit: ~9.25 dB (from MSE $\approx 0.1175\sigma^2$, SQNR = $10\log_{10}(1/0.1175) \approx 9.3$ dB)
- 4-bit: ~20.2 dB (from MSE $\approx 0.009497\sigma^2$, SQNR = $10\log_{10}(1/0.009497) \approx 20.2$ dB)
- 8-bit: ~46.5 dB (from MSE $\approx 0.0000226\sigma^2$, SQNR = $10\log_{10}(1/0.0000226) \approx 46.5$ dB)

[ERROR] The approximate SQNR values in the document are inflated by about 3-4 dB compared to what the document's own MSE figures from Section 2.4 would imply. However, the exact values depend on the specific assumptions (optimal clipping vs. full-range, Gaussian vs. uniform input). The qualitative point (12 dB difference between 2-bit and 4-bit) is approximately correct.

**The difference between 4-bit and 2-bit (line 234):**
"Going from 4-bit to 2-bit loses about 12 dB" [VERIFIED] -- this is correct regardless of the formula, since each bit contributes ~6 dB.

### Section 2.4 -- MSE as Function of Bit Width (lines 246-270)

**MSE formula (line 252):**
$$\text{MSE} = \frac{\Delta^2}{12} = \frac{R^2}{12(2^N - 1)^2}$$

[VERIFIED] Correct substitution of $\Delta = R/(2^N-1)$.

**Large-N approximation (lines 254-256):**
$(2^N-1)^2 \approx 2^{2N}$, so $\text{MSE} \propto R^2 \cdot 4^{-N}$.

[VERIFIED] Correct.

**"Each additional bit reduces MSE by a factor of 4 (6 dB)" (line 258):**

[VERIFIED] $4^{-(N+1)}/4^{-N} = 1/4$. And $10\log_{10}(4) = 6.02$ dB.

**Optimal clipping values for Gaussian (lines 267-270):**
- $N = 2$: $k \approx 1.71$, MSE $\approx 0.1175\sigma^2$
- $N = 3$: $k \approx 2.15$, MSE $\approx 0.03454\sigma^2$
- $N = 4$: $k \approx 2.51$, MSE $\approx 0.009497\sigma^2$
- $N = 8$: $k \approx 3.29$, MSE $\approx 0.0000226\sigma^2$

[VERIFIED] These values are consistent with published optimal clipping results for Gaussian sources (e.g., from Max's 1960 paper and subsequent numerical optimizations). The clipping ratios and MSE values are well-established in quantization theory literature.

### Section 2.5 -- Bit Packing (lines 272-345)

**Values per byte table (lines 278-285):**

[ERROR] **Line 283, 5-bit row:** "Values per 32-bit Word: 6 (30 bits used)" -- $6 \times 5 = 30$ bits, correct. [VERIFIED]

[ERROR] **Line 284, 6-bit row:** "Values per 32-bit Word: 5 (30 bits used)" -- $5 \times 6 = 30$ bits, correct. [VERIFIED]

Actually, all values in this table check out:
- 2-bit: 4 per byte (8/2=4), 16 per 32-bit word (32/2=16), 0 wasted [VERIFIED]
- 3-bit: 2 per byte (6 bits used, 2 wasted), 10 per 32-bit word (30 bits used, 2 wasted) [VERIFIED]
- 4-bit: 2 per byte (8/4=2), 8 per 32-bit word (32/4=8), 0 wasted [VERIFIED]
- 5-bit: 1 per byte (5 bits used, 3 wasted), 6 per 32-bit word (30 bits used, 2 wasted) [VERIFIED]
- 6-bit: 1 per byte (6 bits used, 2 wasted), 5 per 32-bit word (30 bits used, 2 wasted) [VERIFIED]
- 8-bit: 1 per byte, 4 per 32-bit word, 0 wasted [VERIFIED]

**Tight packing formulas (lines 296-297):**
- Total bits: $K \times N$ [VERIFIED]
- Total bytes: $\lceil K \times N / 8 \rceil$ [VERIFIED]

**Extraction pseudocode (lines 300-306):** [VERIFIED] Standard bit-stream extraction algorithm.

**Worked example: packing four 3-bit values (lines 310-330):**

Values: $[5, 3, 7, 2]$ = $[101, 011, 111, 010]$

Total bits: $4 \times 3 = 12$ bits, needing $\lceil 12/8 \rceil = 2$ bytes.

[ERROR] **Line 320:** "Four bits wasted." This should say "four bits unused" (or "four bits of padding"). $16 - 12 = 4$. The number is correct, the terminology is fine.

[VERIFIED] The packing layout and extraction example are correct.

**Block packing table for B=64 (lines 338-345):**

| Bit Width | Bits per Block | 32-bit Words | Bytes |
|:-:|:-:|:-:|:-:|
| 2 | 128 | 4 | 16 |
| 3 | 192 | 6 | 24 |
| 4 | 256 | 8 | 32 |
| 5 | 320 | 10 | 40 |
| 6 | 384 | 12 | 48 |
| 8 | 512 | 16 | 64 |

Checking: $64 \times N$ bits, divided by 32 for words, divided by 8 for bytes:
- $64 \times 2 = 128$ bits, $128/32 = 4$ words, $128/8 = 16$ bytes [VERIFIED]
- $64 \times 3 = 192$ bits, $192/32 = 6$ words, $192/8 = 24$ bytes [VERIFIED]
- $64 \times 4 = 256$ bits, $256/32 = 8$ words, $256/8 = 32$ bytes [VERIFIED]
- $64 \times 5 = 320$ bits, $320/32 = 10$ words, $320/8 = 40$ bytes [VERIFIED]
- $64 \times 6 = 384$ bits, $384/32 = 12$ words, $384/8 = 48$ bytes [VERIFIED]
- $64 \times 8 = 512$ bits, $512/32 = 16$ words, $512/8 = 64$ bytes [VERIFIED]

---

## 3. Scale and Zero-Point Computation

### Section 3.2 -- Min-Max Scaling Worked Example (lines 380-389)

**Scale:** $s = 1.1/15 = 0.07\overline{3}$ [VERIFIED]
**Zero-point:** $z = \lfloor 0.3/0.0733 \rceil = \lfloor 4.09 \rceil = 4$ [VERIFIED] ($0.3/0.07333 = 4.091$)

**Comparison claim (line 389):** "Min-max gives 56% better resolution because it does not waste grid points on the empty region $[-0.8, -0.3]$."

Absmax step size: $0.8/7 = 0.1143$. Min-max step size: $0.0733$.
Improvement: $1 - 0.0733/0.1143 = 35.8\%$. Alternatively, $0.1143/0.0733 = 1.559$, so min-max is 55.9% better (i.e., 1.559x finer resolution).

[VERIFIED] The "56% better" claim is correct (interpreted as 56% finer step size, i.e., $\Delta_{\text{absmax}}/\Delta_{\text{minmax}} \approx 1.56$).

### Section 3.3 -- Block Overhead (lines 406-417)

**Overhead ratio (line 413):**
$$\text{overhead ratio} = \frac{32}{N \cdot B}$$

[VERIFIED] Metadata is 32 bits per block; weight data is $N \cdot B$ bits per block.

**Specific values:**
- $N=2, B=64$: $32/(2 \times 64) = 32/128 = 0.25 = 25\%$ [VERIFIED]
- $N=4, B=64$: $32/(4 \times 64) = 32/256 = 0.125 = 12.5\%$ [VERIFIED]
- $N=2, B=32$: $32/(2 \times 32) = 32/64 = 0.50 = 50\%$ [VERIFIED]

### Section 3.4 -- GGUF Q2_K Super-Block (lines 423-448)

**Effective bits per weight (line 433):**
$84 \times 8 / 256 = 672 / 256 = 2.625$ bits/weight.

[VERIFIED] The byte breakdown: $2 + 2 + 8 + 8 + 64 = 84$ bytes. $84 \times 8 = 672$ bits. $672 / 256 = 2.625$ bpw.

### Section 3.5 -- Optimal Scale Factor (lines 451-486)

**MSE formula for Gaussian with clipping (line 474):**
$$\text{MSE}(c) = \underbrace{2\sigma^2 \!\int_{c}^{\infty}\! (w - c)^2 \phi(w/\sigma) \, dw}_{\text{clipping error}} + \underbrace{\frac{c^2}{3(2^{N-1} - 1)^2} \left(1 - 2\Phi(-c/\sigma)\right)}_{\text{granularity error}}$$

[ERROR] The clipping error integral has a dimensional issue. The term $2\sigma^2$ in front is incorrect. The correct clipping error for a $\mathcal{N}(0, \sigma^2)$ distribution clipped at $\pm c$ is:

$$E_{\text{clip}} = 2 \int_{c}^{\infty} (w - c)^2 \frac{1}{\sigma}\phi(w/\sigma) \, dw$$

Note: $\phi(w/\sigma)$ is the standard normal PDF evaluated at $w/\sigma$, which equals $\frac{1}{\sqrt{2\pi}}e^{-w^2/(2\sigma^2)}$. The actual PDF of $\mathcal{N}(0, \sigma^2)$ is $\frac{1}{\sigma}\phi(w/\sigma)$. So the integral should have $\frac{1}{\sigma}\phi(w/\sigma)$, not $\sigma^2 \cdot \phi(w/\sigma)$.

The factor $2\sigma^2$ appears to conflate a normalization factor with the variance. The correct expression is:

$$E_{\text{clip}} = 2 \int_{c}^{\infty} (w - c)^2 \cdot \frac{1}{\sigma}\phi\!\left(\frac{w}{\sigma}\right) dw$$

After substitution $t = w/\sigma$, this becomes:

$$E_{\text{clip}} = 2\sigma^2 \int_{c/\sigma}^{\infty} (t - c/\sigma)^2 \phi(t)\, dt$$

So IF $\phi$ in the document is interpreted as the standard normal PDF $\phi(t) = \frac{1}{\sqrt{2\pi}}e^{-t^2/2}$ (not $\phi(w/\sigma)$), then the factor $2\sigma^2$ makes sense but the argument to $\phi$ must be $w/\sigma$ as the integration variable, with the Jacobian already absorbed. The document writes the integrand as $(w - c)^2 \phi(w/\sigma) \, dw$ with a prefactor $2\sigma^2$. This is dimensionally off: $(w-c)^2$ has dimension $[\text{weight}]^2$, $\phi(w/\sigma)$ is dimensionless, $dw$ has dimension $[\text{weight}]$, and $2\sigma^2$ has dimension $[\text{weight}]^2$, giving total dimension $[\text{weight}]^5$. The MSE should have dimension $[\text{weight}]^2$.

**Correction:** The clipping error should be:

$$E_{\text{clip}} = 2 \int_{c}^{\infty} (w - c)^2 \cdot \frac{1}{\sigma}\phi\!\left(\frac{w}{\sigma}\right) dw$$

or equivalently, with a change of variable $u = w/\sigma$:

$$E_{\text{clip}} = 2\sigma^2 \int_{c/\sigma}^{\infty} (u - c/\sigma)^2 \phi(u)\, du$$

The document's formula can be made correct if we interpret the integral variable as $u = w/\sigma$ (dimensionless), so $dw = \sigma\,du$ and $(w-c)^2 = \sigma^2(u-c/\sigma)^2$. But as literally written (integral in $w$ with $\phi(w/\sigma)$), the $2\sigma^2$ prefactor should be $2/\sigma$ (to provide the missing $1/\sigma$ normalization of the PDF).

For the granularity error term: For symmetric $N$-bit quantization with range $[-c, c]$, the step size is $\Delta = 2c/(2^N - 1)$, and $\text{MSE}_{\text{gran}} = \Delta^2/12 \cdot P(\text{not clipped})$.

$\Delta = 2c/(2^N - 1)$, so $\Delta^2/12 = 4c^2/(12(2^N-1)^2) = c^2/(3(2^N-1)^2)$.

But wait -- for symmetric quantization with $s = c/(2^{N-1}-1)$, the step size is $\Delta = s = c/(2^{N-1}-1)$, so $\Delta^2/12 = c^2/(12(2^{N-1}-1)^2)$.

The document writes $\frac{c^2}{3(2^{N-1} - 1)^2}$ for the granularity coefficient.

[ERROR] This should be $\frac{c^2}{12(2^{N-1} - 1)^2}$, not $\frac{c^2}{3(2^{N-1} - 1)^2}$.

The step size for symmetric N-bit quantization is $s = c/(2^{N-1}-1)$ (mapping $c$ to the max positive integer $2^{N-1}-1$). The quantization MSE within a bin is $s^2/12 = c^2/(12(2^{N-1}-1)^2)$.

The denominator should have $12$, not $3$. The factor of $3$ would correspond to using $\Delta = 2c/(2^{N-1}-1)$ and then $\Delta^2/12 = 4c^2/(12(2^{N-1}-1)^2) = c^2/(3(2^{N-1}-1)^2)$, but that doubled step size is incorrect -- the step size is $c/(2^{N-1}-1)$, not $2c/(2^{N-1}-1)$.

**However**, there is an alternative convention where the symmetric range $[-c, c]$ is divided into $2^N - 1$ intervals (not $2(2^{N-1}-1) = 2^N - 2$ intervals), giving $\Delta = 2c/(2^N - 1)$ and $\Delta^2/12 = 4c^2/(12(2^N-1)^2) = c^2/(3(2^N-1)^2)$.

The document uses $(2^{N-1}-1)$ in the denominator rather than $(2^N - 1)$, which is the convention consistent with $s = c/(2^{N-1}-1)$ and $\Delta = s$. Under that convention, the coefficient should be $1/12$, not $1/3$.

**Summary of this error:** The granularity term has $3$ where it should have $12$ (factor of 4 too large).

**The probability factor $(1 - 2\Phi(-c/\sigma))$:** This is $P(|W| \leq c) = P(-c \leq W \leq c)$ for $W \sim \mathcal{N}(0, \sigma^2)$. By symmetry, $P(|W| \leq c) = 1 - 2P(W > c) = 1 - 2\Phi(-c/\sigma)$... actually:

$P(W > c) = 1 - \Phi(c/\sigma)$, and $\Phi(-c/\sigma) = 1 - \Phi(c/\sigma)$. So $P(|W| \leq c) = 1 - 2(1-\Phi(c/\sigma)) = 2\Phi(c/\sigma) - 1 = 1 - 2\Phi(-c/\sigma)$. [VERIFIED] -- the probability factor is correct.

**Optimal clipping table (lines 480-485):**

| $N$ | Optimal $c/\sigma$ | Fraction Clipped |
|:-:|:-:|:-:|
| 2 | 1.71 | 8.7% |
| 3 | 2.15 | 3.2% |
| 4 | 2.51 | 1.2% |
| 8 | 3.29 | 0.10% |

For a $\mathcal{N}(0,1)$: $P(|W| > k) = 2(1 - \Phi(k))$:
- $k = 1.71$: $P = 2(1 - \Phi(1.71)) = 2(1 - 0.9564) = 0.0872 = 8.7\%$ [VERIFIED]
- $k = 2.15$: $P = 2(1 - \Phi(2.15)) = 2(1 - 0.9842) = 0.0316 = 3.2\%$ [VERIFIED]
- $k = 2.51$: $P = 2(1 - \Phi(2.51)) = 2(1 - 0.9940) = 0.0120 = 1.2\%$ [VERIFIED]
- $k = 3.29$: $P = 2(1 - \Phi(3.29)) = 2(1 - 0.9995) = 0.0010 = 0.10\%$ [VERIFIED]

---

## 4. GPTQ and Rounding

### Section 4.2 -- GPTQ Update Rule (line 524)

$$w_j \leftarrow w_j - \frac{\delta_i \cdot [H^{-1}]_{ij}}{[H^{-1}]_{ii}}$$

[VERIFIED] This is the standard OBS (Optimal Brain Surgeon) update rule applied column-by-column. When weight $i$ is quantized with error $\delta_i$, the optimal compensation for weight $j$ is $-\delta_i [H^{-1}]_{ij}/[H^{-1}]_{ii}$, derived from the condition that minimizes the Hessian-weighted error increase.

### Section 4.3 -- AdaRound (lines 540-558)

**Rounding interpolation (line 544):**
$$\hat{w}_i = s \cdot \left(\lfloor w_i/s \rfloor + h(v_i)\right)$$

[VERIFIED] Correct. $h(v_i) \in [0, 1]$ interpolates between floor and ceil.

**Stretched sigmoid:** $h(v) = \text{clamp}(\sigma(v) \cdot (\zeta - \gamma) + \gamma, 0, 1)$ with $\zeta = 1.1$, $\gamma = -0.1$.

[VERIFIED] Consistent with the AdaRound paper (Nagel et al., 2020). The stretched sigmoid maps to $[-0.1, 1.1]$ before clamping, which encourages convergence to $\{0, 1\}$.

**Loss function (line 550):**
$$\min_{\{v_i\}} \; \|XW - X\hat{W}\|_F^2 + \lambda \sum_i \beta \cdot h(v_i) \cdot (1 - h(v_i))$$

[VERIFIED] The regularizer $h(v)(1-h(v))$ penalizes values near $0.5$ (maximum penalty) and vanishes at $0$ and $1$. This is the standard AdaRound formulation. (Note: in the original paper, $\beta$ is an annealing parameter that increases during training. The document's use of both $\lambda$ and $\beta$ is slightly redundant but not incorrect.)

### Section 4.4 -- Error Bound (line 583)

$$\|y - \hat{y}\| \leq \sum_{l=1}^{L} \|W_l - \hat{W}_l\| \cdot \prod_{k=l+1}^{L} \|\hat{W}_k\| \cdot \|x\|$$

[VERIFIED] This is a standard perturbation bound for feedforward networks. For $y = W_L \cdots W_1 x$ and $\hat{y} = \hat{W}_L \cdots \hat{W}_1 x$, a telescoping decomposition gives this bound (assuming no nonlinearities, or with 1-Lipschitz nonlinearities absorbed into the norms). The product $\prod_{k=l+1}^L \|\hat{W}_k\|$ captures error amplification through subsequent layers.

---

## 5. Information Theory

### Section 5.1 -- Rate-Distortion Function (lines 600-606)

**Gaussian R(D) (line 602):**
$$R(D) = \begin{cases} \frac{1}{2} \log_2 \frac{\sigma^2}{D} & \text{if } D \leq \sigma^2 \\ 0 & \text{if } D > \sigma^2 \end{cases}$$

[VERIFIED] This is the textbook Gaussian rate-distortion function (Cover & Thomas, Chapter 10, Theorem 10.3.2). Correct.

**Distortion-rate function (line 606):**
$$D(R) = \sigma^2 \cdot 2^{-2R}$$

[VERIFIED] Correct. Inverting $R = \frac{1}{2}\log_2(\sigma^2/D)$ gives $2R = \log_2(\sigma^2/D)$, so $\sigma^2/D = 2^{2R}$, thus $D = \sigma^2 \cdot 2^{-2R}$.

**"Each additional bit per weight reduces the MSE by a factor of $2^2 = 4$ (6.02 dB)" (line 608):**

[VERIFIED] From $D(R+1)/D(R) = 2^{-2(R+1)}/2^{-2R} = 2^{-2} = 1/4$. And $10\log_{10}(4) = 6.02$ dB.

### Section 5.2 -- Shannon Entropy (lines 617-643)

**Differential entropy of Gaussian (line 625):**
$$h(W) = \frac{1}{2} \log_2(2\pi e \sigma^2)$$

[VERIFIED] Standard result. The differential entropy of $\mathcal{N}(0, \sigma^2)$ is $\frac{1}{2}\ln(2\pi e \sigma^2)$ nats $= \frac{1}{2}\log_2(2\pi e \sigma^2)$ bits.

**Numerical example for $\sigma = 0.01$ (line 629):**
$h(W) = \frac{1}{2}\log_2(2\pi e \times 10^{-4})$

$2\pi e \approx 17.0795$, so $2\pi e \times 10^{-4} = 1.70795 \times 10^{-3}$.

$\log_2(1.70795 \times 10^{-3}) = \log_2(1.70795) + \log_2(10^{-3}) = 0.7723 + (-3 \times 3.3219) = 0.7723 - 9.9658 = -9.1935$.

$h(W) = -9.1935/2 = -4.597 \approx -4.60$ bits.

The document says "$\frac{1}{2}\log_2(1.71 \times 10^{-3}) \approx \frac{1}{2}(-9.19) = -4.60$".

$\log_2(1.71 \times 10^{-3}) = \log_2(1.71) - 3\log_2(10) = 0.774 - 9.966 = -9.192$. So $\frac{1}{2}(-9.192) = -4.596$.

[VERIFIED] All numerical values check out.

### Section 5.4 -- Lloyd-Max Quantizer (lines 672-702)

**Distortion functional (line 678):**
$$D = \sum_{k=0}^{K-1} \int_{b_k}^{b_{k+1}} (w - c_k)^2 \, p(w) \, dw$$

[VERIFIED] Correct definition of the expected squared-error distortion for a quantizer with levels $c_k$ and boundaries $b_k$.

**Nearest-neighbor rule (line 683):**
$$b_k = \frac{c_{k-1} + c_k}{2}, \quad k = 1, \ldots, K-1$$

[VERIFIED] The optimal boundary between two levels is their midpoint (for squared-error distortion).

**Centroid condition (line 686):**
$$c_k = \frac{\int_{b_k}^{b_{k+1}} w \, p(w) \, dw}{\int_{b_k}^{b_{k+1}} p(w) \, dw} = \mathbb{E}[W \mid b_k \leq W < b_{k+1}]$$

[VERIFIED] The optimal level is the conditional mean (centroid) of the source within each partition. This is the standard necessary condition.

**Optimal 2-bit Lloyd-Max levels for Gaussian (line 696):**
$$c \in \{-1.510\sigma, -0.4528\sigma, +0.4528\sigma, +1.510\sigma\}$$

[VERIFIED] These are the well-known optimal 2-bit Lloyd-Max quantizer levels for a Gaussian source. The values match published tables (e.g., Max 1960, Table I).

**Comparison with uniform 2-bit symmetric (line 698):**
"with optimal clipping at $1.71\sigma$, gives $c \in \{-1.71\sigma, -0.57\sigma, +0.57\sigma, +1.71\sigma\}$"

For symmetric 2-bit with $\alpha = 1.71\sigma$: $s = 1.71\sigma / (2^1 - 1) = 1.71\sigma / 1 = 1.71\sigma$. The levels for signed 2-bit are $\{-1, 0, 0, 1\} \times s$... wait, let me reconsider.

For 2-bit signed symmetric: integers $\{-2, -1, 0, 1\}$ (range $[-2, 1]$, which is asymmetric) -- actually, for 2-bit signed: $[-2^{N-1}, 2^{N-1}-1] = [-2, 1]$.

With the convention from Section 1.2: $s = \alpha/(2^{N-1}-1) = 1.71\sigma/(2^1 - 1) = 1.71\sigma/1 = 1.71\sigma$. Levels: $\{-2, -1, 0, 1\} \times 1.71\sigma = \{-3.42\sigma, -1.71\sigma, 0, 1.71\sigma\}$. This does not match what the document claims.

The document seems to be using a different convention for 2-bit symmetric: representable values $\{-3s, -s, +s, +3s\}$ (from Section 2.3, line 219). This corresponds to unsigned integers $\{0, 1, 2, 3\}$ mapped as $q - 1.5$ (centered), or equivalently a custom mapping with 4 evenly-spaced levels symmetric about zero. With this convention, $s = 1.71\sigma / 3 = 0.57\sigma$, and levels are $\{-1.71\sigma, -0.57\sigma, +0.57\sigma, +1.71\sigma\}$.

[VERIFIED] Under the convention that 2-bit symmetric uses levels $\{-3s, -s, +s, +3s\}$ (4 equally-spaced levels centered at zero with spacing $2s$), the claim is correct. Note: this convention differs from the one in Section 1.2 (which uses the standard signed integer range $[-2, 1]$). The document is internally slightly inconsistent about the 2-bit symmetric convention, but the mathematical claim about the levels is valid under the stated convention.

**Lloyd-Max MSE vs uniform MSE (line 700):**
"Lloyd-Max achieves MSE of $0.1175\sigma^2$ for 2-bit Gaussian, while uniform achieves $0.1188\sigma^2$ -- only 1.1% worse."

$(0.1188 - 0.1175)/0.1175 = 0.0013/0.1175 = 1.1\%$. [VERIFIED]

Note: The $0.1175\sigma^2$ for Lloyd-Max matches the well-known result. The $0.1188\sigma^2$ for uniform with optimal clipping is also a standard result. The near-equality for the Gaussian case is an important and correct observation.

### Section 5.5 -- Mixed-Precision Optimality (lines 704-746)

**Total distortion (line 710):**
$$D_{\text{total}} = \sum_l \rho_l \cdot \sigma_l^2 \cdot 2^{-2R_l}$$

[VERIFIED] Follows from applying the rate-distortion bound $D_l(R_l) = \sigma_l^2 \cdot 2^{-2R_l}$ to each layer.

**Lagrangian (line 716):**
$$\mathcal{L} = \sum_l \rho_l \sigma_l^2 \cdot 2^{-2R_l} + \lambda \left(\sum_l n_l R_l - B_{\text{total}}\right)$$

[VERIFIED] Standard Lagrangian for constrained optimization.

**Derivative (line 720):**
$$\frac{\partial \mathcal{L}}{\partial R_l} = -2 \ln 2 \cdot \rho_l \sigma_l^2 \cdot 2^{-2R_l} + \lambda n_l = 0$$

[VERIFIED] $\frac{d}{dR}(2^{-2R}) = -2\ln 2 \cdot 2^{-2R}$. Correct.

**Solution (line 722):**
$$2^{-2R_l} = \frac{\lambda n_l}{2 \ln 2 \cdot \rho_l \sigma_l^2}$$

[VERIFIED] Direct rearrangement of the first-order condition.

**Bit allocation (line 724):**
$$R_l = \frac{1}{2} \log_2 \frac{2 \ln 2 \cdot \rho_l \sigma_l^2}{\lambda n_l}$$

[VERIFIED] Taking $\log_2$ of both sides of $2^{-2R_l} = \frac{\lambda n_l}{2\ln 2 \cdot \rho_l \sigma_l^2}$:

$-2R_l = \log_2\!\left(\frac{\lambda n_l}{2\ln 2 \cdot \rho_l \sigma_l^2}\right) = -\log_2\!\left(\frac{2\ln 2 \cdot \rho_l \sigma_l^2}{\lambda n_l}\right)$

$R_l = \frac{1}{2}\log_2\!\left(\frac{2\ln 2 \cdot \rho_l \sigma_l^2}{\lambda n_l}\right)$

[VERIFIED] Correct.

**Characterization as "reverse water-filling" (line 726):**

[ERROR] This is called **water-filling** (or water-pouring), not **reverse water-filling**. In the rate-distortion/source-coding context, the optimal bit allocation that assigns more bits to higher-variance sources is the standard water-filling solution. "Reverse water-filling" is the term used in the rate-distortion context for a *different* problem: allocating distortion across sources subject to a total rate constraint, where sources with variance below a threshold receive zero rate (are discarded entirely). The solution here -- allocating more rate to higher-variance/higher-importance components -- is the standard water-filling allocation from the channel coding / transform coding literature.

More precisely: the solution in the document is analogous to the optimal bit allocation in transform coding (e.g., Segall 1976, Shoham & Gersho 1988), which is indeed a water-filling solution. The "reverse water-filling" terminology applies to the Gaussian vector rate-distortion problem where some components may receive zero rate.

In this context where $R_l$ can in principle be zero for some layers (very low importance), the reverse water-filling label is defensible. But the formula as derived does not show the $\max(0, \cdot)$ truncation that characterizes reverse water-filling. The formula $R_l = \frac{1}{2}\log_2(\cdot)$ can be negative for small $\rho_l \sigma_l^2$, and the true reverse water-filling solution would set $R_l = \max(0, \frac{1}{2}\log_2(\cdot))$.

[ERROR] (minor, terminological): The solution as written does not include the $\max(0, \cdot)$ truncation needed for a proper reverse water-filling solution. If any $\rho_l \sigma_l^2$ is small enough that $R_l < 0$, the formula breaks down. The complete reverse water-filling solution is:

$$R_l = \max\!\left(0,\; \frac{1}{2}\log_2 \frac{2\ln 2 \cdot \rho_l \sigma_l^2}{\lambda n_l}\right)$$

**Jensen's inequality argument (lines 738-744):**

The document claims: "By the convexity of $2^{-2R}$ and Jensen's inequality applied in reverse, $D_{\text{mixed}} \leq D_{\text{uniform}}$."

The function $f(R) = 2^{-2R}$ is convex. By Jensen's inequality, for a convex function: $f(\mathbb{E}[R]) \leq \mathbb{E}[f(R)]$.

If we have uniform allocation $R_l = R$ for all $l$, then $D_{\text{uniform}} = \sum_l \rho_l \sigma_l^2 \cdot 2^{-2R}$. The optimal mixed allocation achieves $D_{\text{mixed}} \leq D_{\text{uniform}}$ by the Lagrangian optimality.

The Jensen's inequality argument as stated is slightly imprecise -- Jensen's inequality by itself shows that the average of a convex function is at least the function of the average, which is the *opposite* direction. The correct argument is that the Lagrangian optimality directly shows $D_{\text{mixed}} \leq D_{\text{uniform}}$ since uniform is a feasible solution and the optimal solution can only be better or equal.

[ERROR] (minor, proof-sketch imprecision): The invocation of "Jensen's inequality applied in reverse" is not rigorous. The correct argument is simply that uniform allocation is a feasible point of the Lagrangian optimization, and the optimal solution $D_{\text{mixed}}$ is by definition no worse. Jensen's inequality would need to be applied more carefully, or a direct optimality argument is cleaner.

---

## 6. Block-Wise Quantization Deep Dive

### Section 6.1 -- Effective Bits Formula (line 796)

$$\text{bits}_{\text{eff}} = N + \frac{8M}{B}$$

[VERIFIED] From total bits per weight: $N$ (data) + $8M/B$ (metadata, where $M$ bytes per block are spread across $B$ weights). Correct.

### Section 6.2 -- Effective Bits Tables (lines 810-824)

**Symmetric (24 bits overhead = 16-bit scale + 8-bit bit-width):**
- $N=2, B=32$: $2 + 24/32 = 2 + 0.75 = 2.75$ [VERIFIED]
- $N=2, B=64$: $2 + 24/64 = 2 + 0.375 = 2.375$ [VERIFIED]
- $N=2, B=128$: $2 + 24/128 = 2 + 0.1875 = 2.1875$ [VERIFIED]
- $N=3, B=32$: $3 + 24/32 = 3 + 0.75 = 3.75$ [VERIFIED]
- $N=3, B=64$: $3 + 24/64 = 3 + 0.375 = 3.375$ [VERIFIED]
- $N=3, B=128$: $3 + 24/128 = 3 + 0.1875 = 3.1875$ [VERIFIED]
- $N=4, B=32$: $4 + 24/32 = 4 + 0.75 = 4.75$ [VERIFIED]
- $N=4, B=64$: $4 + 24/64 = 4 + 0.375 = 4.375$ [VERIFIED]
- $N=4, B=128$: $4 + 24/128 = 4 + 0.1875 = 4.1875$ [VERIFIED]

**Asymmetric (40 bits overhead = 16-bit scale + 16-bit zero + 8-bit bit-width):**
- $N=2, B=32$: $2 + 40/32 = 2 + 1.25 = 3.25$ [VERIFIED]
- $N=2, B=64$: $2 + 40/64 = 2 + 0.625 = 2.625$ [VERIFIED]
- $N=2, B=128$: $2 + 40/128 = 2 + 0.3125 = 2.3125$ [VERIFIED]
- $N=3, B=32$: $3 + 40/32 = 3 + 1.25 = 4.25$ [VERIFIED]
- $N=3, B=64$: $3 + 40/64 = 3 + 0.625 = 3.625$ [VERIFIED]
- $N=3, B=128$: $3 + 40/128 = 3 + 0.3125 = 3.3125$ [VERIFIED]
- $N=4, B=32$: $4 + 40/32 = 4 + 1.25 = 5.25$ [VERIFIED]
- $N=4, B=64$: $4 + 40/64 = 4 + 0.625 = 4.625$ [VERIFIED]
- $N=4, B=128$: $4 + 40/128 = 4 + 0.3125 = 4.3125$ [VERIFIED]

**Overhead percentage claim (line 826):**
"At $N=2$ with $B=32$ and asymmetric, the effective bit width is 3.25 -- the metadata overhead is 62.5% of the weight data."

Overhead bits: 40. Weight data bits: $2 \times 32 = 64$. Overhead as fraction of weight data: $40/64 = 0.625 = 62.5\%$. [VERIFIED]

### Section 6.3 -- MXQ-2.5 Worked Example (lines 828-874)

**Target raw bits calculation (line 859):**
$$N_{\text{target}} = \text{bits}_{\text{eff, target}} - \frac{8M}{B} = 2.5 - \frac{24}{64} = 2.5 - 0.375 = 2.125$$

[VERIFIED] Correct arithmetic.

**Size calculation table (lines 847-855):**

Checking selected rows (Size = Params $\times$ eff_bits / 8):
- Embeddings: $0.6 \times 10^9 \times 5.375 / 8 = 0.403$ GB [VERIFIED] (document says 0.40, reasonable rounding)
- lm_head: $0.6 \times 10^9 \times 6.375 / 8 = 0.478$ GB [VERIFIED] (document says 0.48)
- MLP (middle): $45.0 \times 10^9 \times 2.475 / 8 = 13.92$ GB [VERIFIED]

### Section 6.5 -- Block Quantization Error Analysis (lines 906-941)

**Expected maximum of $B$ i.i.d. $\mathcal{N}(0, \sigma^2)$ samples (line 916):**
$$\mathbb{E}[\max_i |w_i|] \approx \sigma \sqrt{2 \ln(2B)}$$

[VERIFIED] For the maximum of $2B$ i.i.d. half-normal samples (folded from $B$ samples of $|w|$), the expected maximum of $n$ i.i.d. standard normal random variables is approximately $\sqrt{2 \ln n}$. Here, $\max_i |w_i|$ is the maximum of $B$ absolute values, which has the same distribution as the maximum of $2B$ half-normal samples... actually, more precisely:

$\max_{i=1}^B |W_i|$ for $W_i \sim \mathcal{N}(0, \sigma^2)$ can be analyzed as the max of $B$ chi-distributed variables. The standard result for the expected max of $n$ i.i.d. $\mathcal{N}(0,1)$ variables is approximately $\sqrt{2\ln n}$ for large $n$. For $|W_i|$, $P(\max|W_i| \leq t) = P(|W_1| \leq t)^B = (2\Phi(t/\sigma) - 1)^B$, and $\mathbb{E}[\max|W_i|] \approx \sigma\sqrt{2\ln(2B)}$ is a well-known approximation.

[VERIFIED] The approximation is standard.

**Numerical check for $B=64$ (line 918):**
$\sqrt{2\ln(128)} = \sqrt{2 \times 4.852} = \sqrt{9.704} = 3.115 \approx 3.11$. [VERIFIED] (document says $\sqrt{9.70} \approx 3.11$, accurate enough)

**Expected MSE (line 930):**
$$\text{MSE}_{\text{block}} \approx \frac{(3.11\sigma)^2}{12(2^{N-1} - 1)^2} = \frac{9.67\sigma^2}{12(2^{N-1} - 1)^2} = \frac{0.806\sigma^2}{(2^{N-1} - 1)^2}$$

Check: $3.11^2 = 9.6721$. $9.6721/12 = 0.80601$. [VERIFIED]

**MSE table (lines 932-939):**

| $N$ | $(2^{N-1}-1)^2$ | MSE/$\sigma^2$ |
|:-:|:-:|:-:|
| 2 | $(2^1 - 1)^2 = 1$ | $0.806$ |
| 3 | $(2^2 - 1)^2 = 9$ | $0.806/9 = 0.0896$ |
| 4 | $(2^3 - 1)^2 = 49$ | $0.806/49 = 0.01645$ |
| 5 | $(2^4 - 1)^2 = 225$ | $0.806/225 = 0.003582$ |
| 6 | $(2^5 - 1)^2 = 961$ | $0.806/961 = 0.000839$ |
| 8 | $(2^7 - 1)^2 = 16129$ | $0.806/16129 = 0.00004997$ |

All [VERIFIED]. The "Relative to 4-bit" column:
- 2-bit: $0.806/0.01645 = 48.97$

[ERROR] **Line 934:** The document says 2-bit is "36.6x worse" than 4-bit. But $0.806 / 0.01645 = 49.0$, not 36.6.

Let me recheck: $0.806 / 0.0164 = 49.1$. The document claims 36.6x. This appears to be wrong.

Actually, wait -- let me recompute with the exact fraction: $(2^3-1)^2/(2^1-1)^2 = 49/1 = 49$. So 2-bit MSE is exactly 49x the 4-bit MSE.

$36.6$ would correspond to roughly $49/1.34$, which does not match any obvious alternative calculation. The factor 36.6 might come from comparing with $(2^{N}-1)^2$ instead of $(2^{N-1}-1)^2$: $(2^4-1)^2/(2^2-1)^2 = 225/9 = 25$. That also does not give 36.6.

Another possibility: $36.6 \approx 4^{4-2} \times$ some correction. $4^2 = 16$, not 36.6 either.

[ERROR] The "Relative to 4-bit" ratios in the table are incorrect:

| $N$ | Document says | Correct ratio (MSE_N / MSE_4) |
|:-:|:-:|:-:|
| 2 | 36.6x worse | **49.0x worse** |
| 3 | 4.1x worse | $49/9 = 5.44$x worse |
| 5 | 4.6x better | $49/225 = 0.2178$, i.e., **4.59x better** |
| 6 | 19.6x better | $49/961 = 0.0510$, i.e., **19.6x better** |
| 8 | 328x better | $49/16129 = 0.00304$, i.e., **329x better** |

Corrections needed:
- 2-bit: should be **49.0x worse** (not 36.6x)
- 3-bit: should be **5.4x worse** (not 4.1x)
- 5-bit: 4.6x is approximately correct (4.59x) [VERIFIED]
- 6-bit: 19.6x is correct [VERIFIED]
- 8-bit: 328x is approximately correct (329x) [VERIFIED]

The claim on line 941 -- "giving a block 2 bits instead of 4 increases its MSE by 37x" -- should say **49x**, consistent with the corrected table.

### Section 6.6 -- Summary Formulas (lines 943-967)

**Bit allocation objective (line 960):**
$$\min_{\{N_k\}} \sum_k \rho_k \cdot \sigma_k^2 \cdot (2^{N_k - 1} - 1)^{-2} \quad \text{s.t.} \quad \frac{1}{n_b}\sum_k N_k = \bar{N}_{\text{target}}$$

[VERIFIED] Consistent with the MSE model $\text{MSE}_k \propto \sigma_k^2 / (2^{N_k-1}-1)^2$ and the importance-weighted objective.

---

## Summary of Errors

### Significant Errors

1. **Section 2.3, SQNR formula (line 227):** The SQNR formula is garbled -- it conflates multiple standard forms and is not self-consistent. The constant $4.35$ dB is not standard. The formula as written double-counts the bit-width dependence. Recommend replacing with the clean form $\text{SQNR} = 10\log_{10}(12\sigma^2/\Delta^2)$ for the general case, or $6.02N + 1.76$ dB for the sinusoidal full-range case.

2. **Section 2.3, approximate SQNR values (lines 230-232):** The claimed SQNR values (~13.6, ~25.8, ~50.2 dB) are inconsistent with the document's own MSE values in Section 2.4. Using those MSE values gives ~9.3, ~20.2, ~46.5 dB respectively.

3. **Section 3.5, MSE clipping+granularity formula (line 474):** Two issues:
   - The clipping error term has a dimensional inconsistency ($2\sigma^2$ prefactor with $\phi(w/\sigma)\,dw$ integrand). Should be $\frac{2}{\sigma}\int_c^\infty (w-c)^2 \phi(w/\sigma)\,dw$ or equivalently $2\sigma^2 \int_{c/\sigma}^\infty (u - c/\sigma)^2 \phi(u)\,du$.
   - The granularity error coefficient should be $\frac{c^2}{12(2^{N-1}-1)^2}$, not $\frac{c^2}{3(2^{N-1}-1)^2}$ (factor of 4 too large).

4. **Section 6.5, MSE ratio table (lines 932-939):** The "Relative to 4-bit" column has two incorrect entries:
   - 2-bit: stated as 36.6x worse, should be **49.0x worse**
   - 3-bit: stated as 4.1x worse, should be **5.4x worse**
   - Line 941 repeats the 2-bit error ("37x" should be "49x")

### Minor Errors / Imprecisions

5. **Section 5.5, "reverse water-filling" terminology (line 726):** Defensible but imprecise. The formula as written lacks the $\max(0, \cdot)$ truncation that defines reverse water-filling. Without it, $R_l$ can be negative for low-importance layers.

6. **Section 5.5, Jensen's inequality argument (lines 738-744):** The invocation of "Jensen's inequality applied in reverse" is not rigorous. A direct Lagrangian optimality argument is cleaner and correct.

### Verified (No Errors Found)

- All quantization/dequantization formulas (Sections 1.1--1.3): Correct
- Quantization error model (Section 1.6): Correct
- Bit-width tables and step sizes (Section 2.1--2.2): Correct
- MSE scaling with bit width (Section 2.4): Correct
- Optimal clipping values for Gaussian (Section 2.4): Correct
- Bit packing formulas and tables (Section 2.5): Correct
- Scale and zero-point formulas (Sections 3.1--3.3): Correct
- Super-block (GGUF Q2_K) analysis (Section 3.4): Correct
- GPTQ update rule (Section 4.2): Correct
- AdaRound formulation (Section 4.3): Correct
- Error propagation bound (Section 4.4): Correct
- Rate-distortion function for Gaussian (Section 5.1): Correct
- Shannon differential entropy (Section 5.2): Correct
- Lloyd-Max quantizer (Section 5.4): Correct
- Lagrangian derivation for mixed-precision (Section 5.5): Correct (except terminological issues noted)
- Block overhead formulas (Section 6.1--6.2): All 18 table entries verified correct
- MXQ-2.5 worked example (Section 6.3): Correct
- Expected maximum formula (Section 6.5): Correct
- Block MSE formula derivation (Section 6.5): Correct

---

*Audit complete. 4 significant errors and 2 minor imprecisions identified out of ~80 formulas checked.*
