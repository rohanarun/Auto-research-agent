# Seven nested affine levels: conditional exact certificate

## Claim and status

Let

\[
 \theta={874809963\over10^9},\qquad
 \ell={125570111\over750000000},\qquad
 h={797221147\over10^9}.
\]

Then `theta = 11/12 - ell/4` exactly. Subject to the same structural hypotheses H1–H5 stated in `depth3/PROOF.md`, iterated to seven levels, the nested-affine construction gives

\[
                         \theta\le 0.874809963.
\]

This improves the frozen four-level certificate `874810282/10^9` by `319/10^9`. It is conditional arithmetic evidence for the tracker architecture, not a complete Lean nonvanishing endpoint and not a claim that H1–H5 have been formalized.

The rational stage parameters are

| parameter | exact value |
|---|---:|
| q0 | 319009933279/500000000000 |
| q1 | 697484037663/1000000000000 |
| q2 | 705807329603/1000000000000 |
| q3 | 714288079159/1000000000000 |
| q4 | 180730963757/250000000000 |
| q5 | 731724083863/1000000000000 |
| q6 | 148138940611/200000000000 |
| window top s | 5042206543/125000000000 |
| first-ledger partition p | 661817367088/1000000000000 |

All entries are interpreted as exact rationals. Decimal values printed by the checker are diagnostics only.

## The inherited H1–H5 framework

The certificate deliberately retains, and does not weaken or silently discharge, the hypotheses in `depth3/PROOF.md`.

**H1 (recursive LOW dispatch).** A step deleting into a non-old region has the recursive O/R/L3/D dispatch, with its deletion and amplifier fixed by the parent cell. At depth seven this is instantiated recursively through the six deeper regions.

**H2 (joint labels and returns).** The pair-label count and returned-sum argument extend to seven labels, with the required missing-radical coordinates and composite-prefix fibre bounds. The zero-density estimate is applied once to the final rows.

**H3 (children between consecutive regions).** The error branch for children in the next region but outside the old region uses the amplifier floor described in the frozen proof, paid from the available deeper slack. It is not replaced by the unsafe direct-routing shortcut.

**H4 (re-bracketed constants).** Hard-coded caps are re-bracketed for the certified seven-level values. The exact checker reports all relevant region slopes, rates, slot caps and amplifier ratios. In particular the largest reported amplifier ratio is about 14.038 and the largest region slope `6k` is about 4.440. These are disclosed requirements, not assertions that the old caps still apply.

The width-floor condition itself is unchanged:

\[
 W=\max\left({2\over5},\ {\max_b\operatorname{overshoot}_{old}(b)\over7/1000}+{1\over1000}\right),
 \qquad \operatorname{room}_D(b)\ge {15\over10000}.
\]

The checker proves the overshoot and room are affine in the continuous cell variable, evaluates both closed endpoints, and obtains `W ≈ 0.4662054` and minimum room about `0.0015001`. Thus neither the width floor nor its `1.5e-3` room threshold is weakened.

**H5 (loss absorption).** The depth-seven analogue of the returned-payment budget absorbs every mesh, pad, price, omega and repeated allowance loss from the positive joint reserves. As in the frozen proof, this is a structural/Lean obligation. The arithmetic certificate supplies positive joint reserves of at least about `3.284e-7`, far larger than the epsilon-scale allowances, but does not claim that the full Lean bookkeeping has been completed.

## Exact model

Write `rho(d)=1-10d/(3(5-2d))` and `cost(d)=5-6rho(d)`. Starting with the old region `(kappa,1)`, define recursively

\[
 R(q;k,I)={6(k-qI)\over q\,cost(q)},\quad
 c={6k-1\over1+Rq},\quad
 I'={I+R(1-\rho(q))-10^{-8}\over1+Rq},\quad k'={1+c\over6}.
\]

The six applications at `q6,...,q1` produce regions `T5,...,T0`; step `j` deletes into `Tj`, while the last step deletes into the old region. `solution/model.py` is the frozen rational port of these definitions. It uses `Fraction` throughout.

For a cell coordinate `b`, the cascade constructs amplifiers `a_j(b)` and deletions `d_j(b)` by the cell-fixed formulas in the frozen proof. Every such quantity is affine in `b`. The LOW reserve at step `j` is

\[
 10^{-8}\left(1-{q_j I_j\over k_j}\right).
\]

The seven exact values are all at least `10^-10`; the least printed value is `1.179e-10`.

For depth `m=2,...,7`, the returned HIGH reserve is

\[
 H_m(\delta,b)=1-\rho(\delta)-\delta\,TA(b)
 +\sum_{i=0}^{m-2}(\delta-q_i)d_i(b)
 -\left(\sum_{i=0}^{m-1}a_i(b)\right)cost(\delta).
\]

The frozen algebraic identity gives

\[
 H_m(\delta,b)-H_m(q_{m-1},b)
 = (\delta-q_{m-1})G_m(\delta,b),
\]

where the certified lower growth factor is obtained from

\[
 {50\over3}{1-6\sum_{i<m}a_i(b)\over(5-2q_{m-1})^2}
 -\left(TA(b)-\sum_{i<m-1}d_i(b)\right).
\]

For each depth the checker proves exactly that the endpoint reserve, growth, and amplifier sum are affine in `b` (zero second difference), then checks both endpoints of the closed cell interval. All growth bounds are positive and all amplifier sums are at most `1/6`. Therefore the HIGH-return inequalities hold for every `b` and every `delta in [q_(m-1),3/4]`, including boundaries.

## First-step source ledger on the whole bin domain

The continuous bin domain `[q0,3/4]` is split at the fixed rational `p`; no floating root is used to define it.

On `[q0,p]`, each denominator and numerator is a rational polynomial. `solution/certify.py` uses SymPy's exact rational polynomial root count (Sturm isolation), checks that denominators have no roots, and evaluates an exact rational sign. This proves the rate (both common-rate branches), linear, inverse, capped, mixed, dominance, gain and cap inequalities. The mixed channel equals the fee exactly at `q0`; after dividing out the exact endpoint factor, Sturm proves strict positivity on `(q0,p]`.

On `[p,3/4]`, the same procedure proves that the unmarked HIGH reserve alone exceeds the fee. Both closed pieces contain `p`, so there is no uncovered rounding gap. Dependence on the cell coordinate is affine; checking the first and top cells covers the entire continuous cell interval.

This supplies all frozen source constraints as well as the deeper return constraints above.

## Top row and rational endpoint witness

Let `lambda,B` be the exact first-ledger plane. Put

\[
 D(\delta)={25\over2}\lambda-10+(6-9\lambda)\delta,
\]

and let `R(delta)=a2 delta^2+a1 delta+a0` use the exact coefficients in `model.root_coeffs`. For the submitted rational `(theta,h)`, the cleared top-row margin is

\[
 8D(\delta)\left[\lambda(9\theta-7)+
 \delta((6\lambda+12)\theta-7\lambda-11)\right]+hR(\delta).
\]

The checker proves this rational function is strictly positive on the complete closed interval `[0,3/4]` by exact Sturm root counting. It also verifies `D>0` at both endpoints. Independently, outward interval evaluation encloses the algebraic minimum class bound by

`[0.87480995912187..., 0.87480995912187...]`,

leaving more than `3.878e-9` below the submitted theta.

For the window, outward interval arithmetic applied directly to the submitted rational `(theta,h)` gives balanced top at most `0.040337187154722`, leaving more than `4.651e-7` below `s`.

## Boundary and rounding controls

The only irrational operations are square roots in the diagnostic algebraic optimum and in the balanced-window formula. `solution/interval.py` rounds every operation outward to a `10^-60` rational grid. Square-root lower and upper bounds are obtained with integer square roots and checked rational squares. Division rejects intervals containing zero.

All stage parameters, the bin partition, LOW/HIGH reserves, width floor, room, polynomial coefficients and endpoint comparisons are exact rationals. Sturm checks cover closed domains; equality at the mixed-channel source boundary is handled separately and the remaining factor is proved positive. Affine endpoint reductions are justified by exact zero second differences. No optimizer output, binary float, or formatted decimal participates in a pass/fail decision.

## Reproduction and scope

Run from the repository root:

```sh
python solution/certify.py
```

It reports `110/110 checks OK`. The saved transcript is `solution/certificate_output.txt`.

This is a complete exact arithmetic certificate for the stated conditional seven-level construction. It is not a proof of H1–H5, not a complete Lean nonvanishing endpoint, and not an unconditional improvement to the underlying quasi-RH theorem.
