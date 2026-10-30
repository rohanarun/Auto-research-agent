# Seven nested affine levels: an exact conditional certificate

We certify

\[
 \theta\leq\frac{874809963}{10^9}=0.874809963,
 \qquad \ell=\frac{125570111}{750000000},
 \qquad h=\frac{398610570291347}{500000000000000}.
\]

Here \(\theta=11/12-\ell/4\). The improvement over the frozen four-level claim is exactly \(319/10^9\). This is an arithmetic certificate in the **H1–H5 conditional framework of `depth3/PROOF.md`, instantiated at depth seven**. It is not a complete proof of Dirichlet nonvanishing, a Lean endpoint, or a verification of H1–H5. It also retains the underlying incumbent/preprint inputs of that framework. In particular, no conclusion here removes any source, return, counting, LOW-dispatch, or loss-absorption hypothesis.

## 1. Exact data and reproduction

The junctions, in order, are

| index | rational junction |
|---|---|
| 0 | 31901/50000 |
| 1 | 43592737/62500000 |
| 2 | 176457017/250000000 |
| 3 | 357161223/500000000 |
| 4 | 361483757/500000000 |
| 5 | 7317659/10000000 |
| 6 | 185180067/250000000 |

The window is the entire closed interval
\[
 b\in[f,s],\qquad f=17/500+10^{-8},\qquad s=2016882571/50000000000.
\]
All original model constants, including \(\alpha=10^{-8}\), region margin \(10^{-8}\), cell mesh \(10^{-8}\), tail slack, marked losses, and source-frequency losses, are unchanged. The LOW floor remains \(10^{-10}\); the direct-deletion room floor remains \(15/10000\).

From the workspace root run

```
python solution/verify.py
```

This calls the **unchanged** `depth3/certify_depth3.py` at the new point (92/92 checks), and then generates and replays an additional exact rational interval certificate. The latter contains 36 rational-function certificates, 47 closed subintervals, and 154 other exact facts or enclosures. `solution/certificate.json` contains all polynomial coefficients, domain endpoints, subinterval endpoints, and rational bounds. No optimizer, floating-point sign decision, or downloaded resource is needed for verification. Floating-point numbers printed in logs are explanatory only.

The search scripts and their unsuccessful intermediate points are retained as an audit trail; none is proof evidence. The accepted rational data are in `solution/parameters.json`.

## 2. Conditional obligations retained

The following are the existing hypotheses, specialized to seven levels, rather than assertions proved by this computation.

* **H1:** Every nonterminal step has the full O/R/L/D dispatch into the next region. Amplifiers and deletion targets use the first-level cell uniformly. Its nested family has the necessary position, width, and cell indices. The old-region moment is used only on states satisfying the old-region or reflection tests.
* **H2:** Joint sources and returned counts extend to seven labels. The prefix-fibre bounds, missing-radical coordinates, orientation indices, and their caps must all be supplied. Zero density is applied once to final rows, not once per level. Every `Pays` instance keeps its own junction and its complete source and return inequalities.
* **H3:** Children inside an intermediate region but outside the old region, and children with too small a gap, have the required amplifier-floor dispatch. They cannot simply skip a stage. The nominal affine recurrence below is not a proof of this exceptional dispatch.
* **H4:** The hard-coded caps and budgets are re-bracketed at this depth, including amplifier ratios, slopes, slot caps, rebased masks, supply, position rooms, and zero-saving constants. The existing width-floor feasibility tests are mandatory and are proved below without relaxation. Other structural H4 obligations remain conditional; this note does not assert that every original hard-coded cap already works.
* **H5:** All source, return, mesh, pad, grid, price-excess, and loss allowances, with their multiplicities and rebase factors, fit into the net reserves. The exact arithmetic below supplies a net HIGH reserve exceeding \(10^{-7}\) at every depth; H5 must realize allowances within that budget as well as the applicable LOW and dispatch budgets.

In particular the source/return constraints described in `depth3/reports/C.md` are retained at each link: source row-norm bound; scale inequality; volume after deletion; fixed target at most actual deletion; amplifier and length caps; nonnegative source savings; the `Pays` inequality on the full HIGH bin; prefix-fibre and missing-radical bounds; and payment of deletion and mesh at that link's junction. They are not replaced by the numerical value of a reserve. H1–H2 provide these structural statements, while the arithmetic below proves the reserve they consume. All H4 feasibility caveats in the baseline remain applicable.

## 3. Regions, plane, and affine recurrence

Write \(\kappa=4499/6000\), \(\rho(x)=1-10x/[3(5-2x)]\), and \(c(x)=5-6\rho(x)\). A region \((k,I)\) consists of
\[
 A+(6k-1)z\leq IM.
\]
Set \(T_6=(\kappa,1)\). For \(j=5,\ldots,0\), obtain \(T_j\) by `region_over(q[j+1], T[j+1])`, explicitly
\[
 R=\frac{6(k-qI)}{q c(q)},\quad
 k'=\frac{1+(6k-1)/(1+Rq)}6,\quad
 I'=\frac{I+R(1-\rho(q))-10^{-8}}{1+Rq}.
\]
Every denominator is nonzero and all region prices, gaps \(k_j-q_jI_j\), and costs are positive. The prices are at most \(\kappa\), and all junctions lie in the original allowed ranges, strictly increasingly below \(3/4\). The frozen checker proves these assertions with fractions.

Let \((\lambda,B,r)\) be `Ledger(T[0]).plane(q[0],s)`, with all channels and constants exactly as in `depth3/model.py`. Numerically, only for orientation,
\[
 \lambda\simeq3.730473638733,\qquad B\simeq0.981458487445.
\]
This recursive definition, together with the rational inputs, specifies exact rational stage parameters; expanded fractions are unnecessary.

For the first step put
\[
 g_0=B-I_0+(6k_0-\lambda)b,\quad
 K_j=\frac{q_j}{6(k_j-q_jI_j)},\quad
 a_0=K_0g_0+\alpha,\quad
 d_0=\frac{g_0+6I_0a_0}{6k_0}.
\]
Start the child at
\[
 (A,z,M)=(B-(\lambda-1)b-d_0,\ b-d_0,\ 1-6a_0).
\]
At each subsequent step set
\[
 g_j=A+(6k_j-1)z-I_jM,\quad a_j=K_jg_j+\alpha.
\]
For a nonterminal step set \(d_j=(g_j+6I_ja_j)/(6k_j)\) and replace \((A,z,M)\) by \((A-d_j,z-d_j,M-6a_j)\). Thus every quantity is affine in \(b\), by induction. This is the cell-uniform recurrence of the baseline, not a per-position independently optimized recurrence. Symbolic second derivatives in the verifier confirm affinity. All nominal gaps and amplifiers and all remaining slots are positive on the closed window, by affine endpoint checks.

At every stage, including the terminal stage's formal deletion,
\[
 a_j-q_jd_j=\alpha\left(1-\frac{q_jI_j}{k_j}\right)=:L_j.
\]
Indeed substitute \(a_j=K_jg_j+\alpha\) and the displayed formula for \(d_j\); the coefficient of \(g_j\) cancels. Deletion places the child exactly on the target-region boundary because \(g_j-6k_jd_j+6I_ja_j=0\). These identities are checked symbolically. Each \(L_j\geq10^{-10}\); the smallest is approximately \(1.1822\cdot10^{-10}\), strictly above the unchanged floor. Also every \(K_j<15\).

## 4. Continuous first-step ledger and sources

Use the exact rational splitting point
\[
 t=661817482376/10^{12}.
\]
On \([q_0,t]\), at both cell endpoints, the two common-rate branches, linear, inverse, and capped channels strictly exceed the original fee \(L_0+\operatorname{markedLossAt}(s)\). The mixed channel equals the fee at \(q_0\), and its difference divided by \(d-q_0\) is strictly positive on the **closed** interval \([q_0,t]\). Therefore equality at the junction is handled exactly and positivity holds everywhere above it.

On \([t,3/4]\) the HIGH reserve alone strictly exceeds that fee at both cell endpoints. The split is certified without trusting an approximate root: the top-cell HIGH difference is negative at \(t-10^{-12}\), positive at \(t\), and has strictly positive derivative throughout \([q_0,3/4]\). In particular the intervals meet and cover the whole required bin domain, including the shared boundary.

The cutoff dominance difference vanishes exactly at \(q_0\); its quotient by \(d-q_0\) is strictly positive on \([q_0,t]\). The half gain and both caps are positive, the cap coefficient is below \(1/20\), and both common-rate branches exceed twice that coefficient throughout this interval. These statements are proved by rational interval certificates, not sampled bins.

Each channel, source frequency, and HIGH difference is affine in \(b\) at fixed \(d\). Their endpoint certificates therefore prove the assertions for every \(b\in[f,s]\). Both branches are checked separately, which also checks their minimum. Source supply is positive and twice the supplied width is at most live frequency at both endpoints, hence throughout the window. The original supply and frequency losses are retained.

## 5. Top-row and rational geometry

The unchanged outward-rounded interval computation gives
\[
 \theta_{\min}\simeq0.874809959085207<\theta,
\]
with enclosure width less than \(2\cdot10^{-57}\). It verifies the discriminant, nonnegative critical point, positive cutoff denominator on \([0,3/4]\), nonnegative square coefficient, and the algebraic optimizer's balanced window top below \(s\). Exact interval endpoints are recorded in the certificate.

We additionally certify the supplied **rational** \(h\), rather than silently identifying it with the algebraic optimizer. Let \((a_2,a_1,a_0)\) be `root_coeffs(lambda,B)` and
\[
 D(d)=\tfrac{25}2\lambda-10+(6-9\lambda)d.
\]
The cleared top-row margin is
\[
 P(d)=8D(d)\big[(9\lambda+6\lambda d+12d)\theta
                  -7\lambda(1+d)-11d\big]
       +h(a_2d^2+a_1d+a_0).
\]
This is the quadratic cleared margin underlying the baseline TopRow formulas: at its algebraic optimum its constant and linear terms about \(\delta_*\) vanish, and its quadratic coefficient is \(24\lambda\kappa_2\). For our rational data, direct Bernstein enclosure proves \(P(d)>0\) on all of \([0,3/4]\). Its 12 closed leaves have least certified lower bound greater than \(2.4\cdot10^{-5}\). Also \(D(d)>0\) there, so clearing its denominator preserves the inequality. The other factor \(9\lambda+6\lambda d+12d\) is positive since \(\lambda>0,d\geq0\).

For the capped window condition use `(ca,cb,cc) = capped_quad(lambda,B,theta,h,s)`. Exact rational inequalities give
\[
 ca>0,\quad cb^2-4ca\,cc<0,\quad
 cc-\frac{cb^2}{4ca}>0.
\]
Thus this capped quadratic is strictly positive for every real argument, which covers its whole required continuous domain. Its global minimum is approximately \(2.4111\cdot10^{-8}\). The outward-rounded balanced-top computation at the rational pair \((\theta,h)\) gives approximately \(0.0403371827325585<s\), independently confirming the correct window branch.

The rational geometry also obeys \(0<h<1\), \(\ell>0\), positive prime-supply margin \(\ell/h-7/37\), positive extension room \(5\ell-h\), positive geometric width \(2h-1-3\ell\), and the original low-side inequalities \(1-h>0\), \(h-3\ell-(11/6)(2h-1-3\ell)>0\), \(1-3\ell>0\), and \(1-h/4>0\). These are checked exactly. This is the nested-affine top-row model; no claim is made that the different frozen Track-A adaptive endpoint establishes this stronger bound.

## 6. Joint returns on the entire continuous domain

Define the mesh-inclusive length
\[
 T_A(b)=B-(\lambda-1)(b-10^{-8}).
\]
For each depth \(m=2,\ldots,7\), the joint HIGH reserve is
\[
 H_m(d,b)=1-\rho(d)-dT_A(b)
       +\sum_{i<m-1}(d-q_i)d_i(b)
       -c(d)\sum_{i<m}a_i(b),
 \qquad(d,b)\in[q_{m-1},3/4]\times[f,s].
\]
It satisfies the exact return identity
\[
 H_m=(1-\rho(d))\Big(1-6\sum_{i<m-1}a_i\Big)
       -d\Big(T_A-\sum_{i<m-1}d_i\Big)-a_{m-1}c(d)
       +\sum_{i<m-1}(a_i-q_id_i).
\]
To verify it, expand and use \(c(d)=6(1-\rho(d))-1\). The last sum is precisely \(\sum_{i<m-1}L_i\), so it is subtracted before assigning a net return budget. The identity is checked symbolically at every depth, for symbolic \(d,b\).

This is the algebra used with the full H1–H2 source/return constraints. The intermediate scales cancel through
\(-M(1-\rho)-\rho M+M=0\); volume and target bounds produce the terms \((d-q_i)d_i\), whose coefficients are nonnegative because the junctions increase. The source and return losses are still charged under H5. No independent density saving is added for an intermediate source.

For completeness, with \(q=q_{m-1}\), \(A_\Sigma=\sum_{i<m}a_i\), and \(D_\Sigma=\sum_{i<m-1}d_i\),
\[
 H_m(d,b)-H_m(q,b)=(d-q)\left[
 \frac{50(1-6A_\Sigma)}{3(5-2d)(5-2q)}-(T_A-D_\Sigma)\right].
\]
Since \(A_\Sigma\leq1/6\) and \(q\leq d\leq3/4\), the bracket is at least its value at \(d=q\). That value is nonnegative at both cell endpoints and is affine in the cell. Hence \(H_m\) is bounded below by its junction value over the whole rectangle. Symbolic identities, endpoint growth, and amplifier bounds are all checked.

In addition, direct Bernstein certificates prove
\[
 H_m(d,b)-\sum_{i<m-1}L_i-10^{-7}>0
\]
throughout each rectangle: certify the rational function of \(d\) at both cell endpoints and interpolate affinely in \(b\). The least net junction reserve across depths is greater than \(3.26\cdot10^{-7}\). Thus the allocated \(10^{-7}\) loss budget is strictly inside every net reserve, not taken from the LOW reserves. This arithmetic budget does not itself prove H5's structural accounting.

## 7. Unweakened width-floor test and dispatch boundaries

The old-region overshoot of the first child is
\[
 O(b)=\operatorname{OLD.overshoot}
 (B-(\lambda-1)b-d_0,\ b-d_0,\ 1-6a_0).
\]
It is affine in \(b\). Use exactly the baseline rule
\[
 W=\max\{2/5,\ \max(O(f),O(s))/(7/1000)+1/1000\}.
\]
Here \(W\simeq0.4662059482298543\). This defines an exact rational width floor. With \(r_o=q_0/(6\kappa)\), the original direct-deletion room is
\[
 R_D(b)=r_o(1-6a_0(b)-2W)
 -\{r_o[B-1+(6\kappa-\lambda)b+6a_0(b)]-a_0(b)\}.
\]
It is affine and its exact endpoint minimum is approximately
\(0.001500002248436778\), strictly above \(0.0015\). No smaller floor or relaxed room requirement was used.

The verifier also proves, with exact rationals:

* \((7/1000)W-\max O=7\cdot10^{-6}\), retaining positive reflection pad room;
* \((7/1000)W+1/1000\leq W/100\), retaining the comparison slack \(\xi_R=1/1000\);
* \(2W<1-6a_0(b)\) throughout the window;
* \(6\min_b a_0(b)\geq1/100\), retaining the original width slack;
* \(\min_bR_D(b)>L_0+r_o(1/100+9\cdot10^{-7})\), retaining the original allowance requirement.

These facts cover the whole cell interval because the relevant expressions are affine. The deeper physical rebases, position rooms and their supply/return constants remain part of H1–H5, just as in the baseline. In particular, this first-child feasibility test is not misrepresented as a verification of every deeper physical dispatch.

## 8. Why the interval certificate proves continuum coverage

For a rational function \(p(x)/q(x)\), the certificate stores exact rational power coefficients. On each closed interval \([u,v]\), substitute \(x=u+(v-u)t\). If the resulting power coefficients are \(c_k\) and the degree is \(n\), its Bernstein coefficients are
\[
 \beta_i=\sum_{k=0}^i\frac{\binom{i}{k}}{\binom{n}{k}}c_k.
\]
For \(0\leq t\leq1\), the Bernstein basis is nonnegative and sums to one, so the polynomial lies between the smallest and largest coefficients, **including both endpoints**. The denominator's Bernstein enclosure must have constant strict sign; a negative denominator is handled by changing both signs. The four ratios of numerator and positive-denominator interval endpoints enclose the rational function. A strictly positive lower bound proves the required inequality on that entire interval.

If an enclosure is inconclusive, the interval is bisected rationally. The replay checks exact adjacency of leaves and equality to both domain endpoints, with no gaps or endpoint omissions. It recomputes every coefficient enclosure and every rational bound. In addition to this replay, `verify.py` reconstructs the expressions from the unchanged model, so arbitrary unrelated positive polynomials cannot substitute for the ledger conditions.

Cell interpolation is justified by symbolic affinity and by the recurrence proof above. HIGH growth is also proved algebraically, not inferred from a finite grid. Equality boundaries are treated by exact factor removal only after confirming the numerator vanishes at that boundary.

Square-root enclosures use `depth3/interval.py`: integer square root of an exact scaled rational gives the lower grid endpoint and the upper endpoint is either equal or one grid unit larger. Every subsequent operation rounds outward on a \(10^{-60}\) rational grid; interval division rejects a denominator containing zero. The root discriminant enclosure is additionally checked by squaring its endpoints. The marked-bin width is certified to be the least \(10^{-8}\)-grid point above its defining square root. The bin split has the independent rational bracket in Section 4. All supplied parameters are exact rationals from the outset; there is no unaccounted decimal rounding.

## 9. Scope of the result

Sections 3–8 establish the arithmetic required by the frozen nested-affine certificate, strengthened by explicit rational geometry and continuum interval coverage. Combining it with the retained H1–H5 implications gives the stated conditional class bound. The unresolved recursive dispatch, exceptional children, joint source/return infrastructure, hard-coded H4 re-bracketing and structural H5 loss accounting are not claimed as completed mathematics or Lean code.

This work builds on the PR #67 lineage and the nested-level port credited in `depth3/PROOF.md`. The new search, certificate and exposition were AI-assisted. Baseline source files were preserved; all new deliverables are in `solution/`.
