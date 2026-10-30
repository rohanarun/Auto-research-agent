# Seven constrained nested affine levels

This is an exact arithmetic certificate for

\[
\boxed{\theta\leq 874809963/10^9=0.874809963.}
\]

It improves the frozen four-level bound by exactly \(319/10^9\). It is conditional on **all of H1–H5 in `depth3/PROOF.md`, instantiated at depth seven**, with the same meaning and outstanding structural obligations as in that document. It also inherits the underlying analytic results used there. It is not a complete Lean proof, a proof of those hypotheses, or a new unconditional nonvanishing theorem. In particular, the recursive LOW dispatch, position coverage, joint source-label count, return construction, and feasibility of the H4 re-bracketing remain hypotheses. None is inferred just from numerical optimization or the checks below.

All baseline source files are preserved. The proof uses their rational functions, not changed versions of the model or frozen verifier. Run from the workspace root:

```
PYTHONDONTWRITEBYTECODE=1 python solution/verify.py
```

This executes every frozen arithmetic test at the new parameters, including the original width-floor rule and its direct-room threshold. It then independently certifies the continuous inequalities with rational Bernstein bounds, symbolic identities, and outward rational intervals. The generated `solution/certificate.json` contains exact rational evidence, rather than just the decimal summaries in this note. `solution/verification.log` records a successful execution. The search scripts are discovery evidence only and are not needed to verify the result.

## Rational parameters

The seven junctions, indexed starting at zero, are

| stage | junction |
|---|---|
| 0 | \(31901/50000\) |
| 1 | \(697488157/10^9\) |
| 2 | \(352913803/500000000\) |
| 3 | \(357161057/500000000\) |
| 4 | \(144593429/200000000\) |
| 5 | \(182941371/250000000\) |
| 6 | \(740719869/10^9\) |

The window top is

\[
s=403376153/10^{10}=0.0403376153.
\]

For the claimed rational boundary take

\[
\ell=125570111/750000000=11/3-4\theta,
\qquad h=797221155889059/10^{15}.
\]

Thus \(11/12-\ell/4=\theta\) exactly. These are the nested-ledger geometry parameters, not an assertion that the unrelated root-level `certify.py` Track A budget attains this boundary. The check here is the nested-level model of `depth3/PROOF.md`.

The constants \(\alpha\), region margin, tail slack, and original cell mesh remain \(10^{-8}\). The first cell top remains \(b_-=3400001/10^8\). Every LOW-reserve floor remains \(10^{-10}\). No width-floor condition is relaxed.

The regions and all stage functions are rationally specified by the following recurrences; this gives exact stage parameters even when their expanded numerators and denominators are large. Write

\[
\kappa=4499/6000,\quad \rho(x)=1-\frac{10x}{3(5-2x)},\quad c(x)=5-6\rho(x),\quad m=\alpha=10^{-8}.
\]

Set \(T_6=(\kappa,1)\). If \(T_{j+1}=(k,I)\), define

\[
R=\frac{6(k-q_{j+1}I)}{q_{j+1}c(q_{j+1})},\qquad
T_j=\left(\frac{1+(6k-1)/(1+Rq_{j+1})}{6},
\frac{I+R(1-\rho(q_{j+1}))-m}{1+Rq_{j+1}}\right).
\]

This is precisely the unchanged `region_over` recurrence. Step \(j\) deletes into \(T_j\), and the last step deletes into the old region. Use the unchanged `Ledger(T_0).plane(q_0,s)` to define \((\lambda,B,r)\). Approximately,

\[
\lambda=3.730471111894451,\qquad B=0.9814582778152914.
\]

The exact fractions are in `certificate.json`; all subsequent computations use those fractions.

## Meaning of the continuous certificate

The arithmetic domain is the entire cell interval \(b\in[b_-,s]\), the first-step bin interval \(\delta\in[q_0,3/4]\), the top-row interval \([0,3/4]\), and, for each depth \(d=2,\ldots,7\), the joint HIGH interval \(\delta\in[q_{d-1},3/4]\). The domains are closed. At the mixed-channel junction and root-dominance junction an exact equality is allowed and proved; strict positivity is proved immediately above it. Continuous cell bounds are not obtained by sampling a mesh.

The other coordinates of actual sources, children, grid positions, and returned classes are subject to the original H1–H5. The certificate checks their algebraic use through cap and deletion slack identities below. It does not replace H1–H5 with an assumption that a numerical sample of positions suffices. Realization of the cell-uniform caps and families at every position, including the H3 gap cases, remains precisely the original structural condition.

For each rational function whose positivity is required, the verifier cancels the expression over the rationals, proves its denominator positive, and proves its numerator positive. To prove a polynomial positive on a rational interval \([a,b]\), substitute \(x=a+(b-a)t\) and express it in the Bernstein basis of its actual degree \(n\):

\[
p(a+(b-a)t)=\sum_{i=0}^n \beta_i {n\choose i}t^i(1-t)^{n-i}.
\]

The basis functions are nonnegative on \([0,1]\) and sum to one. Consequently \(p\geq\min_i\beta_i\). Every coefficient and every comparison is rational and exact. If the lower bound is not positive, the verifier bisects the interval into two closed rational intervals and repeats. The resulting 57 leaves cover their parent intervals by construction, with exact endpoints and strictly positive rational lower bounds. The leaf list is exported in `certificate.json`. There is no assumption of positivity between sample points.

At a permitted left equality the verifier first proves the numerator vanishes exactly at \(a\), divides it exactly by \(x-a\), checks zero remainder, and proves the quotient positive on the entire closed interval. The resulting rational function is zero at \(a\) and positive on \((a,b]\). Denominators are certified separately, so neither cancellation nor a boundary root hides a pole.

## First-step channels

Let \(F\) be the frozen `high_fee(q_0,s)`. Instead of relying on approximate rounding of the unmarked-channel root, the independent certificate chooses the explicit rational split

\[
t=33091/50000=0.66182.
\]

At both \(b=b_-\) and \(b=s\), rational Bernstein certificates prove:

* both branches of the common-rate channel exceed \(F\) on \([q_0,t]\);
* the linear, inverse, and capped channels exceed \(F\) on \([q_0,t]\);
* the mixed channel equals \(F\) at \(q_0\), and exceeds it on \((q_0,t]\);
* the unmarked HIGH channel exceeds \(F\) on \([t,3/4]\).

Both common-rate branches are checked, so taking their minimum retains the result even at the branch switch. The gain, inverse capacity, plain capacity, common-rate bounds, capacity upper bound, and dominance of the root cutoff are also certified on the full interval. Supplied width is positive and twice that width is at most the live frequency at both cell ends.

The verifier symbolically checks that every channel branch and the frequency are affine in \(b\). Therefore every bound holds for every \(b\) between the endpoints, for each fixed bin. Taking a minimum of the two already certified rate branches causes no loss. Together the two closed bin intervals cover \([q_0,3/4]\), including their common endpoint.

The original first-cell mesh convention is retained. The plane HIGH expression uses \(b-\text{MESH}\), total length uses that same upper envelope, and supplied width deducts the mesh. The cell with top \(b_-\) covers the physical lower slot boundary \(17/500\); no argument treats \(b_-\) as an unrounded physical lower slot. The fixed mesh is not set to zero in the HIGH or supply accounting.

## Top row and rational geometry

The unchanged outward interval implementation uses exact rational endpoints, outward rounding after arithmetic operations to a \(10^{-60}\) grid, and integer square-root brackets. Division fails if the denominator interval contains zero. Its square-root lower and upper endpoints are justified by exact integer square inequalities. Thus the computed intervals enclose the real algebraic quantities, regardless of the displayed decimal precision.

The interval checks give

\[
\theta_{\min}\approx0.8748099600378937<874809963/10^9,
\]

and

\[
\operatorname{balancedTop}\approx0.040337615170746584<s.
\]

The complete exact rational enclosures are exported. The upper class-bound endpoint is strictly below the claim; the upper window endpoint is strictly below \(s\). The checks also prove \(\delta_*\geq0\), \(\kappa_2>0\), nonzero quadratic coefficient, positive root discriminant, positive plane slope, and positivity of the affine cutoff denominator on all of \([0,3/4]\) by its endpoint values. These are the top-row conditions in the frozen proof.

For completeness, a rational \(h\) is certified directly, rather than requiring the submission to encode the algebraic \(h_*\). Let \(P(x)=a_2x^2+a_1x+a_0\) have the unchanged `root_coeffs(lambda,B)`, and put

\[
D(x)=\tfrac{25}{2}\lambda-10+(6-9\lambda)x,
\quad U=(6\lambda+12)\theta-7\lambda-11,
\quad V=9\lambda\theta-7\lambda.
\]

The cleared top-row expression is

\[
N(x)=hP(x)+8D(x)(Ux+V).
\]

The verifier gives an exact Bernstein proof of \(N(x)>0\) for every \(x\in[0,3/4]\). The normalization \(24\lambda\) is positive. This expression is consistent with the frozen tangency construction: at \(\theta=\theta_{\min}\), \(Ux_*+V=0\), and the definition of \(h_*\) makes the derivative of \(N\) vanish at \(x_*\); its leading coefficient divided by \(24\lambda\) is exactly the frozen \(\kappa_2\). The direct rational check avoids losing this inequality when rounding \(h\).

At the submitted rational \((\theta,h,s)\), let \(A_c,B_c,C_c\) be `capped_quad(lambda,B,theta,h,s)`. The verifier proves

\[
A_c>0,\qquad 4A_cC_c-B_c^2>0.
\]

Hence the capped quadratic is positive for every real argument, by completing the square. This proves the required window inequality on its whole domain, including both boundaries. The rational geometry also satisfies \(h>2\ell\), \(2h-1-3\ell>0\), \(h<5\ell\), \(h<1\), and \(\ell>1/6\).

## Fixed stage deletions and LOW reserves

Write \(T_j=(k_j,I_j)\), and define

\[
K_j=\frac{q_j}{6(k_j-q_jI_j)},\qquad L_j=\alpha(1-q_jI_j/k_j).
\]

All price gaps and \(K_j\) are positive. Every \(q_j\) belongs to the original zero-density range \([3/5,3/4)\), with the stricter first-step brackets also checked. Every region price is at most the old price.

For a cell \(b\), begin with

\[
g_0=B-I_0+(6k_0-\lambda)b,
\quad a_0=K_0g_0+\alpha,
\quad d_0=(g_0+6I_0a_0)/(6k_0).
\]

After this deletion the corner state is

\[
A=B-(\lambda-1)b-d_0,\quad z=b-d_0,\quad M=1-6a_0.
\]

At each subsequent stage set

\[
g_j=A+(6k_j-1)z-I_jM,\quad a_j=K_jg_j+\alpha.
\]

For \(j<6\), set \(d_j=(g_j+6I_ja_j)/(6k_j)\), and replace \((A,z,M)\) by \((A-d_j,z-d_j,M-6a_j)\). This is exactly the unchanged `Chain.cascade(b)` recurrence. Deletions and amplifiers are tied to the original cell. They are not recomputed independently from individual child positions, and they are not replaced by independently maximized caps.

The verifier proves all these maps affine in \(b\), positive amplifiers and overshoots at both ends, nonnegative deletions, positive remaining lengths and slots, and the exact identities

\[
a_j-q_jd_j=L_j\qquad(0\leq j<6).
\]

The same general deletion identity applies at the terminal old-region stage. All seven LOW reserves satisfy the unchanged floor \(L_j\geq10^{-10}\). Their minimum is approximately \(1.1815145540\cdot10^{-10}\). In particular, the failed seven-level LOW floor noted in the historical proof is avoided by different junctions, not by lowering the floor.

## Joint HIGH reserves, source constraints, and returns

Put

\[
TA(b)=B-(\lambda-1)(b-10^{-8}).
\]

For depth \(d=2,\ldots,7\), the reserve is exactly

\[
H_d(x,b)=1-\rho(x)-xTA(b)
 +\sum_{i=0}^{d-2}(x-q_i)d_i(b)
 -c(x)\sum_{i=0}^{d-1}a_i(b).
\]

The verifier symbolically proves, for arbitrary \(x,b\), the telescoping identity

\[
H_d=(1-\rho(x))\left(1-6\sum_{i<d-1}a_i\right)
-x\left(TA-\sum_{i<d-1}d_i\right)
-a_{d-1}c(x)+\sum_{i<d-1}(a_i-q_id_i).
\]

Thus deletions are priced at their own source junctions \(q_i\), and the intermediate LOW reserves enter the return with the same signs as in the original joint split. No deletion is priced at the final junction to obtain an artificial gain.

The original source and return requirements remain: source scale at least one, total length at most \(TA\), actual amplifiers at most the coupled cell caps (plus accounted allowance), fixed deletion at most target at most deleted width, the source scale and volume inequalities of Report C, and `Pays` for each returned class at its own junction. Intermediate missing-radical coordinates and prefix fibres are retained by H2. At depth seven the count uses seven labels and six missing-radical coordinates, with the corresponding sum-of-prefix-misses cap. Zero density is applied once to the final rows; it is not independently applied to each prefix.

The algebraic monotonicity needed to use these inequalities is checked explicitly. If the source scale is increased by \(u\geq0\), the length reduced by \(v\geq0\), actual amplifiers reduced by \(w_i\geq0\), and effective deleted widths increased by \(t_i\geq0\), the change in the reserve is

\[
(1-\rho(x))u+xv+c(x)\sum_iw_i+
\sum_{i<d-1}(x-q_i)t_i\geq0.
\]

Every coefficient has the indicated sign on \([q_{d-1},3/4]\). The verifier checks the full symbolic identity and the junction ordering; it also proves the amplifier cost positive over the entire bin range. The intermediate scale cancels with coefficient

\[
-(1-\rho)-\rho+1=0.
\]

Realizing all the required coupled source bounds for every position is still H1–H3, exactly as in the frozen proof. The slack identity does not claim to construct those sources or to prove the missing fibre-count theorem.

For the remaining continuous bin range, the exact growth identity is

\[
H_d(x,b)-H_d(q,b)=(x-q)\left[
\frac{50(1-6\sum_{i<d}a_i)}{3(5-2x)(5-2q)}
-\left(TA-\sum_{i<d-1}d_i\right)\right],\quad q=q_{d-1}.
\]

The amplifier sum is less than \(1/6\). As \(x\) increases from \(q\), the fraction in brackets is at least its value at \(q\). Its endpoint growth lower bounds exceed \(3/10\). The reserve and growth at the starting junction are affine in \(b\), so checking both cell ends proves the inequality on the entire rectangle. At every depth the initial HIGH reserve exceeds \(3\cdot10^{-7}\). The minimum over all six depths and all cells and bins is at least the certified endpoint minimum, approximately \(3.27847642735\cdot10^{-7}\). Subtracting the preceding LOW reserves still leaves more than \(2.9\cdot10^{-7}\).

This argument covers \(x=q\), \(x=3/4\), both cell ends, and every intermediate point. It also proves the exact reserve split rather than only evaluating the proposed reserve formula numerically.

## Width floor and direct-deletion room

Let \(O(b)\) be the first child's overshoot over the **old** region:

\[
O(b)=B-(\lambda-1)b-d_0+(6\kappa-1)(b-d_0)-(1-6a_0).
\]

Use precisely the frozen width-floor rule

\[
W=\max\left(\frac25,\frac{\max\{O(b_-),O(s)\}}{7/1000}+\frac1{1000}\right).
\]

This is a rational stage parameter, given exactly by that expression and in expanded form in `certificate.json`. Approximately \(W=0.466183404284268\). The original direct-deletion room is

\[
D(b)=r_o(1-6a_0-2W)-[r_o(G_{\rm old}(b)+6a_0)-a_0],
\quad r_o=q_0/(6\kappa),
\quad G_{\rm old}=B-1+(6\kappa-\lambda)b.
\]

The verifier checks the unchanged condition

\[
\min_bD(b)\geq15/10000.
\]

Its actual minimum is approximately \(0.00150641526512294\), leaving more than \(6.4\cdot10^{-6}\) above the test. The overshoot and room are symbolically affine in \(b\), so endpoint bounds cover every cell. The checks also prove \(O(b)<(7/1000)W\), \(2W<1-6a_0\), and the original width slack \(1/100\leq6a_0\).

An explicit reflection choice is

\[
\delta_R=\max_bO(b)+10^{-6},\qquad \xi_R=1/1000.
\]

With pad \(10^{-12}\), the checks prove

\[
O(b)+2\,\mathrm{pad}<\delta_R,\quad
\delta_R+\xi_R<W/100,
\quad D(b)>L_0+r_o(1/100+3\,\mathrm{pad}).
\]

Thus neither the old-region overshoot nor the price of raising the width floor is omitted. Using the overshoot over a nested region in place of \(O\) would be incorrect.

As in the frozen five- and six-level certificates, feasibility of every deeper normalized position family and all its H4 caps remains conditional. The certificate does not claim that unchanged PR #67 hard-coded caps already work at seven levels. Here \(K\) reaches approximately \(14.01060351\), so H4 needs a bound at least this large. Region slopes, slot caps, zero-room choices, mask rebasing, supply exponents, meshes, and coverage of positions must be re-bracketed under the same H4. In particular the consequences of changing `zeroSaving` and the compounded mask constants are not asserted proved by this arithmetic certificate. The first-child width-floor test above is retained in full, alongside those conditional obligations.

## Loss and rounding accounting

The original \(10^{-8}\) cell mesh is present in \(TA\), the channels, and the supply deduction. The marked-bin width is checked to be the least multiple of \(10^{-8}\) whose square is at least its exact radicand. It is never rounded downward.

To show a concrete arithmetic allowance budget is available, the verifier takes additional arbitrary allowances \(\epsilon=10^{-16}\) and position-grid spacing \(g=10^{-18}\). Let

\[
K_{\max}=\max_jK_j,\quad
S_a=K_{\max}(3\epsilon+(1+6\kappa)g).
\]

The displayed budget is

\[
L=10^{-8}\sum_{j=0}^6q_j+
7\left[10\epsilon+\tfrac34(\epsilon+6\kappa g)+\tfrac{23}{7}S_a\right]
\approx4.95112097438\cdot10^{-8}.
\]

The verifier proves exactly that \(L\) is below the minimum HIGH reserve minus the sum of all seven LOW reserves, hence also below each depth's reserve after subtracting its preceding LOW reserves. The sum includes the original fixed mesh; it does not replace that mesh by the smaller position grid. This is an arithmetic illustration of room for the allowances. **H5 still requires that every actual analytic, label-binning, price-excess, mesh, pad, and rebasing loss be bounded by such a budget.** The script does not claim to derive all of those losses from missing Lean constructions. H5 is retained rather than discharged by assigning arbitrary small numbers to unproved quantities.

All rational junctions and \(s\) are used exactly as submitted. The class-bound comparison uses the upper directed interval endpoint; the window comparison also uses its upper endpoint. The independent rational split avoids approximate root-rounding in the frozen bin-top finder. At the physical slot boundary the original upward cell rounding and mesh deductions are retained. Symbolic affine extension and the growth identity cover the continuous domains, and the Bernstein intervals include their boundaries.

## Conditional conclusion

The frozen proof reduces the first-step bound to its ledger channels, top row and window, nested LOW and joint HIGH payments, and H1–H5. The certificate proves the arithmetic channel conditions on the whole first-step domain; the top-row signs, interval comparisons, and rational geometry; all LOW floors and coupled deletion identities; every depth's HIGH reserve and growth on its whole rectangle; and the unchanged old-region width-floor and direct-room tests. H1–H5 retain responsibility for the recursive dispatch, actual source and return families, joint count, gap cases, H4 re-bracketing, and complete loss absorption. Under those same hypotheses at depth seven, the argument of `depth3/PROOF.md` therefore gives the asserted rational bound.

This work was AI-assisted. It extends the supplied conditional nested-affine model and its certificate, building on the PR #67 port and the earlier nested-level work credited in the snapshot. It makes no claim to a complete Lean nonvanishing endpoint.
