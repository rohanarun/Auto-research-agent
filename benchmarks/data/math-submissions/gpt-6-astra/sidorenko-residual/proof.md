# An adapted-basis bound for the nondirect residual term

## Claim and exact analytic dependency

Under the upstream geometric residual reduction specified below, the fixed nondirect residual term is

\[
O_D(q^{-D+189})\qquad\text{for every even }D\ge190.
\]

In fact the dimension estimate proved here gives the stronger internal bound
\(O_D(q^{-D+148})\). The submitted metric is 189. This is a claim only about the nondirect residual counting step. It is not a dimension bound for the full construction, a change of graph or cuts, or a smaller Sidorenko counterexample.

Here is the precise imported analytic input, before applying the crude arrangement count. Its specification matters: the scalar exponent printed in `proof.tex`, by itself, does not imply this input.

* The ambient space is a common nondegenerate symplectic space \(V\) of dimension \(2D\) over \(\mathbb F_q\).
* An equality-profile residual configuration assigns a two-dimensional subspace \(U_e\) to each active pair \(e\). Its local spans \(W_i=\sum_{e\ni i}U_e\) are totally isotropic for that **same** symplectic form. Put \(K=\sum_eU_e\), \(k=\dim K\), and \(k_i=\dim W_i\).
* After the unchanged upstream reduction and gain bound, the absolute weight of any fixed configuration is at most
  \[
  C_Dq^{2m-D\sum_i k_i+\sum_i\binom{k_i}{2}},                 \tag{1}
  \]
  where \(m\) is the number of active pairs. There are boundedly many other discrete profile choices, absorbed in \(C_D\).
* The equality-profile exclusion in `proof.tex` applies: a pair with separation probability greater than \(1/3\) cannot be active.

These are analytic dependencies, not conclusions of the arithmetic checks. In particular the common-form isotropy assertion is essential. The offline snapshot contains the original finite-complex section but not the original analytic section. Thus applicability to that missing section is explicitly conditional on its having the geometric reduction just stated; it has not been independently verified here. No assertion that the aggregate scalar bound alone suffices is made.

For clarity, (1) is the usual unchanged Lagrangian activation bound: if an independent uniform Lagrangian is required to contain an isotropic \(r\)-space, its probability is
\[
\prod_{j=D-r+1}^{D}(q^j+1)^{-1}
\le q^{-Dr+\binom r2}.
\]
Multiplying the local bounds and the source's gain \(q^{2m}\) gives (1). To verify the formula, let \(L_D\) be the number of Lagrangians in a symplectic \(2D\)-space. Counting pairs consisting of a Lagrangian and one of its lines gives
\(L_D(q^D-1)/(q-1)=(q^{2D}-1)L_{D-1}/(q-1)\).
Since \(L_0=1\), \(L_D=\prod_{j=1}^D(q^j+1)\). Lagrangians containing an isotropic \(r\)-space correspond to Lagrangians in its symplectic quotient of dimension \(2(D-r)\). This proves the probability formula. For a nonisotropic local span the probability is zero. This calculation explains the geometric input; it does not establish its provenance in the unavailable analytic source.

## The unchanged finite data

Read the 22 face triples and three cut bits directly from the hash-pinned original table. Every pair occurs twice. The five pairs separated by more than one of the three cuts are
\[
\{0,2\},\ \{7,12\},\ \{4,8\},\ \{5,6\},\ \{5,10\}.
\]
They are inactive by the equality-profile exclusion. The remaining graph has 28 edges and degree sequence, in the original vertex order,
\[
(3,6,3,6,4,2,4,4,4,6,4,6,4).                             \tag{2}
\]
Consequently every active graph has \(m\le28\) and maximum degree at most six. No modification of faces, cuts, probabilities, or activation parameters is used. Our verifier reads the original table, rather than substituting the differently encoded cut certificate in `experiments/original-baseline.json`.

## Two counting facts

For fixed integers \(0\le r\le N\), the number of \(r\)-subspaces of an \(N\)-dimensional space is
\[
{N\brack r}_q
=q^{r(N-r)}\prod_{j=0}^{r-1}\frac{1-q^{-(N-j)}}{1-q^{-(r-j)}}
\le C_Nq^{r(N-r)}.                                      \tag{3}
\]
All dimensions used below are bounded in terms of the fixed graph, so every factor \(C_N\) is uniform in \(q\). Formula (3) also counts extensions of a fixed subspace after passing to a quotient.

If \(A\subset W\) and \(B\subset Z\), then
\[
\dim(W\cap Z)\le\dim(A\cap B)+(\dim W-\dim A)+(\dim Z-\dim B). \tag{4}
\]
Indeed the map \(W\cap Z\to(W/A)\oplus(Z/B)\), sending \(x\) to its two quotient classes, has kernel \(A\cap B\). This proof allows all possible intersections and degeneracies.

## Adapted bases and the defect

Fix an active edge set and a configuration. Choose a basis of \(K\) from the union of bases of the two-dimensional spaces \(U_e\). Assign each selected vector to the edge from whose basis it was selected. Let \(b_e\in\{0,1,2\}\) be the number selected from edge \(e\), and let \(B_e\) be their span. The selected vectors are globally independent, so their subspaces \(B_e\) form a direct sum and
\[
\sum_e b_e=k.
\]
Define
\[
A_i=\bigoplus_{e\ni i}B_e,\quad n_i=\dim A_i=\sum_{e\ni i}b_e,
\quad \delta_i=k_i-n_i,\quad T_i=\sum_{e\ni i}(2-b_e).
\]
Then
\[
\sum_i n_i=2k,\quad d:=\sum_i k_i-2k=\sum_i\delta_i,
\qquad 0\le\delta_i\le T_i\le12.                         \tag{5}
\]
For distinct endpoints \(i,j\) of an active edge \(e\), the directness of the selected basis and simplicity of the graph imply
\[
A_i\cap A_j=B_e.                                         \tag{6}
\]
If \(d=0\), all \(W_i=A_i\), whence \(U_e\subset W_i\cap W_j=B_e\). Since \(\dim U_e=2\), every \(b_e=2\), and the \(U_e\) form a direct sum. Thus every nondirect configuration has \(d\ge1\). Moreover nondirectness gives
\[
2\le k\le2m-1\le55.                                    \tag{7}
\]
The upper bound \(\sum_i k_i\le4m\) gives \(d\le4m-2k\le108\). Every defect stratum is therefore included below.

## Counting frames while retaining necessary isotropy equations

Fix the discrete numbers \(b_e,k_i\), and order the selected basis vectors in blocks labelled by their edges. Any two selected vectors whose labels have a common endpoint belong to its isotropic \(W_i\); their symplectic pairing is therefore zero.

The number of distinct required pair equations is
\[
c_0=\sum_i\binom{n_i}{2}-\sum_e\binom{b_e}{2}.            \tag{8}
\]
In the first sum a pair from the same edge is counted twice, once at each endpoint; the second sum removes the extra copy. Distinct edges in a simple graph share at most one endpoint. There are no diagonal equations because the form is alternating, including in characteristic two.

Expose the ordered selected vectors successively. If \(r\) equations concern the next vector, they pair it with \(r\) distinct earlier selected vectors. Those vectors are independent. Nondegeneracy identifies \(V\) with \(V^*\), so these \(r\) linear functionals are independent. There are exactly \(q^{2D-r}\) solutions before imposing independence of the new vector; imposing independence only reduces the number. Multiplication over all steps gives the rigorous upper bound
\[
q^{2Dk-c_0}                                               \tag{9}
\]
for selected frames. The equations remain independent as linear constraints at each step even when the Gram matrix of the already exposed frame is singular. Thus (9) needs no nonsingularity assumption on \(K\) or on a local restriction of the form.

Every configuration admits at least one such frame. We use an upper bound for all frame-decorated configurations; **we do not divide by a presumed frame multiplicity**.

## Counting all extensions and edge completions

Fix a selected frame. It determines \(K\), all \(B_e\), and all \(A_i\). The possible \(W_i\subset K\) of dimension \(k_i=n_i+\delta_i\), containing \(A_i\), number at most
\[
Cq^{\delta_i(k-k_i)}.                                    \tag{10}
\]
This follows from (3) in \(K/A_i\). We can discard all additional isotropy conditions in this step, since (9) has already imposed a necessary subset of them and we are proving an upper bound.

For edge \(e=ij\), (4) and (6) give
\[
\dim(W_i\cap W_j)\le b_e+\delta_i+\delta_j.
\]
A completion \(U_e\) must contain \(B_e\) and be a two-dimensional subspace of this intersection. Set \(t_e=2-b_e\). If \(\delta_i+\delta_j<t_e\), there is no completion and the stratum is empty. Otherwise (3), in the quotient by \(B_e\), bounds the number of completions by
\[
Cq^{t_e(\delta_i+\delta_j-t_e)}.                           \tag{11}
\]
This remains an upper bound if the actual intersection is smaller than its displayed dimension bound. The product of (10) and (11) may include choices where the recovered local spans or global span are too small. Such choices are merely overcounted; every original configuration is included. No general-position assumption occurs anywhere in this count.

## Exact exponent cancellation

Multiply (9)--(11) by (1). Since \(\sum_i k_i=2k+d\), the powers involving \(D\) give exactly \(-Dd\). The remaining exponent is
\[
R=-c_0+\sum_i\delta_i(k-k_i)
 +\sum_{e=ij}t_e(\delta_i+\delta_j-t_e)
 +2m+\sum_i\binom{k_i}{2}.                                \tag{12}
\]
Let \(m_0=\#\{e:b_e=0\}\). Equation (12) simplifies identically to
\[
\boxed{R=kd+\sum_i\left(\delta_iT_i-\binom{\delta_i+1}{2}\right)
       +2k-m-m_0.}                                       \tag{13}
\]
For completeness, the local cancellation is
\[
-\delta_i(n_i+\delta_i)
 +\binom{n_i+\delta_i}{2}-\binom{n_i}{2}
=-\binom{\delta_i+1}{2}.
\]
Also \(\sum_{e=ij}t_e(\delta_i+\delta_j)=\sum_i\delta_iT_i\), and for each \(b\in\{0,1,2\}\),
\[
\binom b2-(2-b)^2+2=2b-1-\mathbf1_{\{b=0\}}.
\]
Summing this last identity proves the edge contribution \(2k-m-m_0\). These identities involve integers only.

## A bound uniform over every positive defect

For every nonnegative integer \(\delta\), \(\binom{\delta+1}{2}\ge\delta\). Equations (2), (5), (7), and (13) yield
\[
R\le(k+11)d+2k-m
\le66d+82
\le148d\qquad(d\ge1).                                   \tag{14}
\]
Here \(2k-m\le3m-2\le82\). This handles every defect at once, including all singular intersections, vanishing \(b_e\), disconnected active supports, and all locally deficient spans. Empty parameter strata make no contribution.

For \(D\ge190\),
\[
-Dd+R\le-(D-148)d\le-D+148\le-D+189.                     \tag{15}
\]
There are finitely many active edge sets, integers \(b_e,k_i\), and vector-label orders, with number depending only on the fixed graph. The constants from (3) are also uniformly bounded. Summing the estimates therefore preserves \(O_D(q^{-D+189})\). In particular this nondirect term tends to zero for every even \(D\ge190\), subject to the analytic dependency stated at the start. Earlier residue, transverse, and other full-construction thresholds are not altered.

## Reproducible checks and their limits

`check_frame_bound.py` checks the hash-pinned original table, reconstructs its eligible graph, checks the cancellation identities by exact integer arithmetic, and exhausts every relaxed integer triple \((m,k,d)\) with \(2\le m\le28\), \(2\le k<2m\), \(1\le d\le4m-2k\). Its maximum for the deliberately relaxed upper bound \(((k+11)d+2k-m)/d\) is exactly 148. This maximizer need not be realizable.

`check_geometry.py` independently exhausts subspaces of \(\mathbb F_2^4\), tests (4) on every pair of coordinate-space extensions, checks independence of all pairing functionals from independent subsets, and verifies the Lagrangian containment formula. These finite checks are sanity checks; the arguments above prove the statements in all dimensions and fields.

The original `audit_residual.py` and `audit_support.py` are run without modification and their outputs are compared to the supplied expectations. These validate baseline preservation, not the new analytic dependency. No check contacts the network or changes a baseline verifier.

## Why the geometric dependency cannot be silently dropped

There is a concrete obstruction to deducing the target from just the displayed scalar weight envelope and abstract subspace dimensions. This obstruction concerns an artificial unrestricted arrangement model, **not** the actual fixed upstream activation law.

In an unrestricted ambient space of dimension N=2D, use all 28 eligible edges. For the triangle on vertices (6,8,12), choose its three edge planes to contain one common line, with independent quotient lines. Choose the other 25 edge planes in direct sum with the four-dimensional triangle span and with one another. Then k=54. At each triangle vertex the local dimension drops by one, and elsewhere it equals twice the eligible degree. Thus sum k_i=109, d=1, and sum binom(k_i,2)=447.

The number of such configurations is Theta_D(q^(54N-107)). Indeed there are [N choose 1]_q choices for the common line, then

    (q^(N-1)-1)(q^(N-1)-q)(q^(N-1)-q^2)/(q-1)^3

ordered independent quotient lines. Each later plane, when the existing span has dimension s, has exactly q^(2s)[N-s choose 2]_q choices disjoint from that span. Here s runs through 4,6,...,52. The leading exponent is

    (N-1)+3(N-2)+25(2N-4)=54N-107.

The common line is uniquely determined by the triangle planes, so this count has no hidden multiplicity. Giving every such configuration exactly the scalar upper-envelope weight in (1) produces a total of order q^(-D+396), since -107+56+447=396. This demonstrates that the scalar envelope by itself cannot prove 189. It does not contradict the conditional theorem: its proof imposes the necessary common-form orthogonality equations, which this unrestricted model omits. `check_dependency.py` independently verifies all the numerical identities in this example.
