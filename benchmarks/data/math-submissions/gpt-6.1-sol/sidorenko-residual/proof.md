# A rank-adapted count for the nondirect residual

## Conditional statement and explicit dependency gap

Under the geometric activation input stated below, for the fixed graph and cuts in the snapshot, the nondirect residual contribution is

\[
                         O_D(q^{-D+189})
\]

for every even D >= 190, subject to precisely the upstream geometric activation and gain input specified below. This is a bound for this residual counting step only. It does not establish a new Sidorenko counterexample, a minimum counterexample, or a sufficient dimension for the full construction.

The analytic section of the original manuscript is not included in this snapshot: the saved upstream file is the finite-complex section. Accordingly the activation input is stated explicitly, not claimed to have been independently verified from that missing section. The argument applies to the unchanged isotropic activation law, with the following input:

* The pair spaces live in a nondegenerate alternating space V of dimension 2D over F_q. Each active pair space H_e has dimension two.
* Put L_i = sum_{e incident to i} H_e and k_i = dim L_i. The unchanged local activation law makes a configuration zero unless every L_i is totally isotropic, and bounds its activation probability by
  O_D(q^{-D sum_i k_i + sum_i binom(k_i,2)}).
* The remaining gain is at most O_D(q^{2m}), where m is the number of active pairs.
* In the residual under consideration, only pairs of cut coverage exactly 1/3 can be active, and nondirect profiles have k = dim(sum_e H_e), 2 <= k <= 2m-1, and d = sum_i k_i - 2k >= 1.

These are geometric/analytic inputs, not conclusions of the finite arithmetic audit. The identification of this input with the original activation law is an unresolved dependency in this offline snapshot; the theorem below is conditional on that identification. In particular the local-isotropy clause is essential: the coarse exponent printed in the snapshot alone does not imply it. If the upstream activation law does not have this clause and this probability bound in the same ambient space, the present improvement cannot be inferred merely from the saved coarse exponent. No new activation law is proposed here.

For example, the local probability bound holds for independent uniform Lagrangian activators in V: for a fixed isotropic r-space, the containment probability is

\[
 \prod_{j=0}^{r-1}(q^{D-j}+1)^{-1}
 \le q^{-Dr+\binom r2};
\]

it is zero for a nonisotropic space. This explains the exact geometric input being used; it is not a replacement for identifying the original activators.

## Deriving the geometric input for symmetric-graph activators

This paragraph gives an exact identification criterion for the retained activation input, so it need not be accepted as an unexplained probability formula. Suppose the original local activator is the graph of a uniform symmetric D-by-D matrix M, in V = F_q^D direct-sum F_q^D. Use the alternating form

\[
 \omega((x,y),(x',y'))=x\mathbin\cdot y'-y\mathbin\cdot x'.
\]

Its matrix is invertible over every field, including characteristic two. Symmetry of M makes its graph totally isotropic. Let L be a fixed r-space. Containment in a graph is impossible unless the first projection is injective on L. Otherwise write L as the graph of a linear map f:U -> F_q^D, where dim U=r. The condition that a symmetric M extend f is consistent exactly when

\[
 x\mathbin\cdot f(x')=x'\mathbin\cdot f(x)\qquad(x,x'\in U),
\]

which is precisely isotropy of L. Choose coordinates putting U first, transforming the dual coordinates contragrediently. The first r columns of a symmetric matrix then specify Dr-binom(r,2) independent entries: the top r-by-r symmetric block contributes r(r+1)/2, and the remaining rectangular block contributes (D-r)r. Thus the exact containment probability is q^{-Dr+binom(r,2)} when consistent. Independent matrices at the vertices give the displayed product probability.

Consequently either uniform Lagrangian activators or uniform symmetric-graph activators imply the exact input needed by this proof, without changing their distribution. This derivation does not assert that either distribution has been identified in the missing analytic section. If that section uses a different law, its implication of the same input remains a required upstream dependency.

## Scope of the conditional improvement

The finite-complex source and the baseline exponent alone do not prove the geometric activation input. In particular, one must not infer local isotropy merely from the positive binomial term in the old exponent: a one-sided counting estimate cannot be inverted to recover the original probability model. The argument below is a complete conditional dimension theorem, and its exact arithmetic checks do not resolve this identification. The benchmark's requirement of keeping the activation law fixed therefore remains an audit obligation, rather than an established fact of this submission.

## Fixed finite graph

Reconstruct the faces and cut bits directly from the saved, hash-pinned complex.tex. Every pair occurs in two faces. Keep the pairs whose face labels differ in exactly one bit. There are 28 such pairs, with degrees

\[
 (3,6,3,6,4,2,4,4,4,6,4,6,4).
\]

The excluded pairs are {0,2}, {4,8}, {5,6}, {5,10}, and {7,12}. Thus m <= 28 and the maximum degree of every active subgraph is at most six. This is the same allowed graph as in the baseline certificate, although its displayed cut bits differ from the source's displayed bit labels. Both reconstructions are checked, without changing either certificate.

## Adapted frames

Fix any actual active edge set A. Write W = sum_{e in A} H_e, dim W = k. Choose bases of the two-dimensional H_e, concatenate them, and extract a basis v_1,...,v_k of W. Each chosen vector is tagged by the edge from which it was extracted. If a_e vectors have tag e, then

\[
 a_e\in\{0,1,2\},\qquad \sum_e a_e=k.
\]

This basis choice always exists. Counting all such choices only overcounts configurations. Edge sets, tag assignments and orderings have boundedly many possibilities, independent of q, so their costs enter O_D(1). No division by a change-of-basis group is used.

For a vertex i, let B_i be the span of basis vectors tagged by incident edges. Set

\[
 r_i=\dim B_i=\sum_{e\ni i}a_e,
 \qquad t_i=k_i-r_i\ge0.
\]

Since each tag has two endpoints,

\[
 \sum_i r_i=2k,\qquad \sum_i t_i=d.
\]

These identities hold even when some local spans vanish and for disconnected active subgraphs.

## Independent equations on the frame

Isotropy of L_i forces any two distinct frame vectors belonging to B_i to be orthogonal. These are constraints on pairs of distinct indices in an independent frame. A pair of distinct indices is counted twice among the B_i exactly when its two tags are the same edge. Different simple graph edges share at most one endpoint. Consequently the number of distinct required equations is exactly

\[
 b=\sum_i\binom{r_i}{2}-\sum_e\binom{a_e}{2}.
\]

The count of frames satisfying these equations is at most q^{2Dk-b}. To prove this without assuming independence of polynomial equations, choose the frame vectors sequentially. When v_j is chosen, suppose c_j of the prescribed equations involve earlier vectors. Those earlier vectors are linearly independent. Nondegeneracy of the alternating pairing makes their c_j linear functionals on V linearly independent. Their common kernel has exactly q^{2D-c_j} elements. Requiring v_j also to avoid the previous span only reduces that count. Multiplication gives the claimed upper bound, since sum_j c_j=b. Empty, exceptional, and degenerate restricted-form strata are all included: only independence of the full frame and nondegeneracy on V are used.

## Completion of local and pair spaces

For a fixed frame, W and all B_i are fixed. The number of k_i-dimensional spaces L_i between B_i and W is

\[
 {k-r_i\brack t_i}_q
       =O_D(q^{t_i(k-k_i)}).
\]

We may count all such spaces, including nonisotropic ones; the orthogonality equations above remain necessary for every genuine configuration. The Gaussian-binomial upper bound has a constant depending only on the bounded dimensions.

For e={u,v}, let A_e be the span of the a_e frame vectors tagged by e. Coordinate independence gives

\[
 B_u\cap B_v=A_e.
\]

Moreover

\[
 \dim(L_u\cap L_v)\le a_e+t_u+t_v.
\]

Indeed, the map from L_u intersection L_v to (L_u/B_u) direct-sum (L_v/B_v) has kernel B_u intersection B_v. Every genuine H_e lies in L_u intersection L_v and contains A_e. If that intersection has dimension s_e, its possible two-spaces containing A_e number

\[
 {s_e-a_e\brack 2-a_e}_q
 =O_D(q^{(2-a_e)(s_e-2)})
 \le O_D(q^{(2-a_e)(t_u+t_v)}).
\]

If s_e<2 there is no such pair space. For a_e=2 there is at most one. These conventions cover every case. Choosing pair spaces independently, and allowing choices that fail to generate the selected local spaces, overcounts and hence preserves an upper bound.

## Combining the exponents

Multiply the frame count, local-completion counts, pair-completion counts, unchanged activation bound, and gain q^{2m}. The exponent is at most -Dd+R', where

\[
\begin{aligned}
 R'={}&-b+\sum_i t_i(k-k_i)+\sum_i\binom{k_i}{2}\\
      &+\sum_{e=\{u,v\}}(2-a_e)(t_u+t_v)+2m.
\end{aligned}
\]

The exact cancellation is

\[
 \binom{r_i+t_i}{2}-\binom{r_i}{2}
           +t_i(k-r_i-t_i)
       =kt_i-\binom{t_i+1}{2}.
\]

Therefore

\[
 R'=kd-\sum_i\binom{t_i+1}{2}
       +\sum_e\binom{a_e}{2}
       +\sum_i t_i\sum_{e\ni i}(2-a_e)+2m.
\]

Use binom(t+1,2) >= t for nonnegative integers t, binom(a_e,2) <= 1, and sum_{e incident to i}(2-a_e) <= 2 deg_A(i) <= 12. This yields the uniform estimate

\[
                    R'\le d(k+11)+3m.                 (1)
\]

This bound does not relax to just defect one: it applies to every positive defect and every tag, local-rank, intersection-rank, and restricted-form stratum.

## All defects and the target constant

For D >= 190,

\[
\begin{aligned}
 [-Dd+d(k+11)+3m]-[-D+189]
   &= (d-1)(k+11-D)+(k+11+3m-189).
\end{aligned}
\]

Here k <= 2m-1 <= 55, so k+11-D <= -124. Also

\[
 k+11+3m\le 5m+10\le150<189.
\]

Both terms are nonpositive for every d>=1. In fact this proof gives the stronger intermediate estimate -D+150 on the requested range D>=190; we submit the requested constant 189. There is no need to use the old coarse bound for defects at least two.

For fixed D there are finitely many active sets, rank choices, tags and combinatorial frame constraint patterns. All Gaussian-binomial and activation constants are absorbed in O_D(1). Summing the bounds therefore gives O_D(q^{-D+189}). At even D>=190 this tends to zero. All upstream conditions needed elsewhere in the construction remain in force.

## Reproducibility and limits

Run `python3 -B solution/verify.py && python3 -B solution/check_activation.py`. It independently reconstructs the allowed graph from both saved descriptions, checks the source hash, exhausts all 28,476 relaxed positive-defect triples with 1<=m<=28, 2<=k<2m, and 1<=d<=4m-2k, and checks the cancellation identity with exact integers. A separate arithmetic implementation in frame_arithmetic.py is compared with this independent reconstruction. The analytic inequality above also checks all defects without enumeration. A small finite-field test verifies the sequential frame-equation count and the exact overlap formula by direct enumeration.

These executable checks supplement the dimension argument; they do not certify the missing upstream analytic section. In particular this submission is conditional on the explicitly displayed geometric activation input being the original law. That dependency must be audited against the original analytic argument before treating the improvement as an unconditional consequence of the snapshot.
