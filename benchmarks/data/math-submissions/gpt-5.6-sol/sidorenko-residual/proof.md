# Nondirect residual bound with constant 189

## Scope

This proves only the estimate for the nondirect residual counting step in the preserved snapshot.  The finite graph, its three cuts, and the activation laws are not changed.  No assertion is made about the dimension threshold of the entire construction, or about a smaller Sidorenko counterexample.  The analytic reduction to the residual exponent is the same upstream dependency used by the baseline proof.

Let `E` be the active-pair set, `m=|E|`, let `U_e` be its two-dimensional pair spaces, let `k` be the dimension of their global span, and let `k_v` denote a local span dimension.  Put

\[
 d=\sum_v k_v-2k>0.
\]

The upstream counting lemma, including its gain estimate, gives exponent

\[
 -Dd+R,\qquad
 R=-k^2+2mk-2m+\sum_{v,c}\binom{k_{v,c}}2.                 \tag{1}
\]

Here `c` is the unique activation coordinate of an active pair.  This is the uncontracted form of the source's local parameter count.  Contracting the three summands at a vertex and applying
`sum_c C(k_{v,c},2) <= C(sum_c k_{v,c},2)` gives the older displayed term `C(k_v,2)`.  We retain the three summands instead.  This is not a new activation rule: the two faces on an eligible edge differ in exactly one of the three fixed cut bits, and the activation law makes mixed-coordinate local coefficients zero.

The equality-stratum argument already present in the snapshot removes the five edges separated by more than one cut.  Reconstructing incidences from the saved faces and bits leaves 28 edges, split into fixed activation classes of sizes

\[
 (9,10,9).                                                   \tag{2}
\]

At vertex `v`, activation block `(v,c)` has cap

\[
 b_{v,c}=2\deg_c(v).
\]

The 38 nonzero caps have total 112 and satisfy the exact identity

\[
 \sum_{v,c}\binom{b_{v,c}}2=128.                            \tag{3}
\]

## Relation-space reduction

Set `V=direct_sum_{e in E} U_e`, let `V -> W` be the global summation map, and put

\[
 L=\ker(V\to W),\qquad r=\dim L=2m-k.
\]

For each vertex and activation coordinate let `V_{v,c}` be the corresponding star block and put `L_{v,c}=L intersect V_{v,c}`.  Write

\[
 A=\sum_{v,c}L_{v,c},\quad
 u=\dim(L/A),
\]

where repeated local generators are counted before taking the sum, and let `t` be their projection excess (the nullity of that local-generator sum map after its forced two endpoint occurrences are removed).  Row reduction of the endpoint-incidence diagram gives

\[
 r=d-u+t,
 \qquad
 \epsilon=d-2u\ge0,
 \qquad
 0\le t\le\binom{\epsilon+1}{2}.                            \tag{4}
\]

For completeness, here is the dimension argument for (4).  Order a basis of `L/A` by its first nonzero edge coordinate and eliminate those pivots from the local-generator matrix.  Every quotient pivot consumes two endpoint occurrences, giving `2u<=d`.  In the residual matrix, group rows by the number `j` of earlier quotient pivots in their support.  A row contributing to projection excess needs a new ordered pair consisting of its last unmatched endpoint and one of those earlier pivots.  With `epsilon=d-2u` unmatched endpoint occurrences, at most

\[
 1+2+\cdots+\epsilon=\binom{\epsilon+1}{2}
\]

such independent rows survive.  Conversely the rank-nullity identities for the local-generator map are `d=2r-s` and `u=r-s+t`, where `s` is the total local-kernel dimension.  Eliminating `s` gives `r=d-u+t`.  This proves (4).  Notice in particular that a nondirect stratum has `u>=1`, so defect one is impossible.

If `x_{v,c}` is the loss from the cap in block `(v,c)`, then

\[
 \sum_{v,c}x_{v,c}=2r-d,
 \qquad 0\le x_{v,c}\le b_{v,c},                            \tag{5}
\]

and the last term of (1) is at most

\[
 H(2r-d)=\max_{\sum x=2r-d}
     \sum_{v,c}\binom{b_{v,c}-x_{v,c}}2.                    \tag{6}
\]

This is a pure finite integer problem.  Convexity says to spend the loss in increasing cap order.  The verifier also obtains every value of `H` by an independent unsorted convolution dynamic program and checks equality for all 113 possible losses.

Replacing an active support by the whole eligible graph is safe in (1), (5), and (6): adjoining a missing two-space raises `m` and adds its two endpoint activation caps; it enlarges the relaxed feasible set, while the coefficient `2k-2` of an added active edge is nonnegative.  Thus it suffices to use `m=28` and the caps reconstructed in (2)-(3).

## All defect strata

For each `2<=d<=6`, enumerate

\[
 1\le u\le\lfloor d/2\rfloor,
 \quad 0\le t\le\binom{d-2u+1}{2},
 \quad r=d-u+t,                                             \tag{7}
\]

and insert `k=56-r` and (6) into (1).  To compare with `-D+189` at the smallest permitted dimension `D=190`, the required scalar is

\[
 B(d)=R-190(d-1).                                           \tag{8}
\]

The executable audit records the maximizing tuple and loss vector for every defect.  Defect one has no nondirect tuple.  Every recorded value for defects two through six is at most 189.  For `d>=7`, the already certified support-independent estimate `R<=1196` gives

\[
 R-190(d-1)\le1196-6\cdot190=56<189.                        \tag{9}
\]

For `D>190`, subtracting `(D-190)(d-1)` from (8) only improves the comparison.  Hence every nondirect residual stratum has exponent at most

\[
 -D+189
\]

for every even `D>=190`.  There are finitely many profiles for fixed `D`, exactly as in the imported counting lemma, so summing them proves

\[
 \boxed{O_D(q^{-D+189})}.
\]

The proof depends conditionally on the snapshot's upstream residual counting lemma and its activation-coordinate vanishing law.  It does not purport to verify other analytic steps or thresholds.

## Reproduction

Run:

```sh
python3 -B solution/verify_residual189.py
```

The script uses only exact integer arithmetic, reconstructs the eligible graph from `experiments/original-baseline.json`, compares two independent computations of (6), enumerates every small-defect tuple in (7), and writes `solution/residual189-evidence.json`.
