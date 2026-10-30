# Conditional coefficient `1/25`: logarithmic short-period cutoff and hybrid balance

## 1. Result and retained inputs

We prove, conditional on the geometric, affinity, counting, prime-power, perturbation, and digit-product lemmas in the pinned source manuscript, that for all sufficiently large `k`, uniformly for every integer `r>=2`,

\[
 W_r(k)>k^{(1/25)k\lfloor\log_2r\rfloor}.
\]

We retain the source coordinate maps, adaptive meshes, primary-factor dilation, light/heavy decomposition, eligible-pattern count, affine-signature count, prime input, and digit product. We change three estimates and prove each change below:

1. the short-period cutoff is `h0=ceil(K D log k)`, rather than `D^2`;
2. the outer coloring is balanced to `1/2-epsilon`, and its disjoint light-outside-block and regular-heavy-block contributions are added;
3. the norm-band multiplicity is sharpened from four to two.

No improved sufficient region is assumed. The source lemmas are used in their parameter-uniform forms proved by their displayed arguments; all new asymptotic requirements and exact margins are derived here.

## 2. Parameters

Fix

\[
 K=10^9,\quad a=\frac13,\quad m=\frac34,\quad b=\frac{33}{100},
 \quad c=\frac1{25},
\]
\[
 \epsilon=\delta=\rho=\frac1{1000},\qquad
 \gamma=\frac{49}{100}.
\]

For sufficiently large `k`, put

\[
 D=\lceil k^a\rceil,
 \quad M=\lceil k^m\rceil,
 \quad h_0=\lceil K D\log k\rceil,
 \quad p=k^{-b},
\]
\[
 H=k^{-2},\qquad \eta=(10^6D)^{-1}.
\]

Define the source primary-factor dilation

\[
 \lambda=\prod_{\substack{\ell\le h_0\\\ell\ {
m prime}}}
 \ell^{\lfloor\log(2M)/\log\ell\rfloor}.
\]

Let `P` be the source prime in `(k,2k^2]`, and let `q` be the least power of `P` satisfying

\[
 q\ge k^{ck/D}.
\]

The fixed constant `K` only enlarges the eventual threshold; it changes no power exponent.

## 3. Dilation and entropy

The source dilation proof works for every integer cutoff `h0`: every prime-power divisor of an integer `h<=2M` whose prime is at most `h0` divides `lambda`. Thus

\[
 h\le h_0\Longrightarrow h\mid\lambda,
\]

and for every `h<=2M` and integer `t`,

\[
 s=\frac h{\gcd(h,\lambda t)}
\]

is either one or exceeds `h0`. Moreover,

\[
 \log\lambda\le h_0\log(2M)=O(D(\log k)^2).
\]

Since `a<m`, eventually `h0<2M`; since `a<1`, eventually `h0<k<P`, so `gcd(lambda,q)=1`. The source primary-factor and distinct-term arguments are therefore unchanged.

The source arrangement proof for global label words gives

\[
 \log T_{\rm glob}=O(D(\log k+\log\lambda))
 =O(D^2(\log k)^2).
\]

Thus the old cubic power `D^3` is replaced legitimately by `D^2` (up to logarithms), solely because the source dilation cutoff need only dominate the local-pattern entropy, not equal `D^2`.

Minimality of `q` gives

\[
 ck\log k\le D\log q
 \le ck\log k+D\log(2k^2),
\]

and hence

\[
 \log q=O(k^{1-a}\log k),\qquad N=q^D\ge k^{ck}.
\]

## 4. Near-half outer coloring

Assign independent uniform bits to the source full labels. We use the same two kinds of tests as the source, with a refined first family described below.

For `l` distinct label variables, Hoeffding gives

\[
 \Pr\bigl(\text{some bit occurs fewer than }(1/2-\epsilon)l\bigr)
 \le2e^{-2\epsilon^2l}.
\]

Give such an event weight `z=e^{-epsilon^2 l}`.

### 4.1 Global light-outside-block tests

For a full-label word having no heavy label, define `O` to be all `k` positions. For a word with a heavy label, choose the first heavy label in a fixed ordering and form its closest-return period `h` exactly as in the source. If `h>h0`, partition the first `h floor(k/h)` positions into full `h`-blocks. Call a block used if it contains a heavy position, and let `O` be the union of the unused full blocks. (Words whose canonical period is at most `h0` need no global event, since the short-period argument makes them affine.) In every case in which `O` is defined, all its positions are light, and `O` is determined by the literal label word.

If `|O|>=delta k`, group equal labels and distribute the groups cyclically among `wmax` rows, as in the source light-row construction. Since every label in `O` is light,

\[
 w_{\max}\le k/M,
 \qquad l\ge\left\lfloor\frac{|O|}{w_{\max}}\right\rfloor
 \ge\delta M-1.
\]

Test every row. The word determines its heavy labels, canonical `h`, used blocks, `O`, and rows, so there are at most `k Tglob` such events. Their total weight is

\[
 kT_{\rm glob}e^{-\epsilon^2(\delta M-1)}=o(1),
\]

because

\[
 D^2(\log k)^2=o(M).
\]

### 4.2 Eligible local-pattern tests

For every source eligible pattern, test its distinct regular labels. It has `l>=h/2` variables. The source anchored-pattern lemma says that the number of events of period `h` incident to a specified label is at most

\[
 (CDh)^{6D}.
\]

Consequently their incident weight is bounded by

\[
 \sum_{h>h_0}(CDh)^{6D}e^{-\epsilon^2h/2}.
\]

For `h<=2M`, eventually `log(CDh)<=3 log k`. At `h0>=K D log k`,

\[
 \frac{6D\log(CDh)}h\le\frac{18}{K}<\frac{\epsilon^2}{4}.
\]

The ratio decreases for larger `h`. Hence the displayed sum is at most a geometric tail

\[
 \sum_{h>h_0}e^{-\epsilon^2h/4}=o(1).
\]

Although actual periods satisfy `h<=2M`, extending the final geometric majorant to all integers `h>h0` only enlarges it.

### 4.3 Local-lemma criterion

Uniformly in every label, the total incident event weight tends to zero. For sufficiently large `k`, make it at most `epsilon^2/8`. An event on `l` labels then has neighboring weight at most `epsilon^2 l/8`. Since `z<1/2` eventually and `log(1-z)>=-2z`,

\[
 z_E\prod_{F\sim E}(1-z_F)
 \ge \exp(-\epsilon^2l-\epsilon^2l/4)
 \ge2e^{-2\epsilon^2l}
\]

for all sufficiently large `l`; here every tested `l` tends to infinity because `M,D,h0` do. The source finite asymmetric local lemma therefore produces one outer coloring with both properties:

* every tested row in `O` contains at least `(1/2-epsilon)` of each bit;
* every eligible pattern contains at least `(1/2-epsilon)` of each bit among its regular positions.

This proves the refined balance rather than assuming it.

## 5. Hybrid disjoint balance

Fix a progression. If it has no heavy label, all `k` positions are light. Apply the global row test to the whole word; each outer bit occurs at least `(1/2-epsilon)k` times.

Suppose it has a heavy label and let `h` be the canonical closest-return period.

If `h<=h0`, the source short-period proof applies because `h|lambda`; it gives a real affine lift of all centered second coordinates.

Now assume `h>h0`. The source rational-path proof makes every full block containing a heavy position an eligible pattern. In a rotating coordinate, the source grid estimate excludes a fraction at most

\[
 4\eta+1/h_0.
\]

A union bound over `D` coordinates gives total nonregular fraction

\[
 D(4\eta+1/h_0)
 \le \frac4{10^6}+\frac1{K\log k}
 \le\rho
\]

for sufficiently large `k`. Thus the used blocks have at least `(1-rho)` regular fraction. Their regular positions contribute at least

\[
 (1/2-\epsilon)(1-\rho)U
\]

of each bit, where `U` is the total length of the used blocks.

The unused full blocks form `O`, disjoint from the used blocks. The incomplete remainder has length less than `h<=2M=o(k)`, and

\[
 U+|O|\ge k-h=k-o(k).
\]

If `|O|>=delta k`, the global rows add at least `(1/2-epsilon)|O|` occurrences of each bit. Therefore

\[
 \#\text{each bit}
 \ge(1/2-\epsilon)(1-\rho)(k-o(k)).
\]

If `|O|<delta k`, discard it and use `U>=k-h-delta k`; then

\[
 \#\text{each bit}
 \ge(1/2-\epsilon)(1-\rho)(1-\delta)k-o(k).
\]

Using `(1-x)(1-y)>=1-x-y` repeatedly, the latter coefficient is at least

\[
 \frac12-\epsilon-\rho-\delta
 =\frac{497}{1000}.
\]

Since

\[
 \frac{497}{1000}-\gamma
 =\frac7{1000}>0,
\]

every nonaffine progression has at least `gamma k` occurrences of each outer bit for all sufficiently large `k`. Notice that the two contributions are genuinely disjoint: regular positions in used blocks and positions in unused blocks. This removes the overlap that would invalidate simply adding all light positions to all heavy-block positions.

The drift, relative-width, mesh-neighborhood, and short-affinity arguments require only

\[
 M/k\to0,
 \quad MH\to0,
 \quad MH/k\to0,
 \quad MH/(k\eta)\to0.
\]

Their respective power exponents are negative; the last is `m+a-3=-23/12`.

## 6. Norm-band multiplicity two

Fix a perturbation key. Source affinity puts all occurrences of its second-system box on one real affine line, and the source digit separation makes the scaled points pairwise distance at least one.

Parameterize the line by signed arclength `t` from its nearest point to the origin. Squared norm is `t^2+C`. Among any three points, two lie on the same half-line. If their absolute parameters are `0<=u<v`, separation gives `v-u>=1`, whence

\[
 v^2-u^2=(v-u)(v+u)\ge1.
\]

But two squared norms in the same half-open band `[j,j+1)` differ by less than one. Therefore every key occurs at most twice.

## 7. Exact power margins

The parameter estimates used above and in the affine count reduce to

\[
 2a<m<1,
 \qquad b<a,
 \qquad 2a<1-b.
\]

For our choices their exact margins are

\[
 m-2a=\frac1{12},\qquad 1-m=\frac14,
\]
\[
 a-b=\frac1{300},\qquad 1-b-2a=\frac1{300}.
\]

Thus global entropy is `o(M)`, while both `D^2(log k)^2` and `log q` are `o(k^(1-b))=o(pk)`.

## 8. Rich-progression probability

For a nonaffine progression and target bit, at least `gamma k` opposite-bit positions must flip. Multiplicity two gives at least `gamma k/2` distinct prescribed flip variables. There are fewer than `q^(2D)` cyclic progressions, so

\[
 R_k\le2q^{2D}p^{\gamma k/2}.
\]

Consequently

\[
 \log R_k\le
 \left(2c-\frac{\gamma b}{2}\right)k\log k
 +2D\log(2k^2)+\log2.
\]

The exact leading margin is

\[
 \frac{\gamma b}{2}-2c
 =\frac{49}{100}\frac{33}{100}\frac12-\frac2{25}
 =\frac{17}{20000}>0.
\]

Since `D=o(k)`, this proves `R_k=o(1)`.

## 9. Affine-progression probability

The retained three-parameter norm arrangement, together with the improved global entropy, gives

\[
 \log\max(1,T_{\rm aff})
 =O(D^2(\log k)^2)+6\log q+O(\log k)
 =o(pk).
\]

Multiplicity two means every affine signature contains at least `k/2` distinct keys. For a fixed target bit, consistent prescriptions therefore have probability at most `(1-p)^(k/2)`; inconsistent prescriptions have probability zero. Hence

\[
 A_k\le2T_{\rm aff}(1-p)^{k/2}
 \le\exp(-pk/2+o(pk))=o(1).
\]

Thus `R_k+A_k<1` eventually, and a binary cyclic coloring of

\[
 N=q^D\ge k^{(1/25)k}
\]

with no monochromatic `k`-term progression exists.

## 10. Digit-product transfer and uniformity

Apply the retained digit-product lemma with `s=floor(log_2 r)`. It uses `2^s<=r` colors and gives a progression-free interval of length at least

\[
 N^s\ge k^{(1/25)k\lfloor\log_2r\rfloor}.
\]

Every threshold above depends only on fixed constants and `k`; `r` enters only here. The sufficiently-large-`k` threshold is therefore independent of every integer `r>=2`.

## 11. Open-region check

With logarithmic cutoff, `a` may approach `1/3`, `b` may approach `1/3`, and hybrid balance makes `gamma` approach `1/2`. Multiplicity two and the rich union bound require

\[
 c<\frac{\gamma b}{4},
\]

whose supremum in this region is `1/24`. The claimed `1/25` is strictly interior, with coefficient margin

\[
 \frac{\gamma b}{4}-c=\frac{17}{40000}>0.
\]
