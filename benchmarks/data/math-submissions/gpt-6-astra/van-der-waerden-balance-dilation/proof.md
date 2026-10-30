# Conditional coefficient 1/25 for the van der Waerden construction

## Statement and scope

Subject to the geometric inputs of the supplied historical manuscript, for every sufficiently large integer k and every integer r >= 2,

\[
 W_r(k)>k^{(1/25)k\lfloor\log_2r\rfloor}.
\]

The threshold is independent of r. The construction uses the source coordinate maps, adaptive meshes, primary-factor dilation, geometric keys, and digit product. This is a conditional refinement of that construction, not a claim of an independently formalized unconditional theorem or a practical finite coloring record. No numerical experiment is used as a proof.

The source is the supplied directory
`upstream/preprints/Quantitative-Superexponential-Bounds-for-van-der-Waerden-Numbers-September-23-2026/build/sections/`.
The inputs retained below are the mesh width, neighborhood, and truncated-complexity statements of `02-setup.tex`, the sign-arrangement lemma of `03-counting.tex`, the affinity statement of `05-dichotomy.tex`, and the elementary prime and digit-product inputs in `02-setup.tex` and `07-transfers.tex`. We state their precise contents where used. We also give the scale checks and proofs of the drift, return, separation, counting, and probability steps, so no improved sufficient parameter region is assumed. In particular, we do **not** take the original outer-color dichotomy or the numerical parameter checker as an input.

The improvement has three parts:

* Balance all eligible tests to a fraction theta close to 1/2, using the same local counts but a new concentration/LLL calculation.
* Test the light positions **outside the full blocks that contain a heavy position**. Those positions and the heavy blocks can then be added without overlap. A dichotomy using only the total number of light positions would lose a factor close to two here.
* Replace the cutoff D^2 by h0 = ceil(k^s), with s only slightly greater than the dimension exponent a. The actual local-lemma requirement is h0/(D log(D h0)) -> infinity. This reduces the dilation entropy from O(k^(3a) log k) to O(k^(a+s) log k).

All logarithms below are natural unless indicated otherwise. Parameters are fixed before k tends to infinity. Implicit constants are independent of r and of the allowed prime P.

## 1. Parameters and scales

Choose the following exact rational constants:

\[
\begin{split}
a&=1/3,&s&=67/200,&m&=3/4,&b&=33/100,&c&=1/25,\\
\theta&=99/200,&\epsilon&=1/2-\theta=1/200,&
\delta&=1/1000,&\rho_*&=1/200,&\tau&=1/1000,&\gamma&=49/100.
\end{split}
\]

The symbol rho_* is a bound on a fraction of nonregular positions; it is distinct from the source distance-to-cut function rho(Y). Set

\[
 D=\lceil k^a\rceil,\quad h_0=\lceil k^s\rceil,\quad
 M=\lceil k^m\rceil,\quad p=k^{-b},\quad H=k^{-2},\quad
 \eta=(1000D)^{-1}.
\]

Thus D -> infinity, h0/D -> infinity, h0 < 2M < k eventually, and 2M/k <= tau eventually. Ceilings multiply these power scales by bounded factors and do not alter any strict power gap used below.

Define, exactly as in the source but with this new cutoff,

\[
 \lambda=\prod_{\substack{\ell\le h_0\\\ell\text{ prime}}}
       \ell^{\lfloor\log(2M)/\log\ell\rfloor}.
\tag{1}
\]

There are at most h0 factors and each factor is at most 2M. Consequently

\[
 \log\lambda\le h_0\log(2M)=O(k^s\log k).
\tag{2}
\]

For every integer h <= 2M, its full ell-primary part divides lambda whenever ell <= h0: if ell^e divides h, then e <= floor(log(2M)/log ell). Hence for every integer t,

\[
 s'=h/\gcd(h,\lambda t)
\]

is either 1 or has every prime factor greater than h0, so s' > h0. In particular h divides lambda for h <= h0. This proves the required dilation property for the new cutoff; it does not require a prime number theorem or a new number-theoretic assumption.

Use the retained elementary prime input to choose k < P <= 2k^2. Let q be the least power of P at least k^(ck/D). Then

\[
 ck\log k\le D\log q\le ck\log k+D\log(2k^2),
 \qquad \log q=O(k^{1-a}\log k).
\tag{3}
\]

Since h0 < k < P, lambda is coprime to q. Put N=q^D and G=Z/NZ. Every nonzero element of G has order at least P>k, so the terms of each nonzero-step cyclic k-progression are distinct.

Set Hx=H/lambda and alpha=q^(-(D+1)). Use the source homomorphisms

\[
 x_i(n)=n/q^i\pmod1,\qquad y_i(n)=\lambda x_i(n),\qquad 1\le i\le D,
\]

with representatives xhat in [0,1)^D and ytilde in [-1/2,1/2)^D. Partition x-coordinates into intervals of width Hx; this is an exact partition since Hx^(-1)=k^2 lambda is an integer.

For clarity, the retained second mesh V is defined by

\[
 R=\log(1+1/(2\alpha)),\quad J=\lceil k^2R\rceil,\quad
 a_l=\alpha(e^{lR/J}-1)\quad(0\le l\le J),
\]

and the half-open intervals [-1/2+a_l,-1/2+a_(l+1)) and [1/2-a_(l+1),1/2-a_l). Let rho(Y) be circle distance from the cut 1/2. The retained mesh statements are:

1. For I in V and Y in its closure,
   \(H(\rho(Y)+\alpha)/4\le |I|\le2H(\rho(Y)+\alpha)\le2H.\)
2. If w=|I|, then every J in V whose closure meets the circular w-neighborhood of the closure of I has |J| >= w/16. That neighborhood contains at most 49 mesh endpoints.
3. Define V_eta(Y)=V(Y) for rho(Y)>=eta and V_eta(Y)=* otherwise. On every arc of length at most 4H, this output has at most 16004D breakpoints, including the threshold points. Values at equalities are determined by the indicated conventions.

Their hypotheses are unchanged: H=k^-2, eta=1/(1000D), alpha tends to zero, and R>=1 eventually. In addition

\[
 R\le(D+1)\log q=O(k\log k),\qquad |V|=2J=O(k^3\log k).
\tag{4}
\]

The bound on R follows from 1+q^(D+1)/2 <= q^(D+1) for q^(D+1)>=2. These mesh conclusions do not use h0=D^2. The full label is

\[
 L(n)=((U(x_i(n)))_{i=1}^D,(V(y_i(n)))_{i=1}^D)
 \in\mathcal L=U^D\times V^D.
\]

## 2. Global words and anchored local counts

We retain the sign-arrangement input: s affine hyperplanes in R^e have at most (e+1)^2(s+1)^e realized joint sign vectors in {-,0,+}^s, also on an arbitrary restricted parameter set. Equality signs are included. One proof selects a linearly independent set of the zero equations and then applies the usual region bound on their intersection, giving

\[
 \sum_{d=0}^e {s\choose d}\sum_{j=0}^{e-d}{s\choose j}
 \le(e+1)^2(s+1)^e.
\]

For any circle partition with F endpoints, the length-k word of a scalar path a+jb mod 1, with a,b in [0,1), is determined by comparisons a+jb with tau+z, where tau ranges over endpoints and -1<=z<=k. There are at most 3k^2 F comparison lines in (a,b). Thus at most 9(3k^2F+1)^2 words occur. Using

\[
 F=|U|+|V|=k^2\lambda+O(k^3\log k)
\]

and allowing the 2D coordinates to vary independently only enlarges the possible collection. It follows that

\[
 T_{\rm glob}\le[9(3k^2F+1)^2]^{2D},\qquad
 \log T_{\rm glob}=O(D(\log k+\log\lambda))
                 =O(k^{a+s}\log k).
\tag{5}
\]

This count records literal interval labels, not relabeled patterns. The exponent here is a+s=401/600.

Here is the local-pattern family, including its parameter conditions. Fix h0<h<=2M and t in {0,...,h-1}^D, and set s_i=h/gcd(h,lambda t_i). Coordinate i of the second system is stationary if s_i=1 and rotating otherwise; in the latter case s_i>h0 by (1). Consider paths

\[
 X_z=x+z(t+u)/h,\qquad Y_z=y+z(\lambda t+v)/h
 \quad(0\le z<h)
\tag{6}
\]

on the torus. A position is regular if rho(Y_(z,i))>=eta in all rotating coordinates. Record the full label at regular positions and * elsewhere. A word is eligible if it has a realization satisfying:

* At least h/2 positions are regular, and their full labels are pairwise distinct.
* |u_i|<=Hx in all coordinates and |v_i|<=H in rotating coordinates.
* In stationary coordinates, |v_i|<=|V(Y_(z,i))| at every regular position.

For every fixed full label beta, the number of pairs (t,eligible word) containing beta is at most

\[
 (CDh)^{6D}
\tag{7}
\]

for an absolute constant C. We give the count to verify that lowering h0 incurs no hidden factor. Fix t and an anchor position z0 whose label is beta. In each scalar coordinate use anchor representative a, in the interval of beta, and drift parameter b=u_i or v_i. At target z the unreduced value is

\[
 a+\tau_z+(z-z_0)b/h,
\]

where tau_z is a fixed rational translation. For first coordinates its range has length at most 3Hx and encounters at most four U endpoints. For stationary second coordinates the translation is integral, and |b|<=|I| for the anchor interval I. Targets lie in the neighborhood of I described above; at most 49 endpoints matter. For rotating second coordinates the anchor width is at most 2H and |b|<=H, so the target range has length at most 4H; at most 16004D breakpoints of V_eta matter. Each relevant endpoint has at most one lift in these ranges, since their lengths are less than one.

For all targets together there are at most 16004Dh affine comparisons in (a,b) in each scalar coordinate. The sign-arrangement bound gives at most C0(Dh)^2 scalar words. Multiplying over 2D coordinates, then multiplying by h^D choices of t and at most h anchors, bounds the count by

\[
 h^{D+1}[C_0(Dh)^2]^{2D}
   =C_0^{2D}D^{4D}h^{5D+1}\le(CDh)^{6D}.
\]

The scalar truncated outputs determine the regular-position mask as well as its full labels. No additional mask count is needed. This derivation is uniform in q, lambda, h0, and the anchor widths.

## 3. A stronger outer coloring and the new light-position tests

A label in a global word is heavy when its total multiplicity exceeds k/M; all other labels are light. For **each** global word w and **each** integer 1<=h<=2M, partition the first h floor(k/h) indices into full consecutive blocks of length h. Let B(w,h) be the union of those full blocks containing at least one heavy position. Let U(w,h) be the union of the other full blocks. Thus B and U are disjoint, U consists entirely of light positions, and

\[
 |B|+|U|=h\lfloor k/h\rfloor>k-h.
\tag{8}
\]

These sets depend only on the literal word and h. The final incomplete block is in neither set.

We prove that an outer label coloring c_* exists with both properties:

(A) Whenever |U(w,h)|>=delta k, each bit occurs at least theta |U(w,h)| times on U(w,h).

(B) For every eligible local pattern, each bit occurs on at least theta times the number of its regular positions.

Assign independent uniform bits to the finite label space. For each U(w,h) meeting the size condition, group equal labels consecutively in a list of its positions. Let wmax be the largest multiplicity of a label in this restricted list. Distribute this list cyclically into wmax rows. Each label has one group of length at most wmax, so it occurs at most once per row. Every row has at least

\[
 \left\lfloor |U|/w_{\max}\right\rfloor\ge\delta M-1
\tag{9}
\]

positions, since wmax<=k/M. Introduce a bad event for each row if either bit has fewer than theta times its size. There are at most 2Mk Tglob such events. Introduce one more bad event for every eligible pattern and its discrete parameters h,t, if either bit has fewer than theta times its number of regular positions. Every event tests l distinct labels; in the second family l>=h/2.

Let epsilon=1/2-theta=1/200 and beta=epsilon^2=1/40000. For l independent fair bits, their sum X satisfies

\[
 \mathbb E e^{t(X-l/2)}=(\cosh(t/2))^l\le e^{lt^2/8}.
\]

The inequality follows directly from cosh u <= exp(u^2/2), by comparing their power series. Taking t=4epsilon and applying Markov's inequality to both tails gives

\[
 \Pr(E)\le2e^{-2\epsilon^2l}=2e^{-2\beta l}.
\tag{10}
\]

This also bounds the strict bad-event inequalities at nonintegral thresholds. Give an event weight z_E=exp(-beta l(E)). The sum of weights of **all** first-family events is at most

\[
 2MkT_{\rm glob}\exp[-\beta(\delta M-1)]=o(1),
\tag{11}
\]

because m>a+s (gap 49/600) and (5) holds. For any fixed label, (7) bounds the total incident weight from the second family by

\[
 \sum_{h>h_0}(CDh)^{6D}e^{-\beta h/2}=o(1).
\tag{12}
\]

Indeed log(CDh)/h is decreasing for h>=h0 eventually, and

\[
 \sup_{h>h_0}\frac{6D\log(CDh)}h
 \le\frac{6D\log(CDh_0)}{h_0}
 =O(k^{a-s}\log k)\longrightarrow0.
\tag{13}
\]

Eventually this ratio is at most beta/4. The sum in (12) is then bounded by the convergent geometric tail sum_(h>h0) exp(-beta h/4), which tends to zero. This is the complete reason the smaller cutoff suffices: s>a, with gap 1/600. The discarded original restriction h0=D^2 is not needed.

Uniformly in the label, the combined incident weight is consequently at most beta/4 eventually. Join events whose label supports intersect. This is a dependency graph with independence from the joint sigma-algebra of all nonneighbors, since the label bits are independent. For an event testing l labels,

\[
 \sum_{F\in\Gamma(E)}z_F\le\beta l/4.
\]

The row sizes in (9) and local sizes h/2 tend uniformly to infinity; assume l>=80000. Then z_E<=exp(-2)<1/2. Using log(1-z)>=-2z for 0<=z<=1/2 gives

\[
 z_E\prod_{F\in\Gamma(E)}(1-z_F)
 \ge e^{-3\beta l/2}\ge2e^{-2\beta l}\ge\Pr(E).
\tag{14}
\]

The middle inequality holds because beta l/2>=1>log 2. Here is the complete finite local-lemma argument used with (14). For a set T of bad events, write A_T for the event that none of T occurs. Simultaneously induct on |T| to prove that Pr(A_T)>0 and Pr(E|A_T)<=z_E for every E outside T. For the empty set, positivity is immediate and (14) implies Pr(E)<=z_E. For a nonempty T, remove any F in T. The induction hypothesis gives

\[
 \Pr(A_T)=\Pr(A_{T\setminus\{F\}})
       (1-\Pr(F\mid A_{T\setminus\{F\}}))>0.
\]

To prove the conditional bound, partition T into T1=T intersect Gamma(E) and T2=T minus T1. If T1 is empty, independence from the joint sigma-algebra of nonneighbors gives Pr(E|A_T)=Pr(E)<=z_E. Otherwise order T1 and condition successively on its members' complements, starting with A_T2. Each conditioning set has size at most |T|-1, so the induction hypothesis gives

\[
 \Pr(A_{T_1}\mid A_{T_2})\ge\prod_{F\in T_1}(1-z_F).
\]

Using nonneighbor independence in the numerator and (14),

\[
 \Pr(E\mid A_T)
 \le \frac{\Pr(E\mid A_{T_2})}{\Pr(A_{T_1}\mid A_{T_2})}
 \le \frac{\Pr(E)}{\prod_{F\in T_1}(1-z_F)}
 \le z_E.
\]

The last inequality holds because T1 is a subset of Gamma(E), so the product over all neighbors is no greater than the product over T1. This completes the induction. Taking T to be the entire finite event family proves that all bad events can simultaneously be avoided. Summing row counts proves (A), and the local events prove (B). Fix this coloring and put c0(n)=c_*(L(n)).

This is a new balance proof, not an assumption that near-half balance is available. The enlarged first family costs only the polynomial factor 2Mk in (11).

## 4. Return geometry at the new scales

We retain the source affinity input: if samples of an affine torus path, at any subset of indices in {0,...,k-1}, have real representatives in one coordinate box of width at most 2H, then those representatives are affine in the index when 2kH<1. To recall its elementary argument, for j1<j2<j3 the integer vector

\[
 (j_3-j_2)(Z_{j_2}-Z_{j_1})-(j_2-j_1)(Z_{j_3}-Z_{j_2})
\]

has each coordinate of magnitude at most 2H(j3-j1)<1. It is zero; interpolation between the extreme indices gives the affine representative path. Our H satisfies 2kH=2/k<1.

Consider a progression with a heavy label having multiplicity m_*>k/M. Choose a smallest positive gap h between its occurrences, at indices j' and j'+h. Let u and v be the differences of their first and second real representative vectors. The smallest gap is at most the span divided by m_*-1, and affinity inside the full box gives

\[
 1\le h\le2M,\qquad
 |u_i|\le\frac{H_x}{m_*-1}\le\frac{2MH_x}{k},\qquad
 |v_i|\le\frac{2H}{m_*-1}\le\frac{4MH}{k}.
\tag{15}
\]

Here m_*-1>=k/(2M) eventually because M/k ->0. The integer vector v-lambda u has coordinates of magnitude at most 6MH/k<1; hence v=lambda u over the reals.

It follows that some t in {0,...,h-1}^D satisfies

\[
 x(n_j)=x(n_0)+j(t+u)/h,\qquad
 y(n_j)=y(n_0)+j(\lambda t+v)/h\quad\text{on the torus}.
\tag{16}
\]

For example take t to be the residue modulo h of the integer vector h xhat(d)-u. Also gcd(t1,...,tD,h)=1. Otherwise a common divisor e>1 would make the position j'+h/e have representative the convex combination, with weight 1/e, of the two representatives in their common full box. Half-open interval boxes are convex. This would produce a smaller positive return gap in the same label, a contradiction.

Thus t/h has order h on the torus. For indices incongruent modulo h, some coordinate of their rational displacement has circle distance at least 1/h from zero. The first-coordinate drift difference has magnitude at most k|u_i|/h<=2MHx/h. Their actual circle distance in that coordinate is at least

\[
 (1-2MH)/h\ge(1-2MH)/(2M)>H\ge H_x
\tag{17}
\]

eventually. Such positions have distinct full labels. In particular every full block of h indices has pairwise distinct labels, and all occurrences of a fixed full label are in a single residue class modulo h.

For every heavy index j_* (not necessarily in the original chosen label), write L_i=|V(y_i(n_(j_*)))|. Its label has multiplicity m'>k/M and extreme occurrence difference a'h, with a'>=m'-1 and a'<k/h. Their representative difference Delta_i is congruent to a' v_i and has magnitude at most L_i<=2H. Since

\[
 |a'v_i|\le4MH/h,\qquad 2H+4MH/h<1,
\]

these two congruent real numbers agree. Therefore

\[
 |v_i|\le L_i/(m'-1)\le(2M/k)L_i.
\tag{18}
\]

All scale conditions just used hold with our parameters. In explicit power form,

\[
 M/k=O(k^{-1/4}),\quad MH=O(k^{-5/4}),\quad
 MH/k=O(k^{-9/4}),\quad MH/(k\eta)=O(k^{-23/12}).
\tag{19}
\]

They tend to zero. The same estimates will justify every small-drift and neighborhood condition below.

## 5. Near-half rich-or-affine dichotomy

If the progression has no heavy label, U(w,1) is all k positions. Property (A) gives at least theta k>gamma k occurrences of each bit.

Otherwise use h,t,u,v from Section 4. If h<=h0, h divides lambda, so (16) has second-system real step w=v/h modulo one. Its total coordinate motion obeys k|w_i|<=4MH/h<1/2. Moreover w_i is in q^(-i) Z, since it is congruent to y_i(d). Thus a nonzero w_i has |w_i|>=q^(-i)>=q^(-D)>=alpha.

If the sampled centered representatives fail to equal ytilde(n0)+jw, one nonconstant coordinate segment meets a cut in 1/2+Z. This includes a cut met at an endpoint. Every sampled point then has distance from the cut at most k|w_i|, and the mesh width estimate gives

\[
 |V(y_i(n_j))|\le2H(k|w_i|+\alpha)
               \le2H(k+1)|w_i|<|w_i|.
\]

All distinct samples in that coordinate have circle separation |j-j'| |w_i|>=|w_i|, since the whole motion is less than 1/2. Thus no two can have the same full label, contradicting heaviness. Hence the entire centered path is affine. This short-period alternative uses exactly the retained adaptive-mesh geometry.

Now suppose h>h0. In a full block of h indices, a rotating coordinate has rational period s_i>h0. Its rational values form an equally spaced s_i-grid with each grid point repeated h/s_i times. The drift within the block is at most |v_i|<=4MH/k<eta by (19). A nonregular actual value therefore corresponds to a grid point within 2eta of the cut. An arc of length 4eta contains at most 4eta s_i+1 points of that grid (also when endpoints are included). The excluded fraction for this coordinate is at most 4eta+1/s_i. A union bound gives a total nonregular fraction at most

\[
 D(4\eta+1/h_0)=4/1000+D/h_0\le\rho_*=1/200
\tag{20}
\]

eventually. The margin rho_*-4/1000 is 1/1000 and D/h0 ->0. In particular the required h/2 regular positions are available.

A block containing a heavy index j_* is eligible. Distinctness follows from (17), and |u_i|<=Hx, |v_i|<=H follow from (15). In a stationary second coordinate, its rational step is integral, so every point of the block is within |v_i|<=(2M/k)L_i of the heavy anchor's interval. The retained neighborhood lemma says its own interval has width at least L_i/16. Eventually 2M/k<=1/16, so that width is at least |v_i|. This verifies the stationary drift condition even if j_* itself is not regular. Consequently property (B) supplies at least theta(1-rho_*)h occurrences of each bit on every such full block.

Use exactly B=B(w,h) and U=U(w,h) from Section 3. The blocks in B supply at least theta(1-rho_*)|B| of each bit. If |U|>=delta k, property (A) supplies an additional theta |U|, on disjoint positions, giving at least theta(1-rho_*)(k-h). If |U|<delta k, discard U and use |B|>k-h-delta k. Both cases therefore give at least

\[
 \theta(1-\rho_*)(1-\delta-\tau)k>\gamma k
\tag{21}
\]

occurrences of each bit, since h<=2M<=tau k. The exact positive margin in (21) is

\[
 \theta(1-\rho_*)(1-\delta-\tau)-\gamma
 =30799/20000000.
\tag{22}
\]

We have proved: every nonzero-step cyclic k-progression is either gamma-rich in both outer colors or has its entire centered second-system path affine. There is no loss from balancing overlapping light positions and heavy blocks; their supports in the preceding argument are disjoint by construction.

## 6. Two occurrences per geometric key

Retain the source key

\[
 \kappa(n)=\bigl((V(y_i(n)))_{i=1}^D,\ \lfloor\|q\widetilde y(n)\|_2^2\rfloor\bigr).
\tag{23}
\]

Any two distinct progression terms give points q ytilde separated by Euclidean distance at least one. Indeed lambda is a unit modulo q^D. For their nonzero difference after multiplication by lambda, choose an integer representative z and the largest a' in {0,...,D-1} such that q^(a') divides z. Write z=q^(a') u, where q does not divide u. In coordinate a'+1 the circle displacement is u/q, of circle distance at least 1/q. The difference of real representatives is at least this circle distance. Scaling by q proves separation. This reasoning works for prime-power q; u need not be a unit modulo q.

For all positions with a fixed second-system box, affinity puts their real vectors on a line. The separation just proved makes this line nonconstant when it contains two points. Parametrize it by signed arclength t from its nearest point to the origin. Squared norm is t^2+C. Among any three points, two have their signed parameters in the same one of (-infinity,0) and [0,infinity). If their absolute values are u<v, separation implies v-u>=1, so

\[
 v^2-u^2=(v-u)(v+u)\ge1.
\]

Two squared norms in the same half-open unit interval differ by strictly less than one. This is a contradiction. Therefore every key occurs at most two times on any progression. The proof includes separation equality and a band meeting the nearest point. A multiplicity of two is possible, for example t=+-1/2, so replacing this lemma by multiplicity one would be invalid.

## 7. Affine entropy, including literal identities

An affine progression has ytilde(n_j)=A+jB. Its norm values are

\[
 f(n_j)=q^2\|A+jB\|^2=\theta_0+\theta_1j+\theta_2j^2,
 \qquad 0\le f(n_j)\le Dq^2/4.
\]

Comparing the quadratic at each j with all integers from 0 to floor(Dq^2/4)+1 uses at most k(Dq^2+2) affine hyperplanes in the **three** real coefficient variables. The retained equality-aware arrangement bound therefore gives at most C[k(Dq^2+2)]^3 possible integer-part words. Allowing arbitrary coefficient triples only increases this upper bound.

A full-label word specifies all second-system boxes and, since c_* is fixed, all outer colors. Thus the number of literal signatures

\[
 ((c_0(n_j),\kappa(n_j)))_{0\le j<k}
\]

of affine progressions obeys

\[
 T_{\rm aff}\le C T_{\rm glob}[k(Dq^2+2)]^3.
\tag{24}
\]

In particular

\[
\begin{split}
 \log\max(1,T_{\rm aff})
 &\le O(k^{a+s}\log k)+6\log q+O(\log k)\\
 &=O((k^{401/600}+k^{2/3})\log k)
   =o(k^{67/100})=o(pk).
\end{split}
\tag{25}
\]

The two strictly positive power gaps are

\[
 (1-b)-(a+s)=1/600,\qquad (1-b)-(1-a)=a-b=1/300.
\tag{26}
\]

This is a derived entropy estimate: (2), (5), and the fixed three-dimensional count imply it. No upgraded affine-counting oracle is assumed, and keys in signatures are never renamed.

## 8. Sparse perturbation and probability bounds

Choose new independent Bernoulli variables E_kappa, one for every actual key in G, with Pr(E_kappa=1)=p. Define C(n)=c0(n)+E_(kappa(n)) mod 2. The outer coloring is already fixed, and p<1/2 eventually.

For a gamma-rich progression and a fixed desired color, at least gamma k positions must flip. Multiplicity at most two means at least gamma k/2 **specified distinct** keys must be one. Their independence bounds this probability by p^(gamma k/2). If some key has conflicting requirements, the event is empty, which still satisfies the bound. There is no binomial factor for selecting keys: the keys requiring flips are determined by the progression and the desired color. There are fewer than q^(2D) cyclic progressions with nonzero step. Hence

\[
\begin{split}
 R_k&\le2q^{2D}p^{\gamma k/2}\\
 &\le\exp\{(2c-\gamma b/2)k\log k
              +2D\log(2k^2)+\log2\}\longrightarrow0,
\end{split}
\tag{27}
\]

because

\[
 \gamma b/2-2c=17/20000>0,
\tag{28}
\]

and D log k=o(k log k).

For an affine signature and a fixed desired color, every entry prescribes the bit of its literal key. Inconsistent prescriptions again give probability zero. Otherwise, with t distinct keys and u prescribed ones, the probability is exactly p^u(1-p)^(t-u). We have t>=k/2 and p<=1-p, so it is at most (1-p)^(k/2)<=exp(-pk/2). Progressions with the same literal signature prescribe exactly the same bits, so counting signatures once is valid. Thus

\[
 A_k\le2T_{\rm aff}(1-p)^{k/2}
      \le\exp(-pk/2+o(pk))\longrightarrow0.
\tag{29}
\]

No independence between different progression events is needed in these union bounds. The dichotomy covers every progression, including overlaps between the alternatives. Eventually R_k+A_k<1, so some choice of the independent key variables yields a cyclic two-coloring with no monochromatic nonzero-step k-progression. From (3), N=q^D>=k^(ck).

## 9. Uniform transfer to every number of colors

Retain the source digit-product input; its argument is short enough to include. Given the cyclic coloring on base N, color each integer 0<=n<N^t by the tuple of colors of its t base-N digits. Suppose a positive-difference k-progression in this interval were monochromatic. Its difference d satisfies 0<d<N^t. Let i<t be the largest integer with N^i dividing d, and write e=d/N^i, so N does not divide e. The exact identity

\[
 \left\lfloor\frac{a+jd}{N^i}\right\rfloor
 =\left\lfloor\frac a{N^i}\right\rfloor+je
\]

shows, upon reducing modulo N, that the ith digits form a nonzero-step cyclic progression monochromatic under C, a contradiction. Translating the interval by one gives W_(2^t)(k)>N^t. For r>=2 take t=floor(log2 r) and embed the 2^t colors in r colors. Therefore

\[
 W_r(k)>N^{\lfloor\log_2r\rfloor}
       \ge k^{(1/25)k\lfloor\log_2r\rfloor}.
\]

All thresholds were chosen before t or r appeared. They depend only on the displayed fixed parameters and the retained absolute geometric constants. The prime choice is uniform in k<P<=2k^2, so a single threshold works for all integer r>=2.

## 10. Exact margins and what the checks establish

The decisive margins, all strict, are:

| Requirement | Exact positive margin |
|---|---:|
| s>a (local entropy versus cutoff) | 1/600 |
| m>a+s (global event entropy) | 49/600 |
| 1-b>a+s (affine label entropy) | 1/600 |
| a>b (affine norm entropy) | 1/300 |
| m>s (period cutoff below 2M) | 83/200 |
| 1>m (sublinear return length) | 1/4 |
| 1>a (prime-power rounding error) | 2/3 |
| rho_*>4/1000 | 1/1000 |
| theta(1-rho_*)(1-delta-tau)>gamma | 30799/20000000 |
| gamma b/2>2c | 17/20000 |
| theta>gamma (all-light case) | 1/200 |
| epsilon=1/2-theta>0 | 1/200 |
| 3-m-a>0 (drift versus eta) | 23/12 |

Positivity of a,s,m,b,c,delta,tau,epsilon and 1-delta-tau is explicit. For the local lemma beta=1/40000, incident weight is eventually <=beta/4, and the exponent gap between the tail 2beta and the LLL lower bound 3beta/2 is beta/2=1/80000.

The executable checks supplement this analytic argument: exact rational margins and boundary rejection controls; finite tests of dilation denominators, the half-open norm-band bound, and the disjoint light-mask row construction; and integrity checks on the supplied source files. They do not verify the imported geometric statements by finite sampling, do not supply an explicit finite k threshold, and are not a substitute for Sections 1–9. Baseline files are not edited.
