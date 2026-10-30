# A conditional coefficient 1/25, with uniformity in the number of colors

## Statement and dependencies

Subject to the retained geometric and coordinate inputs of the supplied September 23, 2026 manuscript, the following refinement of its construction holds: there is an integer K, independent of r, such that for every integer k >= K and r >= 2,

\[
 W_r(k)>k^{(1/25)k\lfloor\log_2 r\rfloor}.
\]

The retained inputs are the source coordinate model, its adaptive-mesh width, neighborhood and truncated-complexity lemmas, affinity inside a small box, the closest-return/rational-path/relative-width arguments, the elementary prime bound, the affine-comparison arrangement lemma, and the digit product. We explain their hypotheses and their transfer below. We do **not** assume either a stronger balance proposition or a stronger sufficient parameter region. Those are proved here. In particular, the historical coefficient 1/193 and its numerical checker are not used as mathematical inputs. The conditional status is the same reliance on the source coordinate/geometric construction as in the task; no numerical experiment is offered as a proof of these inputs.

All logarithms below are natural. All constants in asymptotic estimates can depend on the fixed rational parameters but never on r, the allowed prime P, or the chosen progression. All limits are as the integer k tends to infinity.

## 1. Parameters and the retained coordinate model

Fix

\[
 a=1/3,\quad m=3/4,\quad b=33/100,\quad c=1/25,
 \quad \varepsilon=1/1000,\quad\theta=1/2-\varepsilon=499/1000,
\]
\[
 \delta=1/1000,\quad \nu=1/200,\quad\gamma=49/100.
\]

Take

\[
 D=\lceil k^a\rceil,\quad M=\lceil k^m\rceil,
 \quad h_0=\lceil D(\log k)^2\rceil,\quad p=k^{-b},
 \quad H=k^{-2},\quad\eta=(1000D)^{-1}.
\]

The change to h0 is essential; it is not D squared. Retain the source primary-factor dilation, with this new cutoff:

\[
 \lambda=\prod_{\substack{\ell\le h_0\\\ell\text{ prime}}}
 \ell^{\lfloor\log(2M)/\log\ell\rfloor}.
\]

For large k, h0 < 2M < k. The retained elementary prime lemma supplies a prime k < P <= 2k². Let q be the smallest power of P at least k^(ck/D), and put N=q^D, G=Z/NZ. Thus

\[
 ck\log k\le D\log q\le ck\log k+D\log(2k^2),\qquad
 \log q=O(k^{1-a}\log k).
\]

The dilation has only prime factors less than P, hence is a unit modulo q^D. A nonzero element of G has order at least P > k, so all terms of a nonzero-step k-term cyclic progression are distinct.

Use the source homomorphisms

\[
 x_i(n)=n/q^i\pmod1,\qquad y_i(n)=\lambda n/q^i\pmod1
 \quad (1\le i\le D),
\]

with representatives hat x in [0,1)^D and tilde y in [-1/2,1/2)^D. Let Hx=H/lambda and alpha=q^(-(D+1)). The partition U has intervals of width Hx. The partition V is exactly the source adaptive mesh, defined by R=log(1+1/(2alpha)), s=ceil(k²R), a_l=alpha(exp(lR/s)-1), and intervals [-1/2+a_l,-1/2+a_(l+1)) and [1/2-a_(l+1),1/2-a_l). Its cut distance is rho(Y)=dist_T(Y,1/2). Record literal interval identities in

\[
 L(n)=((U(x_i(n)))_{i=1}^D,(V(y_i(n)))_{i=1}^D).
\]

The mesh inputs give, at every Y in the closure of I in V,

\[
 H(\rho(Y)+\alpha)/4\le |I|\le2H(\rho(Y)+\alpha)\le2H.
\]

Every interval whose closure meets the circular |I|-neighborhood of the closure of I has width at least |I|/16, and that neighborhood contains at most 49 mesh endpoints. The truncated output V_eta, equal to V when rho >= eta and to a dummy symbol otherwise, has at most 16004D breakpoints on an arc of length at most 4H. Equality and half-open conventions are retained throughout.

These mesh inputs still apply: R <= (D+1)log q=O(k log k), alpha tends to zero, H tends to zero, and |V|=O(k³ log k). Their proofs use only these relations and eta=(1000D)^(-1), and do not use h0=D².

## 2. Improved dilation and entropy estimates

Every prime factor in lambda contributes at most log(2M). Consequently

\[
 \log\lambda\le h_0\log(2M)=O(D(\log k)^3).
\]

No prime-distribution estimate beyond the retained elementary prime input is used. If 1 <= h <= 2M, every prime-power part of h whose prime is at most h0 divides lambda. Hence

\[
 s=h/\gcd(h,\lambda t)
\]

is either 1 or has every prime factor greater than h0; in particular s > h0 in the latter case. Also h divides lambda if h <= h0. This is exactly the source dilation property, now with a smaller cutoff.

We give the global count explicitly. For a circle partition with F endpoints and scalar start and step in [0,1), the labels of the k samples are determined by comparisons of a+jb with endpoint lifts tau+z. At most 3k²F affine lines suffice. The source arrangement lemma, including zero signs, gives at most 9(3k²F+1)² scalar words. Taking independent starts and steps in all 2D coordinates enlarges the actual family, so with F=|U|+|V|,

\[
 T_{glob}\le[9(3k^2F+1)^2]^{2D},\qquad
 \log T_{glob}=O(D(\log k+\log\lambda))
 =O(D^2(\log k)^3).
\]

For our parameters this is O(k^(2/3)(log k)^3)=o(M). This improvement follows from the new cutoff and the arrangement count, rather than from assuming a new entropy inequality.

We also retain and justify the source anchored local count. For h0 < h <= 2M and t in {0,...,h-1}^D consider paths

\[
 X_z=x+z(t+u)/h,\qquad Y_z=y+z(\lambda t+v)/h\quad(0\le z<h).
\]

Call coordinate i stationary when h/gcd(h,lambda t_i)=1 and rotating otherwise. A position is regular when rho(Y_(z,i)) >= eta in every rotating coordinate. An eligible recorded word records its full label at regular positions and a dummy elsewhere; at least h/2 positions must be regular, their labels must be distinct, |u_i| <= Hx, |v_i| <= H in rotating coordinates, and |v_i| <= |V(Y_(z,i))| at every regular position in stationary coordinates.

For a fixed full label beta, the number of (t,eligible word) pairs containing beta is at most

\[
 (CDh)^{6D},                                      \tag{2.1}
\]

with an absolute C, independent of lambda and q. To see this, anchor at a position z0 carrying beta. For a scalar coordinate the unreduced target is a+tau_z+(z-z0)b/h, with fixed tau_z. In a first coordinate a ranges over an interval of width Hx and |b|<=Hx: at most four relevant endpoints occur for each target. In a stationary second coordinate a lies in the anchor interval I and |b|<=|I|; translations are integral, so the neighborhood lemma gives at most 49 relevant endpoints. In a rotating second coordinate the target range has length at most 4H, so the truncated-complexity bound gives at most 16004D breakpoints. Across all h targets there are at most 16004Dh lines in (a,b), with at most C0(Dh)² joint outputs by the arrangement lemma. The product over 2D coordinates determines both the dummy mask and every recorded literal label. There are h^D choices of t and h anchors. The resulting C0^(2D) D^(4D) h^(5D+1) is bounded by (CDh)^(6D). Discarding restrictions away from the anchor only enlarges this count. This argument does not require h0=D².

The crucial new cutoff satisfies

\[
 \sup_{h>h_0}\frac{6D\log(CDh)}h
 \le\frac{6D\log(CDh_0)}{h_0}=O(1/\log k)=o(1),       \tag{2.2}
\]

for large k. The inequality follows from monotonicity of log(CDh)/h in this range. Also D/h0 <= (log k)^(-2). These are the two uses formerly served by h0=D².

## 3. Near-half balance with complementary light-block tests

Here is the new balance argument, including its probability proof. A full label in a global k-word is heavy if its total multiplicity is greater than k/M, and light otherwise. For every realized global full-label word w and every integer 1 <= h <= 2M, partition the first h floor(k/h) positions into full blocks of length h. Let B(w,h) be the union of those full blocks that contain no heavy label. All its positions are light. If |B(w,h)| >= delta k, introduce balance tests on these selected positions as follows.

Group equal labels, and distribute the resulting list cyclically into wmax rows, where wmax is the largest selected-label multiplicity. No label occurs twice in a row, because each group has length at most wmax. Every row has at least

\[
 \lfloor |B(w,h)|/wmax\rfloor\ge\delta M-1
\]

positions, since wmax <= k/M. Fix grouping choices deterministically as functions of (w,h). There are at most 2Mk Tglob such rows in total. In particular h=1 tests the full word when all its labels are light.

Independently color all full labels with fair bits. For each such row, a bad event means that some bit appears fewer than theta times its length. Add one bad event for each eligible local word and its h,t, with the same definition on its regular positions. Each event E tests l(E) distinct label variables. We claim that all these events can be avoided simultaneously.

For X a sum of l independent fair bits,

\[
 E\exp(t(X-l/2))=(\cosh(t/2))^l\le\exp(lt^2/8).
\]

The inequality follows by comparing the cosh power series with exp(t²/2). Markov's inequality with t=4epsilon, and the analogous lower tail, therefore gives

\[
 P(E)\le2\exp(-2\varepsilon^2 l(E)).                 \tag{3.1}
\]

This derivation is valid for either nonintegral threshold and for strict bad events. Set tau=epsilon²/2=1/2000000 and z_E=exp(-tau l(E)). The total weight of the row family is bounded by

\[
 2MkT_{glob}\exp[-\tau(\delta M-1)]=o(1),            \tag{3.2}
\]

because log Tglob=o(M). For a specified label, the local-family incident weight is at most

\[
 \sum_{h>h_0}(CDh)^{6D}\exp(-\tau h/2)=o(1).         \tag{3.3}
\]

Indeed (2.2) eventually bounds the logarithmic counting term by tau h/4, leaving a geometric tail sum of exp(-tau h/4), which tends to zero because h0 tends to infinity. Thus uniformly for every label beta, the sum of z_E over events using beta tends to zero. In particular it is eventually at most tau/4.

Use the dependency graph joining events sharing a label variable. Independence of the underlying bits gives the required independence from the joint sigma-algebra of nonneighbors. For an event E, the sum of neighboring weights is at most tau l(E)/4. The minimum tested length tends to infinity (at least delta M-1 for rows and h0/2 for local words). Thus all z_E<1/2 eventually, and log(1-z)>=-2z gives

\[
 z_E\prod_{F\sim E}(1-z_F)
 \ge\exp[-3\tau l(E)/2]
 \ge2\exp[-4\tau l(E)]\ge P(E).
\]

The middle inequality holds when (5tau/2)l(E)>=log 2. The finite asymmetric local lemma from the source applies, so a fixed coloring c_* avoids all tests. Summing rows proves that each bit occurs on at least theta |B(w,h)| of the selected light-block positions whenever |B(w,h)|>=delta k. Every eligible local pattern has at least theta times its regular-position count of each bit.

The finite local lemma needs no new assumption: its elementary conditional-probability proof in source Section 4 applies to precisely this finite family and this dependency graph. The row-mask count and bounds (3.1)--(3.3) establish its hypotheses here.

## 4. Geometric transfer and the improved dichotomy

Put c0(n)=c_*(L(n)). We prove that every nonzero-step cyclic progression either has at least gamma k occurrences of each outer bit, or its entire centered y-lift is affine.

For clarity, we track the retained geometric arguments at the new scales. Affinity says that representatives lying in a common box of coordinate widths <=2H are affine as a function of their indices, since for j1<j2<j3 the interpolation defect is integral and each coordinate has absolute value <=2H(j3-j1)<1. It therefore vanishes. Here 2kH=2/k<1.

If a progression has no heavy label, apply the h=1 row-mask test: B contains all k positions, hence each bit occurs at least theta k > gamma k times. Otherwise choose a heavy label and its closest return gap h. Let u and v be the first- and second-coordinate representative differences at that return. The affinity argument at repeated visits gives

\[
 1\le h\le2M,\qquad |u_i|\le2MHx/k,
 \qquad |v_i|\le4MH/k,\qquad v=\lambda u.             \tag{4.1}
\]

Indeed the repeated label has more than k/M visits, its extreme gap is at least (number of visits minus one) times its closest gap, and its widths are Hx and at most 2H. The integral vector v-lambda u has coordinate absolute values <=6MH/k<1 and is zero.

The rational-path argument then gives t in {0,...,h-1}^D with

\[
 x(n_j)=x(n_0)+j(t+u)/h,\qquad
 y(n_j)=y(n_0)+j(\lambda t+v)/h\pmod1,               \tag{4.2}
\]

and gcd(t_1,...,t_D,h)=1. To verify the latter, a common divisor e>1 would put the representative at the intervening gap h/e on the segment between the two selected visits, inside their convex half-open box, contradicting closest return. The rational vector t/h has order h. At incongruent indices modulo h some first-coordinate rational displacement has circle distance at least 1/h; its drift is at most 2MHx/h. The actual distance exceeds Hx eventually, since h<=2M, MH tends to zero, and M Hx tends to zero. Thus distinct residue classes modulo h have distinct full labels.

For any heavy label, not just the one selecting h, its extreme visits differ by A h with integer A at least its multiplicity minus one. The second-coordinate representative difference Delta_i is congruent to A v_i and |Delta_i| <= L_i, its interval width. Since |Av_i|<=4MH/h and L_i<=2H, their integral difference is smaller than one in absolute value and vanishes. Consequently at every heavy position j*,

\[
 |v_i|\le (2M/k)|V(y_i(n_{j*}))|.                   \tag{4.3}
\]

These are the source drift, path and relative-width arguments, explicitly retained. Their small quantities remain small: M/k ->0, MH ->0, MH/k ->0, and MH/(k eta) ->0. For our parameters their respective powers of k are -1/4, -5/4, -9/4, and -23/12.

If h<=h0, then h divides lambda. Hence the centered second path has real candidate step w=v/h, of total coordinate motion k|w_i| <=4MH/h <1/2. Its nonzero coordinates lie in q^(-i) Z, so |w_i| >=q^(-D)>=alpha. If the sampled centered path does not equal its real affine candidate, some nonzero coordinate segment crosses or meets a cut. All its samples then have rho<=k|w_i|, so their V-interval widths are at most 2H(k+1)|w_i|<|w_i|. Distinct samples in that coordinate have circle separation |j-j'| |w_i|>=|w_i| because the total motion is less than 1/2. No two can share a V interval, contradicting the heavy label. Thus this case gives an affine lift. Closed segments and the mesh half-open convention cover exact cut hits as in the source argument.

Now suppose h>h0. Partition into full h-blocks and a remainder of size less than h. In a rotating coordinate the rational path in a full block visits an equally spaced s_i-grid, s_i>h0, with equal multiplicities. Its drift from the grid is at most |v_i|<eta eventually. Nonregular samples have grid values within 2eta of the cut, an arc of length 4eta. This contains at most 4eta s_i+1 grid points, including any endpoints. A union bound gives a nonregular fraction at most

\[
 D(4\eta+1/h_0)=1/250+D/h_0
 \le1/250+(\log k)^{-2}<\nu=1/200                 \tag{4.4}
\]

for all sufficiently large k. Thus there are at least (1-nu)h regular positions in each full block.

Every full block containing a heavy position is eligible: its labels are distinct by (4.2); its first drift is <=Hx and rotating drift <=H by (4.1). In a stationary coordinate, all points in the block are within |v_i| <=(2M/k)L_* of the heavy position's interval I_* of width L_*, by (4.3). The neighborhood input gives every encountered interval width at least L_*/16, which exceeds |v_i| eventually because 2M/k<=1/16. Thus the stationary eligibility condition holds at every regular position. The heavy position itself need not be regular. Such a block therefore contributes at least theta(1-nu)h of each bit by Section 3.

Let S be the total size of these heavy-containing full blocks and let L be the total size of the other full blocks. The latter are exactly B(w,h), so they consist of light positions. Write t_rem<h for the remainder, so S+L=k-t_rem. If L>=delta k, the row-mask test and the heavy-block tests together give at least

\[
 \theta L+\theta(1-\nu)S\ge\theta(1-\nu)(k-h)
\]

positions of each bit. If L<delta k, the heavy blocks alone give more than theta(1-nu)(k-h-delta k). Since h<=2M=o(k), eventually h<=delta k. In either case each bit consequently occurs at least

\[
 \theta(1-\nu)(1-2\delta)k>\gamma k.                \tag{4.5}
\]

This disjoint-block combination is why there is no loss from choosing between a light-rich case and a heavy-rich case. The strict difference in (4.5) is exactly 551199/100000000. Equations (4.1)--(4.4) verify eligibility and the new cutoff rather than postulating a better dichotomy.

## 5. Two occurrences per key

Retain the source geometric key

\[
 \kappa(n)=((V(y_i(n)))_{i=1}^D,\lfloor\|q\tilde y(n)\|_2^2\rfloor).
\]

Any key occurs at most twice on a progression. First, affinity puts the representatives of all visits to one second-system box on an affine line. Any two distinct group elements give scaled representatives separated by Euclidean distance at least one: write a nonzero representative of lambda(n-n') as q^A u with 0<=A<D and q not dividing u. In coordinate A+1 its circle displacement is u/q, at least 1/q from zero. Scaling by q proves separation. This argument works for prime powers q, even when u is not a unit.

Parameterize the resulting nonconstant line by signed arclength t from its closest point to the origin. Squared norm is t²+C. Among three parameters two lie on the same half-line (partition into t<0 and t>=0). Their absolute values 0<=u<v satisfy v-u>=1, hence v²-u²=(v-u)(v+u)>=1. Two squared norms in a common half-open band [j,j+1) differ by strictly less than one. This contradiction proves the claimed multiplicity two, including equality in separation and a band meeting the closest point.

## 6. Affine signatures and their entropy

A literal signature records (c0(n_j),kappa(n_j)) at every position of an affine lifted progression. Write tilde y(n_j)=A+jB. Then

\[
 \|q(A+jB)\|^2=\theta_0+\theta_1j+\theta_2j^2,
 \qquad 0\le\|q(A+jB)\|^2\le Dq^2/4.
\]

Its integer-part word is determined by at most k(Dq²+2) affine comparisons in the three real coefficient parameters. The arrangement lemma including equality signs bounds its number by C[k(Dq²+2)]³. A full-label word specifies the literal boxes and all outer colors, because c_* is fixed. Multiplication of these two upper bounds (with no independence assertion) therefore yields

\[
 T_{aff}\le C T_{glob}[k(Dq^2+2)]^3,
\]
\[
 \log\max(1,T_{aff})
 \le O(D^2(\log k)^3)+6\log q+O(\log k)
 =O(k^{2a}(\log k)^3+k^{1-a}\log k)
 =o(k^{1-b}).                                     \tag{6.1}
\]

For our parameters both polynomial gaps 1-b-2a and a-b equal 1/300. Thus fixed logarithmic factors are absorbed. This derives the improved affine entropy from the construction; it is not an assumed improved counting region. Keys are never renamed, which is essential for the next probability argument.

## 7. Perturbation, counting events, and the cyclic coloring

For every realized key choose independent E_kappa in {0,1} with P(E_kappa=1)=p, and put C(n)=c0(n)+E_kappa(n) modulo 2. The outer coloring is already fixed; this is new randomness. Eventually p<1/2.

For a progression having at least gamma k occurrences of each outer bit, and a fixed target bit, at least gamma k positions must flip. Key multiplicity two requires at least gamma k/2 specified independent keys to equal one. Conflicting prescriptions give probability zero; otherwise the probability is at most p^(gamma k/2). There is no binomial subset count, because these keys are determined by the progression and target before sampling. There are fewer than q^(2D) nonzero-step cyclic progressions. Therefore the probability R_k that any such rich progression becomes monochromatic obeys

\[
 R_k\le2q^{2D}p^{\gamma k/2}
 \le\exp\{(2c-\gamma b/2)k\log k
              +2D\log(2k^2)+\log2\}=o(1),           \tag{7.1}
\]

because gamma b/2-2c=17/20000>0 and D=o(k).

For a fixed affine signature and target bit, monochromaticity prescribes a value to every key it contains. If consistent, there are at least k/2 distinct keys by Section 5; with u prescribed flips and s distinct keys its probability is p^u(1-p)^(s-u)<=(1-p)^(k/2). Two affine progressions with the same literal signature define the same event on the same variables. Thus the probability A_k that any affine progression becomes monochromatic satisfies

\[
 A_k\le2T_{aff}(1-p)^{k/2}
 \le2\max(1,T_{aff})\exp(-pk/2)
 =\exp(-pk/2+o(pk))=o(1),                           \tag{7.2}
\]

by (6.1). No independence between distinct progressions or signatures is needed in either union bound. Section 4 covers every progression by these two classes, possibly with overlap. Equations (7.1)--(7.2) show that the total failure probability is below one eventually. Hence a cyclic two-coloring exists on N=q^D>=k^(ck) with no nonzero-step monochromatic k-term progression.

## 8. Transfer to all numbers of colors

We recall the retained digit product to make the uniformity explicit. For any integer s>=1, color n in {0,...,N^s-1}, with base-N digits n_i, by the tuple (C(n_0),...,C(n_(s-1))). If a,a+d,...,a+(k-1)d in this interval were monochromatic with d>0, take the largest i with N^i dividing d. Such i satisfies 0<=i<s. Writing d=N^i e with N not dividing e gives the exact identity

\[
 \left\lfloor(a+jd)/N^i\right\rfloor
 =\left\lfloor a/N^i\right\rfloor+je.
\]

Reducing modulo N shows that the ith digits form a nonzero-step cyclic progression, on which C cannot be constant. This is a contradiction. Translation by one gives W_(2^s)(k)>N^s. For r>=2 take s=floor(log2 r) and embed the 2^s color tuples in r colors. Consequently

\[
 W_r(k)>N^{\lfloor\log_2r\rfloor}
 \ge k^{(1/25)k\lfloor\log_2r\rfloor}.
\]

All preceding thresholds were chosen for fixed numerical parameters before r was introduced. The estimates for q use only k<P<=2k², and every dependence in the mesh, counts and probabilities is uniform over those primes. Taking the maximum of finitely many sufficiently-large-k thresholds proves existence of one K independent of r. No explicit practical finite threshold is claimed.

## 9. Exact margins and verification scope

The decisive rational margins are:

| Required strict inequality | Positive difference |
|---|---:|
| m>2a (global entropy against M) | 1/12 |
| 1-b>2a (affine global entropy) | 1/300 |
| a>b (affine q entropy) | 1/300 |
| gamma b/2>2c (rich-event coefficient) | 17/20000 |
| theta(1-nu)(1-2delta)>gamma | 551199/100000000 |
| nu>4/1000 (regularity reserve) | 1/1000 |
| m<1 (return blocks sublinear) | 1/4 |
| a<1 (prime cutoff and q rounding) | 2/3 |
| m>a (cutoff below 2M) | 5/12 |

For the local lemma, tau=1/2000000 and the tail-rate reserve 2epsilon²-3tau/2=5tau/2=1/800000 is positive. The incident-weight bound tau/4 is reached asymptotically by (3.2)--(3.3). The nonregular bound has reserve nu-1/250=1/1000, reached once (log k)^(-2)<1/1000. All remaining geometric small terms have the negative powers listed in Section 4. Ceilings only change constant factors, not any of these strict gaps.

Executable checks in solution/check.py verify these exact rational margins, reject boundary controls, exercise the row decomposition and complementary-block bookkeeping, compare finite scalar arrangement counts, check the norm-band obstruction, and check dilation denominators on finite cases. They supplement the displayed proof and do not verify imported geometric inputs or constitute a proof of an asymptotic theorem. The unmodified baseline audit is run separately. No baseline file is changed.
