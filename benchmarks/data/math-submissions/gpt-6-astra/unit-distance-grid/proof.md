# A conditional 8-by-4 final witness

## Claim and scope

Under the same upstream generic-grid and private-valuation hypotheses used in the baseline, **eight left points and four right anchors suffice for the full rank-two final algebraic contradiction**. The selected-edge metric is therefore **32**. The saving is one coefficient-field generator: after the baseline affine gauge, projectivize the remaining homogeneous right-hand data.

This is a conditional reduction of the final witness. It neither proves the imported generic-grid or private-valuation lemmas nor improves the headline unit-distance exponent. The pinned upstream source is not present locally; its hypotheses are retained as dependencies, not claimed as independently checked. The original `proof.tex`, `README.md`, and tests are left unchanged.

## 1. Hypotheses and precisely where they enter

Work in an algebraically closed ambient field Omega of characteristic zero. The starting system is the baseline system

    2(p_i-r_l) . (H_i-U_l) = R_il (A_i-B_l),     A_i-B_l != 0,

for i=1,...,8 and l=1,...,4. Here positions and velocities have two coordinates, the scalar differences are all nonzero, and R_il is a nonzero square root of the upstream radicand D_il.

Retain the baseline upstream hypotheses in the following operational form.

* **Generic grid.** After adjoining the four anchor positions to the upstream base B_0 and taking relative algebraic closure, obtain an algebraically closed field F_0. The left locus X is an irreducible variety over F_0 of dimension d in {1,2}. The eight left points lie on X and are jointly generic: trdeg over F_0 of F_0(p_1,...,p_8) is 8d. The four anchors are distinct. For v=r_2-r_1, v.v is nonzero, as in the baseline real finite-index argument.
* **Private valuations after coefficient enlargement.** For a left point p surviving as generic over the finitely enlarged coefficient field K used below, the source's private-valuation construction applies over F=K(p). For each l it supplies a discrete valuation nu_l of F with nu_l(D_l) odd and nu_l(D_m) even for m != l. Here D_l=D_il for the selected point. This is the same after-enlargement private-valuation input invoked in the baseline proof; it is not an assertion that adjoining arbitrary elements preserves square classes.

The enlargement below uses only rational expressions in the baseline normalized right-hand data and anchor positions. In particular, any upstream requirement that these data lie in the permitted coefficient/negligible-height field is preserved by field operations. No entire negligible-height field is adjoined. We do not add a new valuation hypothesis concerning a scale or left velocities. If the upstream private-valuation input were unavailable for such finite rational-data enlargements, the result would remain conditional on precisely that input, just as the baseline does.

## 2. The baseline gauge, with all parameters accounted for

Let J(x,y)=(-y,x), v=r_2-r_1, and

    omega = ((U_2-U_1).Jv)/(v.v).

For every left point and anchor set

    Hbar_i = H_i-U_1-omega J(p_i-r_1),
    Ubar_l = U_l-U_1-omega J(r_l-r_1),
    Abar_i = A_i-B_1,
    Bbar_l = B_l-B_1.

Since x.Jx=0, subtracting the common translation and infinitesimal rotation preserves the displayed equations. Subtracting B_1 preserves scalar differences. Thus

    2(p_i-r_l).(Hbar_i-Ubar_l) = R_il(Abar_i-Bbar_l),
    Abar_i-Bbar_l != 0.

Moreover Ubar_1=0, Bbar_1=0, and Ubar_2=tau v, where

    tau = ((U_2-U_1).v)/(v.v).

The identity Ubar_2=tau v follows by decomposing U_2-U_1 in the orthogonal basis v,Jv; its determinant v.v is nonzero. Anchor coordinates already belong to F_0. Consequently all normalized right-hand data are represented by these eight scalars:

    z = (tau, Ubar_3x, Ubar_3y, Ubar_4x, Ubar_4y,
         Bbar_2, Bbar_3, Bbar_4).

There are five velocity parameters and three logarithmic parameters. The removed translation, rotation, and scalar offset are not adjoined to the new coefficient field.

## 3. Projectivization saves one generator, including the exceptional case

Suppose first z is nonzero. Choose any k with z_k != 0 and put q=z_k. Divide **all** the normalized velocities and logarithmic scalars, on **both** sides of the grid, by q:

    h_i=Hbar_i/q,  V_l=Ubar_l/q,
    a_i=Abar_i/q,  b_l=Bbar_l/q.

The system is homogeneous in these data, so

    2(p_i-r_l).(h_i-V_l)=R_il(a_i-b_l),
    a_i-b_l=(A_i-B_l)/q != 0.                     (1)

The radicals and positions are not scaled. Define

    K = acl_Omega(F_0(z_j/q : j != k)).           (2)

The omitted ratio z_k/q is 1. Every V_l coordinate and b_l lies in K: V_1=0, b_1=0, V_2=(tau/q)v, and the remaining entries are the displayed ratios. Thus K costs at most **seven** generators over F_0.

The scale q is deliberately **not** adjoined. It need not be recoverable from the ratios, and h_i,a_i need not initially belong to K. They are unknowns in Omega; Cramer's rule below places the selected unknowns in the appropriate radical field. There is no requirement to reconstruct the original velocities, or to realize the normalized data as a new derivation. These points prevent mistakenly counting q as an eighth generator. Selecting one of eight charts is a finite case distinction, not an additional transcendental generator. The argument works in every chart, including charts with a logarithmic parameter as pivot.

If z=0, take q=1, h_i=Hbar_i, a_i=Abar_i, V_l=0, b_l=0, and K=F_0. Equation (1) and its nonzero differences still hold. This case costs zero generators and is not discarded. In either case

    trdeg(K/F_0) <= 7.                           (3)

## 4. Eight left points leave a generic survivor

Let P denote the tuple of all eight left points. The tower formula and (3) give

    trdeg_K K(P)
      = trdeg_F_0 K(P) - trdeg_F_0 K
      >= 8d-7.

Taking relative algebraic closure in (2) does not change transcendence degree. Since F_0 is algebraically closed, X remains irreducible of dimension d on extension to K. If every p_i failed to be generic on X over K, then trdeg_K K(p_i) <= d-1 for all i. Subadditivity would imply

    trdeg_K K(P) <= 8(d-1)=8d-8,

a contradiction. Hence some p is generic over K. This works for d=1 as well as d=2; positional rank two is not being confused with dimension two. Also p differs from every anchor, since anchor coordinates are in K whereas trdeg_K K(p)=d>0.

The strict inequality is the improvement over the baseline: with eight unprojectivized generators, eight points give equality and cannot prove survival. Seven points with the new seven-generator bound would again give equality; no claim of such a further uniform saving is made.

## 5. Radical independence, not merely distinct radicands

Write F=K(p), D_l=R_l^2, and apply the imported private-valuation hypothesis over this very field. For any nonempty subset S of {1,2,3,4}, choose j in S. Then

    nu_j(product_{l in S} D_l)

is odd: its j term is odd and all other terms are even. A square in F has even valuation. Thus no nonempty product of the D_l is a square in F. Their classes in F*/F*^2 are independent.

For completeness, this implies that E=F(R_1,R_2,R_3,R_4) has degree 16 and basis

    R_S = product_{l in S} R_l,   S subset {1,2,3,4},

and admits all independent sign changes. Here is a proof that does not assume that conclusion. Inductively suppose the first n-1 radicals give that basis and sign automorphisms. If D_n were a square x^2 in their field, every sign automorphism sigma would satisfy sigma(x)=+x or -x. Thus x is a simultaneous eigenvector for the sign changes. Expanding x in the inductive basis, distinct monomials have distinct sign characters, so only one basis coefficient can be nonzero: x=c R_S for some c in F. Then D_n=c^2 product_{l in S}D_l, contradicting square-class independence. Therefore adjoining R_n is genuinely quadratic. The old sign changes extend by fixing R_n, and the new sign change fixes the previous field; the basis and automorphism assertions follow by induction. In particular R_1,R_2,R_3 are linearly independent over F, and flipping only R_4 fixes F(R_1,R_2,R_3).

No inference here assumes that independence automatically survives arbitrary coefficient extension. The private valuations must be supplied after the enlargement, exactly as stated in Section 1. Conversely, the projective normalization never adjoins an extra radical or the scale q.

## 6. Full rank-two contradiction: three equations and a fourth sign

Let d_l=p-r_l=(x_l,y_l) in F^2. Suppose these four positional rows have rank two. Select an independent pair and relabel it as rows 1,2; select any other row as 3 and leave the fourth as 4. Relabeling for this argument does not change K or undo the earlier gauge. Define Delta_ij=x_i y_j-y_i x_j, so Delta_12 != 0.

For the selected point write h=(h_x,h_y), a=a_i. The first three equations (1) are M(h_x,h_y,a)^T=c, where

    M = [[2x_1, 2y_1, -R_1],
         [2x_2, 2y_2, -R_2],
         [2x_3, 2y_3, -R_3]],
    c_l = 2 d_l.V_l - R_l b_l.

Direct expansion gives

    det M = -4(Delta_23 R_1 - Delta_13 R_2 + Delta_12 R_3).

Its R_3 coefficient is nonzero, and the three single-radical monomials are independent over F. Hence det M != 0. This justifies Cramer's rule even if some of the other minors vanish; no assumption that every triple of anchors is noncollinear is needed. All matrix and right-hand entries belong to E_3=F(R_1,R_2,R_3), because V_l,b_l belong to K. Thus h_x,h_y,a belong to E_3.

The fourth equation is

    2 d_4.(h-V_4)=R_4(a-b_4).

Apply the E_3-automorphism of E that sends R_4 to -R_4. The left side and a-b_4 are fixed. Subtracting the equations yields

    2 R_4(a-b_4)=0.

Since the characteristic is zero and R_4 != 0, this forces a=b_4, contrary to (1). Applying the automorphism requires no extension to all of Omega: after Cramer's rule every quantity in the fourth equality already lies in E.

In the zero-right-data case c=0, invertibility even gives h_x=h_y=a=0 immediately, contradicting a-b_l != 0 with b_l=0. Thus the exceptional chart is fully covered, rather than presumed impossible.

## 7. Rank-one branch for completeness

If the four rows have rank one, all anchors and p lie on their common affine line. Put v=r_2-r_1 in K^2 and write d_l=s_l v with s_l in F and s_l != 0. Set h=v.h_i and u_l=v.V_l in K. Then (1) becomes

    2s_l h - R_l a = 2s_l u_l - R_l b_l.

The first two equations have determinant 2(s_2 R_1-s_1 R_2), nonzero by the same single-radical independence. Consequently h,a lie in F(R_1,R_2). Flipping only R_3 in the third equation gives 2R_3(a-b_3)=0, again a contradiction. If all right data vanish the first two equations already force a=0. Rank zero is impossible because the anchors are distinct. This branch does not supply the claimed improvement on its own: Section 6 explicitly establishes the required full rank-two case with all four anchors.

## 8. Generator and edge accounting

For four anchors the baseline gauge leaves 1+2(4-2)+(4-1)=8 parameters. On every nonzero projective chart their ratios cost at most 8-1=7; on the zero chart the cost is zero. Eight jointly generic points then leave one generic point. Four anchors supply three determining equations and one independent radical sign change in positional rank two. The full witness is therefore 8 times 4 = **32 selected edges**, versus the baseline 36.

No cost is hidden in anchor coordinates (already in F_0), algebraic closure (zero transcendence cost), choice of chart or row relabeling (finite choices), or the unknown left velocities (never adjoined). The scale q, the removed gauge parameters, and the entire upstream negligible-height field are excluded from K.

## 9. Verification and limitations

The companion executable checks verify the exact gauge and scaling identities, all eight projective charts and their Jacobian ranks, the zero-data case, the transcendence-degree budgets, the full symbolic 3-by-3 determinant and adjugate/Cramer identities, fourth-sign subtraction, and the rank-one identities. A rational multiquadratic fixture checks the 16 basis monomials and sign automorphisms exactly. Deliberate controls reject wrong signs, one-sided scaling, zero pivots, dependent radical classes, a missing fourth radical, and a bad choice of determining rows. These controls also show why a formal sign substitution without square-class independence is insufficient.

These are checks of the displayed algebra, not a numerical proof and not verification of the imported upstream lemmas. The proof is Sections 2–8 under the explicitly retained hypotheses in Section 1. The original baseline test runner is executed unchanged as part of verification, and its input files are checked against a preservation manifest.
