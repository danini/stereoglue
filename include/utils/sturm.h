// Copyright (c) 2020, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>
#include <math.h>
#include <stdio.h>

#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcount __popcnt
#endif

namespace stereoglue {
namespace sturm {

// Constructs the quotients needed for evaluating the sturm sequence.
template <int N> void build_sturm_seq(const double *fvec, double *svec) {

    double f[3 * N];
    double *f1 = f;
    double *f2 = f1 + N + 1;
    double *f3 = f2 + N;

    std::copy(fvec, fvec + (2 * N + 1), f);

    for (int i = 0; i < N - 1; ++i) {
        const double q1 = f1[N - i] * f2[N - 1 - i];
        const double q0 = f1[N - 1 - i] * f2[N - 1 - i] - f1[N - i] * f2[N - 2 - i];

        f3[0] = f1[0] - q0 * f2[0];
        for (int j = 1; j < N - 1 - i; ++j) {
            f3[j] = f1[j] - q1 * f2[j - 1] - q0 * f2[j];
        }
        const double c = -std::abs(f3[N - 2 - i]);
        const double ci = 1.0 / c;
        for (int j = 0; j < N - 1 - i; ++j) {
            f3[j] = f3[j] * ci;
        }

        // juggle pointers (f1,f2,f3) -> (f2,f3,f1)
        double *tmp = f1;
        f1 = f2;
        f2 = f3;
        f3 = tmp;

        svec[3 * i] = q0;
        svec[3 * i + 1] = q1;
        svec[3 * i + 2] = c;
    }

    svec[3 * N - 3] = f1[0];
    svec[3 * N - 2] = f1[1];
    svec[3 * N - 1] = f2[0];
}

// Evaluates polynomial using Horner's method.
// Assumes that f[N] = 1.0
template <int N> inline double polyval(const double *f, double x) {
    double fx = x + f[N - 1];
    for (int i = N - 2; i >= 0; --i) {
        fx = x * fx + f[i];
    }
    return fx;
}

// Daniel Thul is responsible for this template-trickery :)
template <int D> inline unsigned int flag_negative(const double *const f) {
    return ((f[D] < 0) << D) | flag_negative<D - 1>(f);
}
template <> inline unsigned int flag_negative<0>(const double *const f) { return f[0] < 0; }
// Evaluates the sturm sequence and counts the number of sign changes
template <int N, typename std::enable_if<(N < 32), void>::type * = nullptr>
inline int signchanges(const double *svec, double x) {

    double f[N + 1];
    f[N] = svec[3 * N - 1];
    f[N - 1] = svec[3 * N - 3] + x * svec[3 * N - 2];

    for (int i = N - 2; i >= 0; --i) {
        f[i] = (svec[3 * i] + x * svec[3 * i + 1]) * f[i + 1] + svec[3 * i + 2] * f[i + 2];
    }

    // In testing this turned out to be slightly faster compared to a naive loop
    unsigned int S = flag_negative<N>(f);

    return __builtin_popcount((S ^ (S >> 1)) & ~(0xFFFFFFFF << N));
}

template <int N, typename std::enable_if<(N >= 32), void>::type * = nullptr>
inline int signchanges(const double *svec, double x) {

    double f[N + 1];
    f[N] = svec[3 * N - 1];
    f[N - 1] = svec[3 * N - 3] + x * svec[3 * N - 2];

    for (int i = N - 2; i >= 0; --i) {
        f[i] = (svec[3 * i] + x * svec[3 * i + 1]) * f[i + 1] + svec[3 * i + 2] * f[i + 2];
    }

    int count = 0;
    bool neg1 = f[0] < 0;
    for (int i = 0; i < N; ++i) {
        bool neg2 = f[i + 1] < 0;
        if (neg1 ^ neg2) {
            ++count;
        }
        neg1 = neg2;
    }
    return count;
}

// Computes the Cauchy bound on the real roots.
// Experiments with more complicated (expensive) bounds did not seem to have a good trade-off.
template <int N> inline double get_bounds(const double *fvec) {
    double max = 0;
    for (int i = 0; i < N; ++i) {
        max = std::max(max, std::abs(fvec[i]));
    }
    return 1.0 + max;
}

// Applies Ridder's bracketing method until we get close to root, followed by newton iterations
template <int N>
void ridders_method_newton(const double *fvec, double a, double b, double *roots, int &n_roots, double tol) {
    double fa = polyval<N>(fvec, a);
    double fb = polyval<N>(fvec, b);

    if (!((fa < 0) ^ (fb < 0)))
        return;

    const double tol_newton = 1e-3;

    for (int iter = 0; iter < 30; ++iter) {
        if (std::abs(a - b) < tol_newton) {
            break;
        }
        const double c = (a + b) * 0.5;
        const double fc = polyval<N>(fvec, c);
        const double s = std::sqrt(fc * fc - fa * fb);
        if (!s)
            break;
        const double d = (fa < fb) ? c + (a - c) * fc / s : c + (c - a) * fc / s;
        const double fd = polyval<N>(fvec, d);

        if (fd >= 0 ? (fc < 0) : (fc > 0)) {
            a = c;
            fa = fc;
            b = d;
            fb = fd;
        } else if (fd >= 0 ? (fa < 0) : (fa > 0)) {
            b = d;
            fb = fd;
        } else {
            a = d;
            fa = fd;
        }
    }

    // We switch to Newton's method once we are close to the root
    double x = (a + b) * 0.5;

    double fx, fpx, dx;
    const double *fpvec = fvec + N + 1;
    for (int iter = 0; iter < 10; ++iter) {
        fx = polyval<N>(fvec, x);
        if (std::abs(fx) < tol) {
            break;
        }
        fpx = static_cast<double>(N) * polyval<N - 1>(fpvec, x);
        dx = fx / fpx;
        x = x - dx;
        if (std::abs(dx) < tol) {
            break;
        }
    }

    roots[n_roots++] = x;
}

template <int N>
void isolate_roots(const double *fvec, const double *svec, double a, double b, int sa, int sb, double *roots,
                   int &n_roots, double tol, int depth) {
    if (depth > 300)
        return;

    int n_rts = sa - sb;

    if (n_rts > 1) {
        double c = (a + b) * 0.5;
        int sc = signchanges<N>(svec, c);
        isolate_roots<N>(fvec, svec, a, c, sa, sc, roots, n_roots, tol, depth + 1);
        isolate_roots<N>(fvec, svec, c, b, sc, sb, roots, n_roots, tol, depth + 1);
    } else if (n_rts == 1) {
        ridders_method_newton<N>(fvec, a, b, roots, n_roots, tol);
    }
}

template <int N> inline int bisect_sturm(const double *coeffs, double *roots, double tol = 1e-10) {
    if (coeffs[N] == 0.0)
        return 0; // return bisect_sturm<N-1>(coeffs,roots,tol); // This explodes compile times...

    double fvec[2 * N + 1];
    double svec[3 * N];

    // fvec is the polynomial and its first derivative.
    std::copy(coeffs, coeffs + N + 1, fvec);

    // Normalize w.r.t. leading coeff
    double c_inv = 1.0 / fvec[N];
    for (int i = 0; i < N; ++i)
        fvec[i] *= c_inv;
    fvec[N] = 1.0;

    // Compute the derivative with normalized coefficients
    for (int i = 0; i < N - 1; ++i) {
        fvec[N + 1 + i] = fvec[i + 1] * ((i + 1) / static_cast<double>(N));
    }
    fvec[2 * N] = 1.0;

    // Compute sturm sequences
    build_sturm_seq<N>(fvec, svec);

    // All real roots are in the interval [-r0, r0]
    double r0 = get_bounds<N>(fvec);
    double a = -r0;
    double b = r0;

    int sa = signchanges<N>(svec, a);
    int sb = signchanges<N>(svec, b);

    int n_roots = sa - sb;
    if (n_roots == 0)
        return 0;

    n_roots = 0;
    isolate_roots<N>(fvec, svec, a, b, sa, sb, roots, n_roots, tol, 0);

    return n_roots;
}

template <> inline int bisect_sturm<1>(const double *coeffs, double *roots, double tol) {
    if (coeffs[1] == 0.0) {
        return 0;
    } else {
        roots[0] = -coeffs[0] / coeffs[1];
        return 1;
    }
}

template <> inline int bisect_sturm<0>(const double *coeffs, double *roots, double tol) { return 0; }

template <typename Derived> void charpoly_danilevsky_piv(Eigen::MatrixBase<Derived> &A, double *p) {
    int n = A.rows();

    for (int i = n - 1; i > 0; i--) {

        int piv_ind = i - 1;
        double piv = std::abs(A(i, i - 1));

        // Find largest pivot
        for (int j = 0; j < i - 1; j++) {
            if (std::abs(A(i, j)) > piv) {
                piv = std::abs(A(i, j));
                piv_ind = j;
            }
        }
        if (piv_ind != i - 1) {
            // Perform permutation
            A.row(i - 1).swap(A.row(piv_ind));
            A.col(i - 1).swap(A.col(piv_ind));
        }
        piv = A(i, i - 1);

        Eigen::VectorXd v = A.row(i);
        A.row(i - 1) = v.transpose() * A;

        Eigen::VectorXd vinv = (-1.0) * v;
        vinv(i - 1) = 1;
        vinv /= piv;
        vinv(i - 1) -= 1;
        Eigen::VectorXd Acol = A.col(i - 1);
        for (int j = 0; j <= i; j++)
            A.row(j) = A.row(j) + Acol(j) * vinv.transpose();

        A.row(i).setZero();
        A(i, i - 1) = 1;
    }
    p[n] = 1;
    for (int i = 0; i < n; i++)
        p[i] = -A(0, n - i - 1);
}

#define RELERROR      1.0e-12   /* smallest relative error we want */
//#define MAXPOW        0        /* max power of 10 we wish to search to */
#define MAXIT         800       /* max number of iterations */
#define SMALL_ENOUGH  1.0e-12   /* a coefficient smaller than SMALL_ENOUGH 
* is considered to be zero (0.0). */
#ifndef MAX_DEG
#define MAX_DEG      64
#endif

/* structure type for representing a polynomial */
typedef struct p {
	int ord;
	double coef[MAX_DEG + 1];
} poly;

/*---------------------------------------------------------------------------
* evalpoly
*
* evaluate polynomial defined in coef returning its value.
*--------------------------------------------------------------------------*/

double evalpoly(int ord, double *coef, double x)
{
	double *fp = &coef[ord];
	double f = *fp;

	for (fp--; fp >= coef; fp--)
		f = x * f + *fp;

	return(f);
}

int modrf_pos(int ord, double *coef, double a, double b,
	double *val, int invert)
{
	int  its;
	double fx, lfx;
	double *fp;
	double *scoef = coef;
	double *ecoef = &coef[ord];
	double fa, fb;

	// Invert the interval if required
	if (invert)
	{
		double temp = a;
		a = 1.0 / b;
		b = 1.0 / temp;
	}

	// Evaluate the polynomial at the end points
	if (invert)
	{
		fb = fa = *scoef;
		for (fp = scoef + 1; fp <= ecoef; fp++)
		{
			fa = a * fa + *fp;
			fb = b * fb + *fp;
		}
	}
	else
	{
		fb = fa = *ecoef;
		for (fp = ecoef - 1; fp >= scoef; fp--)
		{
			fa = a * fa + *fp;
			fb = b * fb + *fp;
		}
	}

	// if there is no sign difference the method won't work
	if (fa * fb > 0.0)
		return(0);

	// Return if the values are close to zero already
	if (fabs(fa) < RELERROR)
	{
		*val = invert ? 1.0 / a : a;
		return(1);
	}

	if (fabs(fb) < RELERROR)
	{
		*val = invert ? 1.0 / b : b;
		return(1);
	}

	lfx = fa;

	for (its = 0; its < MAXIT; its++)
	{
		// Assuming straight line from a to b, find zero
		double x = (fb * a - fa * b) / (fb - fa);

		// Evaluate the polynomial at x
		if (invert)
		{
			fx = *scoef;
			for (fp = scoef + 1; fp <= ecoef; fp++)
				fx = x * fx + *fp;
		}
		else
		{
			fx = *ecoef;
			for (fp = ecoef - 1; fp >= scoef; fp--)
				fx = x * fx + *fp;
		}

		// Evaluate two stopping conditions
		if (fabs(x) > RELERROR && fabs(fx / x) < RELERROR)
		{
			*val = invert ? 1.0 / x : x;
			return(1);
		}
		else if (fabs(fx) < RELERROR)
		{
			*val = invert ? 1.0 / x : x;
			return(1);
		}

		// Subdivide region, depending on whether fx has same sign as fa or fb
		if ((fa * fx) < 0)
		{
			b = x;
			fb = fx;
			if ((lfx * fx) > 0)
				fa /= 2;
		}
		else
		{
			a = x;
			fa = fx;
			if ((lfx * fx) > 0)
				fb /= 2;
		}


		// Return if the difference between a and b is very small
		if (fabs(b - a) < fabs(RELERROR * a))
		{
			*val = invert ? 1.0 / a : a;
			return(1);
		}

		lfx = fx;
	}

	//==================================================================
	// This is debugging in case something goes wrong.
	// If we reach here, we have not converged -- give some diagnostics
	//==================================================================

	/*fprintf(stderr, "modrf overflow on interval %f %f\n", a, b);
	fprintf(stderr, "\t b-a = %12.5e\n", b - a);
	fprintf(stderr, "\t fa  = %12.5e\n", fa);
	fprintf(stderr, "\t fb  = %12.5e\n", fb);
	fprintf(stderr, "\t fx  = %12.5e\n", fx);*/

	// Evaluate the true values at a and b
	if (invert)
	{
		fb = fa = *scoef;
		for (fp = scoef + 1; fp <= ecoef; fp++)
		{
			fa = a * fa + *fp;
			fb = b * fb + *fp;
		}
	}
	else
	{
		fb = fa = *ecoef;
		for (fp = ecoef - 1; fp >= scoef; fp--)
		{
			fa = a * fa + *fp;
			fb = b * fb + *fp;
		}
	}

	/*fprintf(stderr, "\t true fa = %12.5e\n", fa);
	fprintf(stderr, "\t true fb = %12.5e\n", fb);
	fprintf(stderr, "\t gradient= %12.5e\n", (fb - fa) / (b - a));

	// Print out the polynomial
	fprintf(stderr, "Polynomial coefficients\n");
	for (fp = ecoef; fp >= scoef; fp--)
		fprintf(stderr, "\t%12.5e\n", *fp);*/

	return(0);
}

/*---------------------------------------------------------------------------
* modrf
*
* uses the modified regula-falsi method to evaluate the root
* in interval [a,b] of the polynomial described in coef. The
* root is returned is returned in *val. The routine returns zero
* if it can't converge.
*--------------------------------------------------------------------------*/

int modrf(int ord, double *coef, double a, double b, double *val)
{
	// This is an interfact to modrf that takes account of different cases
	// The idea is that the basic routine works badly for polynomials on
	// intervals that extend well beyond [-1, 1], because numbers get too large

	double *fp;
	double *scoef = coef;
	double *ecoef = &coef[ord];
	const int invert = 1;

	double fp1 = 0.0, fm1 = 0.0; // Values of function at 1 and -1
	double fa = 0.0, fb = 0.0; // Values at end points

	// We assume that a < b
	if (a > b)
	{
		double temp = a;
		a = b;
		b = temp;
	}

	// The normal case, interval is inside [-1, 1]
	if (b <= 1.0 && a >= -1.0) return modrf_pos(ord, coef, a, b, val, !invert);

	// The case where the interval is outside [-1, 1]
	if (a >= 1.0 || b <= -1.0)
		return modrf_pos(ord, coef, a, b, val, invert);

	// If we have got here, then the interval includes the points 1 or -1.
	// In this case, we need to evaluate at these points

	// Evaluate the polynomial at the end points
	for (fp = ecoef - 1; fp >= scoef; fp--)
	{
		fp1 = *fp + fp1;
		fm1 = *fp - fm1;
		fa = a * fa + *fp;
		fb = b * fb + *fp;
	}

	// Then there is the case where the interval contains -1 or 1
	if (a < -1.0 && b > 1.0)
	{
		// Interval crosses over 1.0, so cut
		if (fa * fm1 < 0.0)      // The solution is between a and -1
			return modrf_pos(ord, coef, a, -1.0, val, invert);
		else if (fb * fp1 < 0.0) // The solution is between 1 and b
			return modrf_pos(ord, coef, 1.0, b, val, invert);
		else                     // The solution is between -1 and 1
			return modrf_pos(ord, coef, -1.0, 1.0, val, !invert);
	}
	else if (a < -1.0)
	{
		// Interval crosses over 1.0, so cut
		if (fa * fm1 < 0.0)      // The solution is between a and -1
			return modrf_pos(ord, coef, a, -1.0, val, invert);
		else                     // The solution is between -1 and b
			return modrf_pos(ord, coef, -1.0, b, val, !invert);
	}
	else  // b > 1.0
	{
		if (fb * fp1 < 0.0) // The solution is between 1 and b
			return modrf_pos(ord, coef, 1.0, b, val, invert);
		else                     // The solution is between a and 1
			return modrf_pos(ord, coef, a, 1.0, val, !invert);
	}
}

/*---------------------------------------------------------------------------
* modp
*
*  calculates the modulus of u(x) / v(x) leaving it in r, it
*  returns 0 if r(x) is a constant.
*  note: this function assumes the leading coefficient of v is 1 or -1
*--------------------------------------------------------------------------*/

static int modp(poly *u, poly *v, poly *r)
{
	int j, k;  /* Loop indices */

	double *nr = r->coef;
	double *end = &u->coef[u->ord];

	double *uc = u->coef;
	while (uc <= end)
		*nr++ = *uc++;

	if (v->coef[v->ord] < 0.0) {

		for (k = u->ord - v->ord - 1; k >= 0; k -= 2)
			r->coef[k] = -r->coef[k];

		for (k = u->ord - v->ord; k >= 0; k--)
			for (j = v->ord + k - 1; j >= k; j--)
				r->coef[j] = -r->coef[j] - r->coef[v->ord + k]
				* v->coef[j - k];
	}
	else {
		for (k = u->ord - v->ord; k >= 0; k--)
			for (j = v->ord + k - 1; j >= k; j--)
				r->coef[j] -= r->coef[v->ord + k] * v->coef[j - k];
	}

	k = v->ord - 1;
	while (k >= 0 && fabs(r->coef[k]) < SMALL_ENOUGH) {
		r->coef[k] = 0.0;
		k--;
	}

	r->ord = (k < 0) ? 0 : k;

	return(r->ord);
}

/*---------------------------------------------------------------------------
* buildsturm
*
* build up a sturm sequence for a polynomial in smat, returning
* the number of polynomials in the sequence
*--------------------------------------------------------------------------*/

int buildsturm(int ord, poly *sseq)
{
	sseq[0].ord = ord;
	sseq[1].ord = ord - 1;

	/* calculate the derivative and normalise the leading coefficient */
	{
		int i;    // Loop index
		poly *sp;
		double f = fabs(sseq[0].coef[ord] * ord);
		double *fp = sseq[1].coef;
		double *fc = sseq[0].coef + 1;

		for (i = 1; i <= ord; i++)
			*fp++ = *fc++ * i / f;

		/* construct the rest of the Sturm sequence */
		for (sp = sseq + 2; modp(sp - 2, sp - 1, sp); sp++) {

			/* reverse the sign and normalise */
			f = -fabs(sp->coef[sp->ord]);
			for (fp = &sp->coef[sp->ord]; fp >= sp->coef; fp--)
				*fp /= f;
		}

		sp->coef[0] = -sp->coef[0]; /* reverse the sign */

		return(sp - sseq);
	}
}

/*---------------------------------------------------------------------------
* numchanges
*
* return the number of sign changes in the Sturm sequence in
* sseq at the value a.
*--------------------------------------------------------------------------*/

int numchanges(int np, poly *sseq, double a)
{
	int changes = 0;

	double lf = evalpoly(sseq[0].ord, sseq[0].coef, a);

	poly *s;
	for (s = sseq + 1; s <= sseq + np; s++) {
		double f = evalpoly(s->ord, s->coef, a);
		if (lf == 0.0 || lf * f < 0)
			changes++;
		lf = f;
	}

	return(changes);
}

/*---------------------------------------------------------------------------
* numroots
*
* return the number of distinct real roots of the polynomial described in sseq.
*--------------------------------------------------------------------------*/

int numroots(int np, poly *sseq, int *atneg, int *atpos, bool non_neg)
{
	int atposinf = 0;
	int atneginf = 0;

	/* changes at positive infinity */
	double f;
	double lf = sseq[0].coef[sseq[0].ord];

	poly *s;
	for (s = sseq + 1; s <= sseq + np; s++) {
		f = s->coef[s->ord];
		if (lf == 0.0 || lf * f < 0)
			atposinf++;
		lf = f;
	}

	// changes at negative infinity or zero
	if (non_neg)
		atneginf = numchanges(np, sseq, 0.0);

	else
	{
		if (sseq[0].ord & 1)
			lf = -sseq[0].coef[sseq[0].ord];
		else
			lf = sseq[0].coef[sseq[0].ord];

		for (s = sseq + 1; s <= sseq + np; s++) {
			if (s->ord & 1)
				f = -s->coef[s->ord];
			else
				f = s->coef[s->ord];
			if (lf == 0.0 || lf * f < 0)
				atneginf++;
			lf = f;
		}
	}

	*atneg = atneginf;
	*atpos = atposinf;

	return(atneginf - atposinf);
}


/*---------------------------------------------------------------------------
* sbisect
*
* uses a bisection based on the sturm sequence for the polynomial
* described in sseq to isolate intervals in which roots occur,
* the roots are returned in the roots array in order of magnitude.
*--------------------------------------------------------------------------*/

int sbisect(int np, poly *sseq,
	double min, double max,
	int atmin, int atmax,
	double *roots)
{
	double mid;
	int atmid;
	int its;
	int  n1 = 0, n2 = 0;
	int nroot = atmin - atmax;

	if (nroot == 1) {

		/* first try a less expensive technique.  */
		if (modrf(sseq->ord, sseq->coef, min, max, &roots[0]))
			return 1;

		/*
		* if we get here we have to evaluate the root the hard
		* way by using the Sturm sequence.
		*/
		for (its = 0; its < MAXIT; its++) {
			mid = (double)((min + max) / 2);
			atmid = numchanges(np, sseq, mid);

			if (fabs(mid) > RELERROR) {
				if (fabs((max - min) / mid) < RELERROR) {
					roots[0] = mid;
					return 1;
				}
			}
			else if (fabs(max - min) < RELERROR) {
				roots[0] = mid;
				return 1;
			}

			if ((atmin - atmid) == 0)
				min = mid;
			else
				max = mid;
		}

		if (its == MAXIT) {
			/*fprintf(stderr, "sbisect: overflow min %f max %f\
							                         diff %e nroot %d n1 %d n2 %d\n",
													 min, max, max - min, nroot, n1, n2);*/
			roots[0] = mid;
		}

		return 1;
	}

	/* more than one root in the interval, we have to bisect */
	for (its = 0; its < MAXIT; its++) {

		mid = (double)((min + max) / 2);
		atmid = numchanges(np, sseq, mid);

		n1 = atmin - atmid;
		n2 = atmid - atmax;

		if (n1 != 0 && n2 != 0) {
			sbisect(np, sseq, min, mid, atmin, atmid, roots);
			sbisect(np, sseq, mid, max, atmid, atmax, &roots[n1]);
			break;
		}

		if (n1 == 0)
			min = mid;
		else
			max = mid;
	}

	if (its == MAXIT) {
		//fprintf(stderr, "sbisect: roots too close together\n");
		/*fprintf(stderr, "sbisect: overflow min %f max %f diff %e\
						                      nroot %d n1 %d n2 %d\n",
											  min, max, max - min, nroot, n1, n2);*/
		for (n1 = atmax; n1 < atmin; n1++)
			roots[n1 - atmax] = mid;
	}

	return 1;
}

int find_real_roots_sturm(
	double *p, int order, double *roots, int *nroots, int maxpow, bool non_neg)
{
	/*
	* finds the roots of the input polynomial.  They are returned in roots.
	* It is assumed that roots is already allocated with space for the roots.
	*/

	poly sseq[MAX_DEG + 1];
	double  min, max;
	int  i, nchanges, np, atmin, atmax;

	// Copy the coefficients from the input p.  Normalize as we go
	double norm = 1.0 / p[order];
	for (i = 0; i <= order; i++)
		sseq[0].coef[i] = p[i] * norm;

	// Now, also normalize the other terms
	double val0 = fabs(sseq[0].coef[0]);
	double fac = 1.0; // This will be a factor for the roots
	if (val0 > 10.0)  // Do this in case there are zero roots
	{
		fac = pow(val0, -1.0 / order);
		double mult = fac;
		for (int i = order - 1; i >= 0; i--)
		{
			sseq[0].coef[i] *= mult;
			mult = mult * fac;
		}
	}

	/* build the Sturm sequence */
	np = buildsturm(order, sseq);

#ifdef RH_DEBUG
	{
		int i, j;

		printf("Sturm sequence for:\n");
		for (i = order; i >= 0; i--)
			printf("%lf ", sseq[0].coef[i]);
		printf("\n\n");

		for (i = 0; i <= np; i++) {
			for (j = sseq[i].ord; j >= 0; j--)
				printf("%10f ", sseq[i].coef[j]);
			printf("\n");
		}

		printf("\n");
	}
#endif

	// get the number of real roots
	*nroots = numroots(np, sseq, &atmin, &atmax, non_neg);

	if (*nroots == 0) {
		// fprintf(stderr, "solve: no real roots\n");
		return 0;
	}

	/* calculate the bracket that the roots live in */
	if (non_neg) min = 0.0;
	else
	{
		min = -1.0;
		nchanges = numchanges(np, sseq, min);
		for (i = 0; nchanges != atmin && i != maxpow; i++) {
			min *= 10.0;
			nchanges = numchanges(np, sseq, min);
		}

		if (nchanges != atmin) {
			//printf("solve: unable to bracket all negative roots\n");
			atmin = nchanges;
		}
	}

	max = 1.0;
	nchanges = numchanges(np, sseq, max);
	for (i = 0; nchanges != atmax && i != maxpow; i++) {
		max *= 10.0;
		nchanges = numchanges(np, sseq, max);
	}

	if (nchanges != atmax) {
		//printf("solve: unable to bracket all positive roots\n");
		atmax = nchanges;
	}

	*nroots = atmin - atmax;

	/* perform the bisection */
	sbisect(np, sseq, min, max, atmin, atmax, roots);

	/* Finally, reorder the roots */
	for (i = 0; i<*nroots; i++)
		roots[i] /= fac;

#ifdef RH_DEBUG

	/* write out the roots */
	printf("Number of roots = %d\n", *nroots);
	for (i = 0; i<*nroots; i++)
		printf("%12.5e\n", roots[i]);

#endif

	return 1;
}

} // namespace sturm
} // namespace stereoglue