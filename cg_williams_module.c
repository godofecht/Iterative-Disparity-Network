/****************************************************************************
*
*   This is an implementation of Peter Williams' conjugate gradient method.
*   The procedure cg_williams requires 6 args.
*
*   1. w_orig: Original values of parameters to be adjusted so as
*        to minimise a function.
*   2. imin: Index of first parameter in w_orig (usually imin=0 or 1 ).
*   3. imax: Index of last parameter in w_orig.
*   4. func: Returns value of function to be minimised by cg_williams.
*   5. dfunc: This procedure returns the derivative of func with respect to
*           each of the parameters in w_orig.
*   6. finished: Returns TRUE if function deemed to be minimised, FALSE
*       otherwise.
*
*       Jim Stone,
*       Cognitive Sciences/Biology,
*       University of Sussex.
*       26/6/93
*
*****************************************************************************/


/*
 * Stephen Eglen Wed Nov 22 1995
 *
 * This has been changed from Jim's code in several ways.
 *
 * Firstly, all vectors are assumed to run from imin to imax, rather
 * than from 1 to len.  This makes it more portable - my arrays
 * started from 0, whereas I guess Jim's started from 1.
 *
 * Secondly, all explicit references in printf format strings have
 * been changed from "%Lf" to "%lf". This caused errors before.
 *
 * The function find_2nd_deriv(g0,g_try,s,sigma,imin, imax) has been
 * changed so that it now takes imin, imax instead of the len
 * parameter.
 *
 * One of the calls to applyfunction2 was wrong. (Pointer not needed)
 *
 * Fri Dec  8 1995
 *
 * Found bugs in header file for definitions of JMAX and FALSE
 *
 */

#include <math.h>
#include <stdio.h>
#include "cg_williams_module.h"

#define LAMBDA_MIN  1.0e-36
#define LAMBDA_MAX  1.0e+36

/* if set rho minmax to 0.25 and 0.75 alpha just gets smalller and smaller.*/
#define RHO_MIN 0.25
#define RHO_MAX 0.75

#define VERBOSE_TRAINING FALSE /* sje mod to TRUE */


#define SJE_SPATIAL		/* If this is defined, then some extra
				 * code will be included so that files
				 * can be written out every few
				 * epochs. */
#ifdef SJE_SPATIAL

extern FILE *opfp;
#include "dispnet.h"

extern Array z, zbar, ztilde;

#endif /* SJE_SPATIAL */

int
cg_williams(w_orig, imin, imax, func, dfunc, finished)
double *w_orig;         /* original weight vector */
int imin,imax;          /* 1ST AND LAST INDICES OF WT VECTOR */
double (*func)();       /* PTR TO FUNC THAT RETURNS VALUE OF FUNCTION (E) */
void (*dfunc)();        /* PTR TO FUNC THAT FILLS ARG WITH PARTIALS VECTOR
(@E/@W = GRAD W(E)) */
char (*finished)();
{
    /*---------------------
    TRAINING PARAMETERS
    -----------------------*/
    double                pi = 0.1;
    double                lambda=1.0;     /* TRUST REGION SIZE */
    double                epsilon=0.001;
    double                sigma;          /* F(W0) AND F(W0+S*SIGMA) USED TO EST 2ND DERIV */
    double                rho;
    double                beta;           /* USED TO MAKE NEW CONJ DIR */
    double                alpha;
    double                mu;
    double                kappa;
    double                cg_gamma;
    double                delta;

    /*----------
    VECTORS
    -----------*/
    double                *w0, *w1;             /* CURRENT AND NEW WEIGHT VEC */
    double                *g0, *g1;             /* DIRECTION OF STEEPEST DESCENT */

    double                *w_try, *g_try;   /* w_try USED TO EST 2ND DERIV */
    double                *s;                       /* SEARCH DIRECTION */
    double                *temp_vec;

    /* MISC */
    double                  E0, E1;
    double                  temp;
    double                *ptr;
    int                 S, Smax;            /* MAX NUMBER OF LINE SEARCHES BEFORE RE-INIT SEARCH DIR */
    int                 success;
    int                 len;
    int                 i, iter=0;
    int                 itemp;
    int                 print_angles=TRUE;
    int                 print_profile=FALSE;  /* PRINT PROFILE OF CURRENT
SEARCH DIRECTION */
    int                 contiguous_fails=0;
    double                  fac=0.1;
    int                     counter=0, fin=FALSE;
    int                 reset_search_dir=FALSE;

#ifdef SJE_SPATIAL    
    char		opstr[80];
#endif
    
    len = imax-imin+1;
    Smax = len;

    w0  = Rvec_create(imin,imax);
    w1  = Rvec_create(imin,imax);

    w_try = Rvec_create(imin,imax);
    g_try = Rvec_create(imin,imax);

    g0 = Rvec_create(imin,imax);
    g1 = Rvec_create(imin,imax);
    s  = Rvec_create(imin,imax);
    temp_vec = Rvec_create(imin,imax);

    /* COPY ORIG WTS INTO W0 */
    for (i=imin;i<=imax;i++)
        w0[i] = w_orig[i];

    /* STEP 0 */
    /* FIND s0 = -g0 */
    APPLY_FUNCTION2(dfunc,w0,g0);
    Rvec_negate(s,g0,imin,imax);
    E0 = APPLY_FUNCTION1(func,w0);
    printf("Initial E=%.3lf\n",E0);

    S=0;
    success = TRUE;
    while ( NOT(finished(w0,iter)) )
        {
        NEWLINE;
        printf("Line search number = %d E=%.6lf\n",iter,E0); /* sje mod*/


#ifdef SJE_SPATIAL

	/* Write out some files to see how conj grad is
	 * getting on. */
	
	if (opfp != NULL) {
	  fprintf(opfp, "%d %.6lf\n",iter,E0);
	  fflush(opfp);
	}

	if ( (iter %10 ) == 0) {
	  sprintf(opstr, "z.%d", iter);
	  writeArray(z, opstr);
#ifdef SJE_WANT_ZBAR	    
	  sprintf(opstr, "zbar.%d", iter);
	  writeArray(zbar, opstr);
	  sprintf(opstr, "ztilde.%d", iter);
	  writeArray(ztilde, opstr);
#endif

	  sprintf(opstr, "wts.%d", iter);
	  writeWts(opstr);
	}
#endif
	
	iter ++;

         /* STEP 1 */
        if (success)    /* IFF LAST ITERATION REDUCED E */
            {
            mu = Rvec_dot(g0,s,imin,imax);
            if (mu>=0)
                {
                APPLY_FUNCTION2(dfunc,w0,g0);
                Rvec_negate(s,g0,imin,imax);
                mu = Rvec_dot(s,g0,imin,imax);
                S = 0;
                }

            /* KAPPA IS LENGTH**2 OF SEARCH VECTOR , MAG OF GRAD */
            kappa = Rvec_dot(s,s,imin,imax);
	    if (VERBOSE_TRAINING) {
	      printf("Step1 : Kappa %e\n", kappa);
	    }
            sigma = epsilon/sqrt((double)kappa);

            /* FIND WT VEC  USED TO EST 2ND DERIV BY STEPPING IN SEARCH DIR A BIT */
            Rvec_linear_comb(w_try,1,w0,sigma,s,imin,imax);

            /* FIND DERIV WRT TO NEW WT */
            APPLY_FUNCTION2(dfunc,w_try,g_try);

	    /* This next line has an extra call to the evaluation
	     * function, which may be doing a lot of work for big
	     * networks, and so this has been commented out for the
	     * moment.
	     */

#ifdef NOT_BIG_PROGRAM	    
            if (VERBOSE_TRAINING)
                    printf("Value of func after stepping sigma=%e in search dir = %le\n",
                    sigma,APPLY_FUNCTION1(func,w_try));
#endif

            /* ESTIMATE MAG OF 2ND DERIV */
            cg_gamma = find_2nd_deriv(g0,g_try,s,sigma,imin, imax);
            }

        /* STEP 2 */
        /*INCREASE WORKING CURVATURE */
        delta = cg_gamma + lambda*kappa;

        /*STEP 3 */
        if (delta<=0)
            {
            /* MAKE DELTA +VE AND INCREASE LAMBDA */
            delta = lambda*kappa;
            lambda = lambda - cg_gamma/kappa;
            }

        /* STEP 4 */
        /* CALC STEP SIZE AND ADAPT EPSILON */
        alpha = -(mu/delta);
        temp = pow((double)(alpha/sigma),(double)pi);

	/* Clip temp so that it must be in the range [0.1, 10.0] */
	/* There was a bug in the definition of JMAX macro - it was the
	 * same as the JMIN macro. */
	
        temp = JMAX(0.1,temp);
        temp = JMIN(10,temp);
        epsilon = temp*epsilon;

	/* Check on the values of alpha and epsilon. */
	if( VERBOSE_TRAINING ) {
	  printf("Alpha %lf\t Epsilon %e\n", alpha, epsilon);
	  printf("mu %e\t delta %e\n", mu, delta);
	}
	
        /* STEP 5 */
        /* CALC COMPARISON RATIO */
        /* MAKE NEW WEIGHT VECTOR AT ESTIMATED MINIMUM */
        Rvec_linear_comb(w1,1,w0,alpha,s,imin,imax);
        E1 = APPLY_FUNCTION1(func,w1);
        rho = 2.0*(E1-E0)/(alpha*mu);

        success = ( (E1<E0) ? TRUE : FALSE );

        if (!success)
            printf("Step to minimum DID NOT REDUCE E: old_E %lf new_E %lf\n",E0,E1);
        (success ? contiguous_fails=0 : contiguous_fails++);
        if (contiguous_fails > 0 AND contiguous_fails % 5 == 0)
            reset_search_dir=TRUE;

        if (print_profile)  /* GET PROFILE OF SURFACE IN SEARCH DIRECTION */
            {
            itemp = 5;
            for (i=(-itemp);i<=itemp;i++)
                {
                Rvec_linear_comb(w_try,1,w1,(double)i/itemp*alpha,s,imin,imax);
                temp = APPLY_FUNCTION1(func,w_try);
                if (VERBOSE_TRAINING)
                    printf("alpha %lf E=%lf\n",(double)i/itemp*alpha,temp);
                }
            }

        /* STEP 6 */
        if (rho>RHO_MAX)
            lambda = (lambda+LAMBDA_MIN)/2.0;
        else if (rho<RHO_MIN && lambda<LAMBDA_MAX)
                    {
                    temp = delta*(1.0-rho)/kappa;
                    if (temp < (LAMBDA_MAX-lambda))
                        lambda = lambda + temp;
                    }

        /* STEP 7 */
        if (success==TRUE)
            {
            APPLY_FUNCTION2(dfunc,w1,g1); /* sje mod, was *dfunc, make it dfunc*/
            /* SWAP OLD FOR NEW WEIGHT AND DERIV VECTORS */
            ptr = g0;       g0 = g1;        g1 = ptr;
            ptr = w0;   w0 = w1;        w1 = ptr;
            E0 = E1;
            S++;
            if (print_angles)
                if (VERBOSE_TRAINING)
                    printf("Angle between old/new GRADIENT = %lf\n",
                                            Rvec_angle(g0,g1,imin,imax));
            }

         if (VERBOSE_TRAINING OR iter % 10 == 0)
                printf("Line search number %d, f(w)= %.3lf \n",iter,E0);

        /* STEP 8 */
        /* CHOOSE NEW SEARCH DIRECTION  */
        if (S==Smax OR reset_search_dir)
            {
            printf("\n\n\nResetting search direction to steepest descent\n\n\n");
            Rvec_negate(s,g0,imin,imax);
            success = TRUE;
            S = 0;
            reset_search_dir = FALSE;
            }
        else
            if (success)  /*  if E has been reduced */
                {
                /* CREATE NEW CONJ DIRECTION */
                beta = Rvec_dot(g0,g1,imin,imax) - Rvec_dot(g0,g0,imin,imax);
                beta = beta/mu;
                Rvec_linear_comb(temp_vec,-1.0,g0,beta,s,imin,imax);
                if (print_angles)
                    if (VERBOSE_TRAINING)
                        printf("Angle between old/new search direction = %lf\n",Rvec_angle(temp_vec,s,imin,imax));
                Rvec_copy(s,temp_vec,imin,imax);
                }
        }



    printf("cg_williams terminating: iter=%d fin=%d\n",iter,fin);

    for (i=imin;i<=imax;i++)
        w_orig[i]=w0[i];

    Rvec_destroy(w0,imin,imax);
    Rvec_destroy(w1,imin,imax);

    Rvec_destroy(w_try,imin,imax);
    Rvec_destroy(g_try,imin,imax);

    Rvec_destroy(g0,imin,imax);
    Rvec_destroy(g1,imin,imax);
    Rvec_destroy(s,imin,imax);
    Rvec_destroy(temp_vec,imin,imax);

    return(fin);  /* FINISH IS CODE FOR REASON FOR TERMINATING */
}

double
find_2nd_deriv(g0,g_try,s,sigma,imin, imax)
double *g0;
double *g_try;
double *s;
double sigma;
int imin;
int imax;
{
    double    *temp_vec;
    double     cg_gamma=0.0;
    int     i;

    for (i=imin;i<=imax;i++)
        cg_gamma = cg_gamma + s[i]*(g_try[i]-g0[i]);

    cg_gamma  = cg_gamma/sigma;
    return((double)cg_gamma);
}

double *
Rvec_create(nl,nh)
int nl,nh;
{
    double  *v;
    int i;
    static int val=1;

    v=(double *)malloc((unsigned int) ( (nh-nl+1)*sizeof(double) ));
    if (!v) printf("allocation failure in Rvec_create()");
    /* jim mod for mac  */
    v = v-nl;
    for (i=nl;i<=nh;i++)
        v[i]=0.0;

    return (v);
}

int
Rvec_destroy(double *v,int nl,int nh)
{
#ifdef SUSSEXCOMPILE
  /* The free() command was orginally passed with two arguments when it 
   * only wants one argument.
   free((char*) (v+nl),(unsigned int)(nh-nl+1)*sizeof(double));
   */
#else
free(v+nl);
#endif
}

int
Rvec_copy(double *v0,double *v1,int imin, int imax)
{
    v0 += imin;
    v1 += imin;
    while (imin++ <= imax)
        *(v0++) = *(v1++);
}
double
Rvec_dot(double *v0,double *v1,int imin,int imax)
{
    register double sum=0.0;

    v0 += imin;
    v1 += imin;
    while (imin++ <= imax)
        sum += *(v0++) * *(v1++);
    return((double)sum);
}

int
Rvec_negate(double *v0,double *v1,int imin,int imax)
{
    v0 += imin;
    v1 += imin;
    while (imin++ <= imax)
        *(v0++) = -(*(v1++));
}
int
Rvec_linear_comb(double *v0,double a,double *v1,double b,double *v2,int imin,int imax)
{
    register int counter = imax-imin+1;

    v0 += imin;
    v1 += imin;
    v2 += imin;
    while (counter--)
            *v0++ =  a * *v1++ + b * *v2++;
}

double
Rvec_angle(double *v1,double *v2,int imin, int imax)
{
    register double     len1, len2, dp, norm, cosang;

    len1 = (double) Rvec_length_sq(v1,imin,imax);
    len2 = (double) Rvec_length_sq(v2,imin,imax);
    dp   = (double) Rvec_dot(v1,v2,imin,imax);
    norm = sqrt((double)len1*len2);
    cosang = dp/norm
;

    return(JACOS( cosang ));
}

double
Rvec_length_sq(v1, imin, imax)
double *v1;
int imin;
int imax;
 {
     register double         sum=0.0;


     v1 += imin;

     while (imin++ <= imax)
         sum += *v1 * *v1, v1++; /* to prevent side effects from order of
eval*/
     return((double)sum);
 }

/****************************************************************************
***
***
*** cg_williams_module.c
*** 
*** Stephen Eglen
*** COGS, University of Sussex.
***
*** Created 23 Nov 95
***
*** $Revision: 1.6 $
*** $Date: 1998/03/24 17:13:15 $
****************************************************************************/


#ifndef lint
static char *rcsid = "$Header: /home/stephen/disparity/cg_williams_module.c,v 1.6 1998/03/24 17:13:15 stephen Exp stephen $";
#endif



/*************************** Version Log ****************************/
/*
 * $Log: cg_williams_module.c,v $
 * Revision 1.6  1998/03/24 17:13:15  stephen
 * Brief check in
 *
 * Revision 1.5  1995/12/11 06:25:15  stephene
 *  put in #ifdef SJE_SPATIAL, to make things slighlty easier to run this code
 *  on other networks.
 *
 *  writing the weights out along with z, zbar and ztilde.
 *
 * Revision 1.4  1995/12/08  15:10:31  stephene
 * simple change to the file
 *
 * Revision 1.3  1995/12/08  00:17:07  stephene
 * Found bugs in header file for definitions of JMAX and FALSE
 *
 * Revision 1.2  1995/12/07  15:42:25  stephene
 * post nips sort out
 *
 * Revision 1.1  1995/11/23  16:29:51  stephene
 * Initial revision
 *
 */
