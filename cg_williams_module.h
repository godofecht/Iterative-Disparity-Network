#include <stdio.h>



#ifdef oldTrueFalseDefns
/* These old definititions caused errors , as the value of FALSE is non
 * zero and so it would return a true value in a conditional.
 */
#define TRUE    (1.0)
#define FALSE   (-1.0)

#else

#define FALSE                  0
#define TRUE                   1

#endif

#define NOT(x)  (!(x))



#ifdef jimdefapply
#define APPLY_FUNCTION1(f,x)  			(*f)(x)
#define APPLY_FUNCTION2(f,x1,x2)  	(*f)((x1),(x2))
#else
#define APPLY_FUNCTION1(f,x)  			f((x))
#define APPLY_FUNCTION2(f,x1,x2)  	f((x1),(x2))
#endif


#define OR      ||
#define AND     &&


/* Bug in the old definition of JMAX -- it was exactly the same as JMIN */

#define JMIN(x,y)   ( (x)<(y)?(x):(y) )
#define JMAX(x,y)   ( (x)>(y)?(x):(y) )
#define JACOS(x)    acos( (double) (x) )

#define NEWLINE printf("\n")


int
CG_Williams();


double
find_2nd_deriv();

double *
Rvec_create();

int
Rvec_destroy();

int
Rvec_copy();
double
Rvec_dot();

int
Rvec_negate();

int
Rvec_linear_comb();


double
Rvec_angle(double *v1,double *v2,int imin, int imax) ;

double
Rvec_length_sq(double *v1, int imin, int imax);

int
Rvec_linear_comb(double *v0,double a,double *v1,double b,double *v2,int imin,int imax)
;
