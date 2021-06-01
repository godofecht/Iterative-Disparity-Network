/****************************************************************************
***
*** Time-stamp: <11 Nov 95 18:15:04 stephene>
***
*** dispnet.h
*** 
*** Stephen Eglen
*** COGS, University of Sussex.
***
*** Created 09 Nov 95
***
*** $Revision: 1.11 $
*** $Date: 1995/12/11 06:25:40 $
****************************************************************************/


#ifndef _DISPNET_H
#define _DISPNET_H

/* Header file for the disparity network */

#define cfree(p) free((p))

/* By default, we will be using doubles for weights and activations */
typedef double Real;


/* Do we need bias for a layer? */
typedef  enum  { Nobias, Bias} BiasType;

/* Possible types of activation function */
typedef  enum {Linearfn, Tanhfn} ActFn;

typedef struct {
  int  numInputs;
  int *inputs;
  Real **ptrInputs;
  Real *wtsStart;
} cellInfo_t;

/* cellInfo_t:	Tells a cell where its input is coming from, and how
 *		many inputs there are.  inputs[i] stores the location
 *		of the input cell in the activations array that is
 *		part of the input to this cell.
 *		ptrInputs stores the pointer to the input cell.
 *		i.e. ptrInput[i] = & (actInfo.op[ inputs[i] ]);
 */

/* There is one cellInfo structure for each cell in the op layer, but
   there is no need to have cellInfo structure for the bias units, as
   these cells do not receive any input. */



/*** Presynaptic cell information ***/

typedef struct {
  int nOutputs;			/* number of connections that this
 				   presynaptic cell makes. Initially
 				   will be zero, and then incremented */
  Real **wts;			/* wts[i] stores a pointer to the ith
 				   fan out weight from this cell*/
  int *wtsIndex;		/* wtsIndex[i] is the weight number
    				   that connects this preCell to
    				   outputCell.*/
  int *outputs;			/* output[i] stores the number of the
 				   postsynaptic cell that this weight
 				   is connected to. */
				/* IS THIS Absolute or relative? */

  Real **ptroutputs;		/* ptrouputs[i] is a pointer to the
  				   activation of the ith postsynaptic
  				   cell */
} preCellInfo_t;

 
typedef struct {
  int tlx;	int tly;	/* Top Left */
  int brx;	int bry;	/* Bottom Right */
} Rect;

   
/* layerInfo_t	Information for a layer. */

typedef struct {
  int       	ncells;		/* this equals nrows * ncols. It
				 * therefore does not include any Bias
				 * weight, which will need to be
				 * explicitly checked for.*/
				   
  int       	nrows;		/* same as the ht */
  int       	ncols;		/* same as the wid */
  cellInfo_t 	*cellInfo;	/* cellInfo[i] stores the input
				   information for cell i in this
				   layer */
  ActFn     	actfn;		/* Which activation function cells in this
				   layer use. */
  BiasType	 bias;		/* Does this layer provide bias for
				   the next layer? */
  preCellInfo_t	*preCellInfo;	/* preCellInfo[i] stores the
  				   presynaptic info for cell i of this
  				   layer */
  int		nPreCellInfo;	/* number of elements in the
  				 * preCellInfo array. This value can
  				 * either be ncells or ncells+1,
  				 * depending on whether there is a
  				 * bias cell in this layer or not. */
  
} layerInfo_t;



/* weightInfo_t:	Information about the weight structure */
typedef struct {
  Real	*data;		/* Actual weight vector */
  int	nextFreeWeight;	/* Index to next free weight element, as
			 * weights are being allocated by the function
			 *  nextFreeWeight() */			   
  int	maxIndex;	/* Maximum index into the weight vector */
  int	numWts;		/* Number of weights allocated.
			 * ie, last weight stored in data[numWts-1] */
  int	*preCell;	/* preCell[i] gives the index number of the
			 * presynaptic cell for the ith weight */
  int	*postCell;	/* postCell[i] gives the index number of the
			 * postsynaptic cell for the ith weight */
} weightInfo_t;



typedef struct {
  Real *actn;
  Real *op;
  int  *startLayer;
  int  *biasIndex;
  int  size;
} activationInfo_t;

/* ActivationInfo:	Stores the activation level of the cells 
 * 
 * actn			actn[i] stores the activation level of the ith cell
 *
 * op			op[i] stores the output of the ith cell.
 *
 * startLayer 		startLayer[i] stores the index to the start of the
 *			cells for the ith layer
 * 
 * biasIndex		biasIndex[i] stores the index to the ith bias unit.
 *			If there is no bias unit, then this value should
 *			equal 99999 to try and cause an error.
 *
 * size			size of the activation array, ie. actn[0 to size-1].
 * */





/* netInfo_t:	General information about the network.
 *
 * nLayers -  number of layers in the network. These will be labelled
 *            from 0 to the Nlayers-1. Layer 0 is the input layer.  */

typedef struct {
  int nLayers;			/* Number of layers in the net. */
  double f;			/* Value of the merit function */
} netInfo_t;


typedef struct {
  Real	*data;
  int	wid;
  int	ht;
}  Array;

/* Array is a simple structure for a 2d array. If the array is one
 * dimensional, then the ht element is set to 1. */


typedef struct {
  Real	*data;
  int	wid;
  int	ht;
  int	centre; /* Centre element of the mask, whether it is 1D or 2D */
  int	maskExtent;
} Mask;	

/* Mask: simple 2d array like structure used for convolutions.
 * If the array is one dimensional, then the height element will be one. 
 */


typedef struct {
  int	num;
  Real	**allActs;
  Real	**allOps;
  Real	**errors;
  int	numUnits;
} allActns_t;

/*
 * allActsn is an array of size [0..num-1].  allActs[i] points to a
 * structure, containing the activations, outputs and errors for the
 * input numbered i.
 *
 * allOps[i] stores the array of output values when the network is
 * presented with the ith input vector.  errors[i] stores the error
 * for cell i.  Each array allOps[i] and allActs[i] are of size
 * [0..numUnits-1].
 *
 * This is quite a large structure, as it stores all of the
 * activations, outputs and errors for each input vector.
 *
 * Relevant functions:
 *
 * createAllActns(), freeAllActns(), storeAllActns().  void
 * printAllActns(char *fname) */

/*** Function definitions ***/
void calcAllActivations();
void calcActivation(int layer);
void clearUpMemory();

/* General Array functions */
void createArray( int wid, int ht, Array *rarr);
void createRndArray( int wid, int ht, Array *rarr);
void freeArray(Array arr);
void writeArray(Array arr, char *fname);
void subArray(Array a1, Array a2, Array result);
void multArrayInPlace(Array a1, double k);
void setArray(Array a1, double k);
void addArrayInPlace(Array a1, Array a2);

void setParamDefaults();
void getParams(char *fname);

void createAllActns();
void freeAllActns();
void printAllActns(char *fname);
void storeActivations(int input);

void getZ();
void createZs();
void freeZs();

void createdw();
void freedw();


double arrayDist(Array a1, Array a2);

Real Rvec_correlate(Real *x, Real *y, int imin, int imax);

/************************* Global Variables *************************/
extern weightInfo_t	weightInfo;
extern netInfo_t	netInfo;
extern activationInfo_t actInfo;
extern layerInfo_t	*layerInfo;
/************************* Global Variables *************************/



/***  Define Statements ***/
#undef dumpArrays		/* Do we want to output all of the
				 * weights and so on every time?  If
				 * so, define dumpArrays */

#endif


/*************************** Version Log ****************************/
/* $Log: dispnet.h,v $
 * Revision 1.11  1995/12/11  06:25:40  stephene
 * *** empty log message ***
 *
 * Revision 1.10  1995/12/08  00:17:48  stephene
 * Inclusion of Rvec_correlate
 *
 * Revision 1.9  1995/12/07  15:42:46  stephene
 * post nips sort out
 *
 * Revision 1.8  1995/11/23  16:19:52  stephene
 * CG now installed
 *
 * Revision 1.7  1995/11/21  23:32:47  stephene
 * About to include CG Code
 *
 * Revision 1.6  1995/11/21  02:32:11  stephene
 * Update - moving towards a merit function
 *
 * Revision 1.5  1995/11/17  00:05:31  stephene
 * Added new element op to the activation Info array - this stores the
 * actual output from the cells.
 *
 * Revision 1.4  1995/11/13  22:10:41  stephene
 * New structure preCellInfo_t to store pre synaptic cell Information
 *
 * Revision 1.3  1995/11/12  23:06:27  stephene
 * Daily change
 *
 * Revision 1.2  1995/11/10  15:29:10  stephene
 * Daily update - next major step is to present net with input and
 * calculate activations.
 *
 * Will also need to initialise weights.
 *
 * Revision 1.1  1995/11/09  20:59:27  stephene
 * Initial revision
 *
 *
 */