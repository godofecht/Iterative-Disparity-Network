#ifndef NN_H
#define NN_H

#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>

using namespace std;

class Neuron;
typedef vector<Neuron> Layer;
class connection
{
public:
	double weight;
	double deltaweight;

	double lambda_s = 0.9785720620877001;
	double lambda_l = 0.999783414964001;

	void setDW(double dw)
	{
		deltaweight = dw;
	}

	//SFA stuff
	double y_bar = 0.0;
	double y_bar_old = 0.0;
	double y_tilde = 0.0;
	double y_tilde_old = 0.0;
	double DUDW = 0.0;
	double DVDW = 0.0;
	double U = 0.0;
	double V = 0.0;

	double DFDW = 0.0;

	double dztdw = 0.0;
	double dzbdw = 0.0;

	double dzdw = 0.0;
	double dzdw1 = 0.0;
	double dztdw1 = 0.0;
	double dzbdw1 = 0.0;

	void CalcdzsHidden(double zjt)
	{
		dzdw = zjt;
		dztdw = lambda_s * (dztdw1) + (1.0 - lambda_s) * dzdw1;
		dzbdw = lambda_l * (dzbdw1) + (1.0 - lambda_l) * dzdw1;
	}

	void CalcdzsInput(double weight_jk, double zjt, double outputVal)
	{
		dzdw = weight_jk * (1.0 - pow(zjt, 2.0)) * outputVal; //Equivalent to tanh derivative
		dztdw = lambda_s * (dztdw1) + (1.0 - lambda_s) * dzdw1;
		dzbdw = lambda_l * (dzbdw1) + (1.0 - lambda_l) * dzdw1;
	}

	void UpdateUVHidden(double last_output, double last_output_tilde, double last_output_bar, double last_output_tilde_old, double last_output_bar_old, double oldOutputVal, double outputVal)
	{
		CalcdzsHidden(outputVal);
		DUDW += (last_output_tilde - last_output) * (dztdw - dzdw); // + (1.0 - lambda_s) * oldOutputVal - outputVal);
		DVDW += (last_output_bar - last_output) * (dzbdw - dzdw);	 // + (1.0 - lambda_l) * oldOutputVal - outputVal);

	}

	void UpdateUVInput(double last_output, double last_output_tilde, double last_output_bar, double last_output_tilde_old, double last_output_bar_old, double hidden_weight, double oldOutputVal, double outputVal, double oldHiddenVal, double hiddenVal)
	{

		/*
		DUDW += (last_output_tilde-last_output) * 
        (lambda_s * dztdw + hidden_weight*
		((1.0 - lambda_s)*(1.0-pow(oldHiddenVal,2.0))*
		oldOutputVal - (1.0 - pow(hiddenVal,2.0)) 
        * outputVal));

        DVDW += (last_output_bar-last_output) * 
        (lambda_l * dzbdw + hidden_weight*
		((1.0 - lambda_l)*(1.0-pow(oldHiddenVal,2.0))*
		oldOutputVal - (1.0 - pow(hiddenVal,2.0)) 
        * outputVal));

*/
		CalcdzsInput(hidden_weight, hiddenVal, outputVal);
		DUDW += (last_output_tilde - last_output) * (dztdw - dzdw); // + (1.0 - lambda_s) * oldOutputVal - outputVal);
		DVDW += (last_output_bar - last_output) * (dzbdw - dzdw);	 // + (1.0 - lambda_l) * oldOutputVal - outputVal);

		//	cout<<DUDW<<endl;
	}

	void Remember()
	{
		dzdw1 = dzdw;
		dztdw1 = dztdw;
		dzbdw1 = dzbdw;
	}

	void ComputeFDerivative(double U, double V)
	{
		//	cout<<V<<" "<<DVDW<<" "<<DUDW<<" "<<U<<endl;
		double v1 = 1.0 / V;
		double u1 = 1.0 / U;
		if (isnan(1.0 / V))
			v1 = 0.0;

		if (isnan(1.0 / U))
			u1 = 0.0;

		DFDW = (v1 * DVDW) - (u1 * DUDW);

		if (isnan(DFDW))
			DFDW = 0.0;


		//		cout<<DFDW<<endl;
	}
};
class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void feedForward(Layer &prevLayer);
	void feedForwardHidden(Layer &prevLayer);
	void feedForwardIO(Layer &prevLayer);
	double getOutputVal();
	void setOutputVal(double n);

	void calcInputGradients(const Layer &nextLayer);
	void calcHiddenGradients(const Layer &nextLayer);
	void calcOutputGradients(double targetVal);

	double getOldOutputVal();

	void updateInputWeights(Layer &prevLayer);
	void updateInputWeights(Layer &prevLayer,vector<double> weights,vector<double> dfdws);

	double eta;
	double alpha;
	vector<connection> m_outputWeights;

	int getIndex();

	double UpdateYBar();
	double UpdateYTilde();

	void clip(double& val)
	{
	/*	if(val <= -1.0)
			val = -1.0;
		if(val >= 1.0)
			val = 1.0;
			*/
	}

	//SFA stuff
	void computeAverages();

	double gamma_long = 0.0;
	double gamma_short = 0.0;

	double lambda_s = 0.9785720620877001;
	double lambda_l = 0.999783414964001;

	void UpdateUVHidden(double networkResult, double networkResultTilde, double networkResultBar, double oldNetworkResultTilde, double oldNetworkResultBar)
	{
		m_outputWeights[0].UpdateUVHidden(networkResult, networkResultTilde, networkResultBar, oldNetworkResultTilde, oldNetworkResultBar, getOldOutputVal(), getOutputVal());
	}

	void UpdateUVInput(double networkResult, double networkResultTilde, double networkResultBar, double oldNetworkResultTilde, double oldNetworkResultBar, vector<double> hidden_connections, vector<double> oldHiddenVals, vector<double> hiddenVals)
	{
		for (int i = 0; i < m_outputWeights.size(); i++)
		{
			m_outputWeights[i].UpdateUVInput(networkResult, networkResultTilde, networkResultBar, oldNetworkResultTilde, oldNetworkResultBar, hidden_connections[i], getOldOutputVal(), getOutputVal(), oldHiddenVals[i], hiddenVals[i]);
		}
	}

	void ComputeFDerivative(double U, double V)
	{
		for (int i = 0; i < m_outputWeights.size(); i++)
		{
			m_outputWeights[i].ComputeFDerivative(U, V);
		}
	}

private:
	double m_gradient;
	double m_outputVal;
	double m_oldOutputVal;
	static double randomWeight();
	unsigned m_myIndex;
	double sumDOW(const Layer &nextLayer) const;
	static double transferFunctionDerivative(double x);
	static double transferFunction(double x);
};

#endif