#include<vector>
#include<cmath>
#include<cassert>
#include<iostream>
#include "NN.h"
#include<time.h>
#include<stdlib.h>
using namespace std;

 Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
	for (unsigned c = 0; c < numOutputs; c++) {

		m_outputWeights.push_back(connection());
		m_outputWeights[c].weight = ((double(rand()) / double(RAND_MAX))*(2.0-1.0)*(0.3));
		m_outputWeights[c].deltaweight = 0.0;
	}
	m_myIndex = myIndex;


	eta = 0.15;
	alpha = 0.5;
}


void Neuron::feedForwardIO(Layer &prevLayer) //Input Output Layer Activations
{
	double sum = 0.0;
	for (unsigned n = 0; n < prevLayer.size(); n++) {
		sum += prevLayer[n].getOutputVal() *
		prevLayer[n].m_outputWeights[m_myIndex].weight;
	}
	
	m_outputVal = sum;
}

void Neuron::feedForwardHidden(Layer &prevLayer) //Hidden Layer Activations
{
	double sum = 0.0;
	for (unsigned n = 0; n < prevLayer.size(); n++) {
		sum += prevLayer[n].getOutputVal() *
			prevLayer[n].m_outputWeights[m_myIndex].weight;
	}


	m_outputVal =  Neuron::transferFunction(sum);
}
void Neuron::updateInputWeights(Layer &prevLayer,vector<double> weights,vector<double> dfdws) //Formulas adapted into code.
{
	for (unsigned n = 0; n < prevLayer.size(); n++)
	{
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaweight;
		
		double newDeltaWeight =
			// Individual input is magnified by the gradient and train rate:
			0.01
			* m_gradient
			* neuron.getOutputVal()
			+ 0.10
			* oldDeltaWeight;

	//	cout<<newDeltaWeight<<endl;

	//	cout<<neuron.m_gradient<<" "<<neuron.m_outputWeights[m_myIndex].DFDW<<" "<<m_gradient <<endl;

		//	cout<<neuron.m_outputWeights[m_myIndex].DFDW<<" "<<m_gradient<<endl;


		neuron.m_outputWeights[m_myIndex].deltaweight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}

void Neuron::updateInputWeights(Layer &prevLayer) //Formulas adapted into code.
{
	for (unsigned n = 0; n < prevLayer.size(); n++) {
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaweight;


		double newDeltaWeight =
			// Individual input is magnified by the gradient and train rate:
			0.01//No learning rate because F adjusts it automatically
			* m_gradient
			* neuron.getOutputVal();
			+ 0.10
			* oldDeltaWeight;
						// Also adding momentum = a fraction of the previous delta weight;


//		cout<<newDeltaWeight<<endl;

//		if(isnan(newDeltaWeight)) newDeltaWeight = 0.0;
//		cout<<newDeltaWeight<<endl;

		neuron.m_outputWeights[m_myIndex].deltaweight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}
double Neuron::getOldOutputVal()
{
	return m_oldOutputVal;
}

double Neuron::getOutputVal()
{
	return m_outputVal;
}
void Neuron::setOutputVal(double n)
{
	m_oldOutputVal = m_outputVal;
	m_outputVal = n;
}
double Neuron::randomWeight()
{
	return (double(rand()) / double(RAND_MAX));
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;
	// Sum our contributions of the errors at the nodes we feed.
	for (unsigned n = 0; n < nextLayer.size(); n++) 
	{
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}
	return sum;
}
double Neuron::transferFunctionDerivative(double x)
{
//	return (1.0 - x*x);
	return(1.0 - tanh(x) * tanh(x));
}
double Neuron::transferFunction(double x)
{
	return tanh(x);
}

int Neuron::getIndex()
{
	return m_myIndex;
}
void Neuron::calcInputGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	double derivative = 1.0;
	m_gradient = dow*derivative;
	clip(m_gradient);
}
void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	double derivative = Neuron::transferFunctionDerivative(m_outputVal);
	m_gradient = dow*derivative;
	clip(m_gradient);
}



void Neuron::calcOutputGradients(double error) //Looks at the difference between the target values and the output values.
{
	m_gradient = error;
	clip(m_gradient);

}

/*
void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - m_outputVal;
	m_gradient = delta * m_outputVal;

	clip(m_gradient);
//	cout<<m_gradient<<endl;
}
*/




//SFA stuff
void Neuron::computeAverages()
{
	for (int i = 0; i < m_outputWeights.size(); i++)
	{
		m_outputWeights[i].y_bar_old = m_outputWeights[i].y_bar;
		m_outputWeights[i].y_tilde_old = m_outputWeights[i].y_tilde;
		m_outputWeights[i].y_tilde = (lambda_s)*m_outputWeights[i].y_tilde + (1.0 - lambda_s) * getOutputVal();
		m_outputWeights[i].y_bar = (lambda_l)*m_outputWeights[i].y_bar + (1.0 - lambda_l) * getOutputVal();
	}
}