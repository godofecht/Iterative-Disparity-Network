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


void Neuron::updateInputWeights(Layer &prevLayer) //Formulas adapted into code.
{
	for (unsigned n = 0; n < prevLayer.size(); n++) {
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaweight;


		double newDeltaWeight =
			// Individual input is magnified by the gradient and train rate:
			0.0001 //No learning rate because F adjusts it automatically
			* m_outputVal
			* m_gradient;
						// Also adding momentum = a fraction of the previous delta weight;

//		cout<<newDeltaWeight<<endl;

//		if(isnan(newDeltaWeight)) newDeltaWeight = 0.0;
//		cout<<newDeltaWeight<<endl;

		neuron.m_outputWeights[m_myIndex].deltaweight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}

double Neuron::getOutputVal()
{
	return m_outputVal;
}
void Neuron::setOutputVal(double n)
{
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
	return 1.0 - x*x;
}
double Neuron::transferFunction(double x)
{
	return tanh(x);
}

int Neuron::getIndex()
{
	return m_myIndex;
}
void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	double derivative = Neuron::transferFunctionDerivative(m_outputVal);
	m_gradient = dow*derivative;
}
void Neuron::calcInputGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	double derivative = 1.0;

//	double derivative = Neuron::transferFunctionDerivative(m_outputVal);
	m_gradient = dow*derivative;
}


/*
void Neuron::calcOutputGradients(double error) //Looks at the difference between the target values and the output values.
{
	double delta = error;
	double derivative = Neuron::transferFunctionDerivative(m_outputVal);
//	m_gradient = dow*derivative;
	m_gradient = delta * 1.0;

}
*/

void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - m_outputVal;
	m_gradient = delta * 1.0;


//	cout<<m_gradient<<endl;
}
