#include "Network.h"
#include "NN.cpp"
#include <vector>

Network::Network(const vector <unsigned> &topology)
{
	srand((unsigned int)time(NULL));

	unsigned numLayers = topology.size();

//	int num_layers = 0;
	for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {
		m_layers.push_back(Layer());
//		num_layers++;
//		cout<<num_layers<<endl;
		unsigned numOutputs;
		if(layerNum == topology.size()-1)
			numOutputs = 0;
		else
			numOutputs = topology[layerNum+1];// layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		// We have a new layer, now fill it with neurons
		int num_neurons = 0;
		for (unsigned neuronNum = 0; neuronNum < topology[layerNum]; neuronNum++) {
			num_neurons++;
		//	cout<<num_neurons<<endl;
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
	//		cout<<neuronNum<<" "<<numOutputs<<endl;
		}
	}
}

double Network::getDUDX(int a) //this is the same as dudw
{

}


double Network::getDVDX(int a)
{

}


double Network::getErrorDerivative(double dvdx,double dudx)
{
	double v1 = 1.0/V;
	double u1 = 1.0/U;

	double v1dvdx = v1 * dvdx;
	double u1dudx = u1 * dudx;



	double deriv;
	deriv = v1dvdx - u1dudx;


//	cout<<deriv<<endl;


	return deriv;
}

void Network::CalcF()
{
	double a = V/U;
//	if(isnan(a)) a = 0.0;
	F = log(a);
//	cout<<F<<endl;
//	cout<<"V: "<<V<<endl;
//	cout<<"U: "<<U<<endl;



}

void Network::CalcFDerivative()
{
//	FDeriv = 1.0/V * dvdw - 1.0/U * dudw;
}

void Network::backPropagate(const vector <double> &targetVals)
{
	// Calculate overall net error (RMS of output neuron errors)

	Layer &outputLayer = m_layers.back();
	Layer &hiddenLayer = m_layers[1];
	Layer &inputLayer = m_layers[0];

	m_error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta;
	}
	m_error /= outputLayer.size() - 1; // get average error squared
	m_error = -sqrt(m_error); // RMS

	// Implement a recent average measurement

	m_recentAverageError =
		(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
		/ (m_recentAverageSmoothingFactor + 1.0);

	// Calculate output layer gradients

	for (unsigned n = 0; n < outputLayer.size(); n++) {
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	// Calculate hidden layer gradients
	for (unsigned n = 0; n < hiddenLayer.size(); n++) 
	{
		hiddenLayer[n].calcHiddenGradients(outputLayer);
	}
	for (unsigned n = 0; n < inputLayer.size(); n++) 
	{
		inputLayer[n].calcInputGradients(hiddenLayer);
	}

	// For all layers from outputs to first hidden layer,
	// update connection weights

	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n) {
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Network::backPropagate(double m_error)
{

	Layer &outputLayer = m_layers.back();
	Layer &hiddenLayer = m_layers[1];
	Layer &inputLayer = m_layers[0];


	// Calculate output layer gradients
	for (unsigned n = 0; n < outputLayer.size(); n++) {
		outputLayer[n].calcOutputGradients(m_error);
	}
	// Calculate hidden layer gradients
	for (unsigned n = 0; n < hiddenLayer.size(); n++) 
	{
		hiddenLayer[n].calcHiddenGradients(outputLayer);
	}
	for (unsigned n = 0; n < inputLayer.size(); n++) 
	{
		inputLayer[n].calcInputGradients(hiddenLayer);
	}
	// For all layers from outputs to first hidden layer,
	//5. Calculate weight changes for all weights
	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; layerNum--) 
	{
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < layer.size(); n++) 
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}



vector<double>  Network::GetWeights() const
{
	//this will hold the weights
	vector<double> weights;
	//for each layer
	for (int i = 0; i<m_layers.size(); i++)
	{
		//for each neuron
		for (int j = 0; j<m_layers[i].size(); j++)
		{
			//for each weight
			for (int k = 0; k<m_layers[i][j].m_outputWeights.size(); k++)
			{
				weights.push_back(m_layers[i][j].m_outputWeights[k].weight);
			}
		}
	}
	return weights;
}






//Double checked functions
//This is done
void Network::feedForward(vector <double> &inputVals)
{
	assert(inputVals.size() == m_layers[0].size());
	// Assign (latch) the input values into the input neurons
	for (unsigned i = 0; i < inputVals.size(); i++) {
		m_layers[0][i].setOutputVal(inputVals[i]);
//		cout<<inputVals[i]<<endl;
	}

	//Feed forward from input layer to hidden layer
	for (unsigned n = 0; n < m_layers[1].size(); n++) 
	{
		m_layers[1][n].feedForwardHidden(m_layers[0]);
	}

	//feed forward from hidden layer to output layer
	for (unsigned n = 0; n < m_layers[2].size(); n++) 
	{
	//	cout<<m_layers[2][n].getOutputVal()<<endl;
		m_layers[2][n].feedForwardIO(m_layers[1]);
	//	cout<<m_layers[2][n].getOutputVal()<<endl;
	}

}


void  Network::getResults(vector <double> &resultVals)
{
	resultVals.clear();
	for (unsigned n = 0; n < m_layers.back().size(); n++) {
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

/*
void Network::PutWeights(vector<double> &weights)
{
	int cWeight = 0.0;
	//for each layer
	for (int i = 0; i<m_layers.size(); i++)
	{
		//for each neuron
		for (int j = 0; j<m_layers[i].size(); j++)
		{
			//for each weight (first 11 weights)
			for (int k = 0; k<m_layers[i][j].m_outputWeights.size(); k++)
			{
				m_layers[i][j].m_outputWeights[k].weight = weights[cWeight++];
			}
		}
	}
}
*/

void Network::PutWeights(vector<double> &weights)
{
	Layer &outputLayer = m_layers.back();
	Layer &hiddenLayer = m_layers[1];
	Layer &inputLayer = m_layers[0];


 	//iterate through hidden layers, taking 1st output of 1st layer


	int weightcounter = 0;


	for (int h = 0; h < hiddenLayer.size(); h++)
	{
		for (int i = 0; i < inputLayer.size(); i++)
		{
			inputLayer[i].m_outputWeights[h].weight = weights[weightcounter];
			weightcounter++;

		}
	}
	for(int j=0;j<hiddenLayer.size();j++)
	{
		hiddenLayer[j].m_outputWeights[0].weight = weights[weightcounter];
		weightcounter++;
	}
	

	//iterate through weights from hidden layer to single output


}