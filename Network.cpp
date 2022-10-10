#include "Network.h"
//#include "NN.cpp"
//#include <vector>

Network::Network(const vector<unsigned> &topology)
{
	srand((unsigned int)time(NULL));

	unsigned numLayers = topology.size();

	//	int num_layers = 0;
	for (unsigned layerNum = 0; layerNum < numLayers; layerNum++)
	{
		m_layers.push_back(Layer());
		unsigned numOutputs;
		if (layerNum == topology.size() - 1)
			numOutputs = 0;
		else
			numOutputs = topology[layerNum + 1]; // layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
		// We have a new layer, now fill it with neurons
		int num_neurons = 0;
		for (unsigned neuronNum = 0; neuronNum < topology[layerNum]; neuronNum++)
		{
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			num_neurons++;
		}
	}


	dztdw = 0.0;
	dzbdw = 0.0;
	dudw = 0.0;
	dvdw = 0.0;
}

void Network::UpdateWeights()
{
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; layerNum--)
    {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < layer.size(); n++)
        {
            layer[n].updateInputWeights(prevLayer, GetWeights(), GetAllDFDW());
        }
    }
}

double dztdw,dzbdw;

double Network::getDUDX() //this is the same as dudw
{
	dztdw = lambda_l * (dztdw) + (1.0-lambda_l) * y_old;
	dudw += (y_tilde - y) * (dztdw - y);
}

double Network::getDVDX()
{
	dzbdw = lambda_s * (dzbdw) + (1.0-lambda_s) * y_old;
	dvdw += (y_bar - y) * (dzbdw - y);
}

double Network::getErrorDerivative()
{
	double v1 = 1.0 / V;
	double u1 = 1.0 / U;

	double v1dvdx = v1 * dvdw;
	double u1dudx = u1 * dudw;

	double deriv;
	deriv = v1dvdx - u1dudx;



	return deriv;
}

void Network::CalcF()
{
	double a = V / U;
	//	if(isnan(a)) a = 0.0;
	F = log(a);
	//	cout<<F<<endl;
	//	cout<<"V: "<<V<<endl;
	//	cout<<"U: "<<U<<endl;
}

void Network::CalcFDerivative() //Not correct, needs to be fixed
{
	Layer &outputLayer = m_layers.back();
//	FDeriv = outputLayer[0].m_outputWeights[0].DFDW;

	dztdw = lambda_s * (dztdw) + (1.0 - lambda_s) * dztdw;
	dzbdw = lambda_l * (dzbdw) + (1.0 - lambda_l) * dzbdw;

	dudw += (y_tilde - y) * (dztdw - y_old); // + (1.0 - lambda_s) * oldOutputVal - outputVal);
	dvdw += (y_bar - y) * (dzbdw - y_old);	 // + (1.0 - lambda_l) * oldOutputVal - outputVal);

//	cout<<(dudw)<<endl;

}

void Network::ConjugateGradientDescent(vector<double> weights,vector<double> search_dirs, vector<double> gradients,vector<double> step_lengths)
{
	for(int n =0;n <weights.size();n++)
	{
		search_dirs[n] = -gradients[n] + 0.1 * search_dirs[n];
		weights[n] = weights[n] + search_dirs[n] * step_lengths[n];
	}
}

void Network::backPropagate(const vector<double> &targetVals,const vector<double> &currentVals)
{
	// Calculate overall net error (RMS of output neuron errors)

	Layer &outputLayer = m_layers.back();
	Layer &hiddenLayer = m_layers[1];
	Layer &inputLayer = m_layers[0];

	double m_error = 0.0;

	for (int n = 0; n < targetVals.size(); n++)
	{
		double delta = targetVals[n] - currentVals[n];
		m_error += (delta * delta);

	}

	m_error /= 1000.0 - 1.0; // get average error squared (I'm not sure about this PRT)

	m_error = -sqrt(m_error);		   // RMS
//	cout<<m_error<<endl;
	// Calculate output layer gradients
	for (int n = 0; n < outputLayer.size(); n++)
	{
		outputLayer[n].calcOutputGradients(m_error);
	}
	// Calculate hidden layer gradients
	for (int n = 0; n < hiddenLayer.size(); n++)
	{
		hiddenLayer[n].calcHiddenGradients(outputLayer);
	}
	for (int n = 0; n < inputLayer.size(); n++)
	{
		inputLayer[n].calcInputGradients(hiddenLayer);
	}
	// For all layers from outputs to first hidden layer,
	// update connection weights



	for (int layerNum = m_layers.size()-1; layerNum > 0; layerNum--)
	{
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum-1];

		for (int n = 0; n < layer.size(); n++)
		{
		//	cout<<n<<endl;
			layer[n].updateInputWeights(prevLayer,GetWeights(),GetAllDFDW());
		}
	}


}

void Network::backPropagate(double m_error)
{
	Layer &outputLayer = m_layers.back();
	Layer &hiddenLayer = m_layers[1];
	Layer &inputLayer = m_layers[0];

	// Calculate output layer gradients
	for (int n = 0; n < outputLayer.size(); n++)
	{
		outputLayer[n].calcOutputGradients(m_error);
	}
	// Calculate hidden layer gradients
	for (int n = 0; n < hiddenLayer.size(); n++)
	{
		hiddenLayer[n].calcHiddenGradients(outputLayer);
	}
	for (int n = 0; n < inputLayer.size(); n++)
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
			layer[n].updateInputWeights(prevLayer,GetWeights(),GetAllDFDW());
		}
	}
	
}

vector<double> Network::GetWeights() const
{
	//this will hold the weights
	vector<double> weights;
	//for each layer
	for (int i = 0; i < m_layers.size(); i++)
	{
		//for each neuron
		for (int j = 0; j < m_layers[i].size(); j++)
		{
			//for each weight
			for (int k = 0; k < m_layers[i][j].m_outputWeights.size(); k++)
			{
				weights.push_back(m_layers[i][j].m_outputWeights[k].weight);
			}
		}
	}
	return weights;
}

//Double checked functions
//This is done
void Network::feedForward(vector<double> &inputVals)
{
	assert(inputVals.size() == m_layers[0].size());
	// Assign (latch) the input values into the input neurons
	for (unsigned i = 0; i < inputVals.size(); i++)
	{
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

void Network::getResults(vector<double> &resultVals)
{
	resultVals.clear();
	for (unsigned n = 0; n < m_layers.back().size(); n++)
	{
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
	for (int j = 0; j < hiddenLayer.size(); j++)
	{
		hiddenLayer[j].m_outputWeights[0].weight = weights[weightcounter];
		weightcounter++;
	}

	//iterate through weights from hidden layer to single output
}
