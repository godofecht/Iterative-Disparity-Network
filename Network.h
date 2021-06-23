#ifndef NETWORK_H
#define NETWORK_H

#include "NN.h"
#include <cmath>

class Network
{
public:
	Network(const vector<unsigned> &topology);
	//	void backPropagate(const vector <double> &targetVals);
	void backPropagate(double m_error);
	void backPropagate(const vector <double> &targetVals);


	void feedForward(vector<double> &inputVals);
	void getResults(vector<double> &resultVals);
	double getRecentAverageError(void) const { return m_recentAverageError; }

	vector<double> GetWeights() const;
	void PutWeights(vector<double> &weights);

	void UpdateWeights();

	void NormalizeWeights(int connection_index);

	vector<Layer> GetLayers()
	{
		return m_layers;
	}

	double V = 0.0;
	double U = 0.0;
	double F = 0.0;
	double A = 1000.0;

	double dzdw = 0.0;
	double dztdw = 0.0;
	double dzbdw = 0.0;

	vector<double> dudxs;
	vector<double> dvdxs;

	double y_tilde = 0.0;
	double y_bar = 0.0;
	double y = 0.0;
	double y_old = 0.0;
	double lambda_s = 0.021661;
	double lambda_l = 0.000021661;

	vector<double> Output_Array;

	double dudx = 0.0, dvdx = 0.0;

	vector<double> maskU;
	vector<double> maskV;
	vector<double> maskDashV;
	vector<double> maskDashU;

	vector<double> errors;

	double h_U = 32.0; //I need to move h out of the function parameters for getconvmaskdashU and getconvmaskdashV
	double h_V = 3200.0;

	//EQN A6
	void CalcV()
	{
		V = V + 0.5 * pow(y_bar - y,2.0);
	}

	void CalcU()
	{
		U = U + 0.5 * pow(y_tilde - y,2.0);
	}

	double calcDUDW()
	{
		dudw += (y_tilde - y) * (dztdw - dzdw);
	}

	double calcDVDW()
	{
		dvdw += (y_bar - y) * (dzbdw - dzdw);
	}

	void Calcdztdw() //Equation A8
	{
		dztdw = lambda_s * dztdw + (1.0 - lambda_s) * y;
	}

	void Calcdzbdw() //Equation A8
	{
		dzbdw = lambda_l * dzbdw + (1.0 - lambda_l) * y;
	}
	vector<double> GetNeuronOutputsVector()
	{
		vector<double> output_values; //of all neurons
		for (int i = 0; i < m_layers.size(); i++)
		{
			//for each neuron
			for (int j = 0; j < m_layers[i].size(); j++)
			{
				output_values.push_back(m_layers[i][j].getOutputVal());
			}
		}
		return output_values;
	}

	vector<double> GetNeuronOutputsOldVector()
	{
		vector<double> output_values; //of all neurons
		for (int i = 0; i < m_layers.size(); i++)
		{
			//for each neuron
			for (int j = 0; j < m_layers[i].size(); j++)
			{
				output_values.push_back(m_layers[i][j].getOldOutputVal());
			}
		}
		return output_values;
	}


	void ComputeAllNeuronalAverages()
	{
		for (int i = 0; i < m_layers.size(); i++)
		{
			//for each neuron
			for (int j = 0; j < m_layers[i].size(); j++)
			{
				m_layers[i][j].computeAverages();
			}
		}
	}


	vector<double> hiddenLayerWeights;
	vector<double> hiddenLayerOutputs;
	vector<double> hiddenLayerOldOutputs;


	void ComputeUVDerivatives()
	{


		hiddenLayerWeights.clear();
		hiddenLayerOutputs.clear();
		hiddenLayerOldOutputs.clear();

		for (int j = 0; j < m_layers[1].size(); j++)
		{
			m_layers[1][j].UpdateUVHidden(y, y_tilde, y_bar, y_tilde_old, y_bar_old);
			m_layers[1][j].ComputeFDerivative(U, V);
		}

		for (int k = 0; k < m_layers[1].size(); k++)
		{
			hiddenLayerWeights.push_back(m_layers[1][k].m_outputWeights[0].weight);
			hiddenLayerOutputs.push_back(m_layers[1][k].getOutputVal());
			hiddenLayerOldOutputs.push_back(m_layers[1][k].getOldOutputVal());
		}

		//for each neuron
		for (int j = 0; j < m_layers[0].size(); j++)
		{

			m_layers[0][j].UpdateUVInput(y, y_tilde, y_bar,y_tilde_old,y_bar_old,hiddenLayerWeights,hiddenLayerOldOutputs,hiddenLayerOutputs);
			m_layers[0][j].ComputeFDerivative(U, V);
			//		cout<<m_layers[i][j].m_outputWeights[0].F<<endl;
		}
	}

	void UpdateNeuronWeights_bray1996(double learning_rate)
	{
		for (int i = 0; i < m_layers.size(); i++)
		{
			//for each neuron
			for (int j = 0; j < m_layers[i].size(); j++)
			{
				for(int k=0;k<m_layers[i][j].m_outputWeights.size();k++)
					m_layers[i][j].m_outputWeights[k].weight += learning_rate * m_layers[i][j].m_outputWeights[k].DFDW;
			}
		}
	}

	void CalcF();

	void CalcFDerivative();
	double FDeriv;

	double getDUDX(int a);

	double getDVDX(int a);

	double dudw, dvdw;
	double dy_bar, dy_tilde;
	double zjt; //This is effectively the output
	double zkt; //These are hidden layer units


	double y_tilde_old = 0.0;
	double y_bar_old = 0.0;

	void calcIncrementedAverages()
	{
		y_tilde_old = y_tilde;
		y_bar_old = y_bar;
		y_tilde = (lambda_s)*y_tilde + (1.0 - lambda_s) * y;
		y_bar = (lambda_l)*y_bar + (1.0 - lambda_l) * y;
	//	dzdw = y; //PRT maybe take it out
	//	calcDUDW();
	//	calcDVDW();
	}

	double calczjt()
	{
	}



	double getErrorDerivative(double dvdx, double dudx); //technically merit derivative because dFdX = -dEdX

	vector<double> getConvolutionalMaskDashU(double h)
	{
		vector<double> kernel = getConvolutionalMaskU(h);
		int w = getW(h_U);
		kernel[w / 2] = kernel[w / 2] - 1.0;
		return kernel;
	}

	vector<double> getConvolutionalMaskDashV(double h)
	{
		vector<double> kernel = getConvolutionalMaskV(h);
		int w = getW(h_V);
		kernel[w / 2] = kernel[w / 2] - 1.0;
		return kernel;
	}

	vector<double> getConvolutionalMaskU(double h)
	{
		vector<double> kernelU;
		double kernelVal;
		int w = getW(h_U);
		int gs;
		for (int a = 0; a < A; a++)
		{
			kernelVal = 0.0;
			for (int x = -w; x <= w; x++)
			{
				gs = getSub(a, x);
				kernelVal = exp(-getLambdaU(h_U) * (double)(gs) * (double)(gs));
				//		cout<<kernelVal<<endl;
				//		cout<<x<<endl;
			}
			kernelU.push_back(kernelVal);
		}
		normalizeVector(kernelU);
		return kernelU;
	}


	//fix masks
	vector<double> getConvolutionalMaskV(double h)
	{
		vector<double> kernelV;
		double kernelVal;
		int w = getW(h_V);
		int gs;
		for (int a = 0; a < A; a++)
		{
			kernelVal = 0.0;
			for (int x = -w; x <= w; x++)
			{
				gs = getSub(a, x);
				//	cout<<(double)(getSub(a,x))<<endl;
				kernelVal = exp(-getLambdaV(h_V) * (double)(gs) * (double)(gs));
			}
			kernelV.push_back(kernelVal);
		}
		normalizeVector(kernelV);
		return kernelV;
	}

	double getLambdaU(double h)
	{
		double l_u = log(2.0) / h_U;
		return l_u;
	}

	double getLambdaV(double h)
	{
		double l_v = log(2.0) / h_V;
		return l_v;
	}

	int getW(double h)
	{
		int w;
		if (4.0 * h < (A / 2.0 - 1.0))
		{
			w = 4.0 * h;
		}
		else
		{
			w = A / 2.0 - 1.0;
		}
		return w;
	}

	double getF()
	{
		return log(V/U);
	}

	void normalizeVector(vector<double> &vec)
	{
		double sum = 0.0;
		for (int i = 0; i < vec.size(); i++)
		{
			sum += vec[i];
		}
		for (int i = 0; i < vec.size(); i++)
		{
			vec[i] /= sum;
		}
	}

	vector<double> GetAllActivations()
	{
		Layer &outputLayer = m_layers.back();
		Layer &hiddenLayer = m_layers[1];
		Layer &inputLayer = m_layers[0];

		vector<double> ActivationVector;

		//iterate through hidden layers, taking 1st output of 1st layer

		int weightcounter = 0;


		for (int i = 0; i < inputLayer.size(); i++)
		{
			ActivationVector.push_back(inputLayer[i].getOutputVal());
		}
		for (int j = 0; j < hiddenLayer.size(); j++)
		{
			ActivationVector.push_back(hiddenLayer[j].getOutputVal());
		}
		for(int k=0;k<outputLayer.size();k++)
		{
			ActivationVector.push_back(outputLayer[k].getOutputVal());
		}


		return(ActivationVector);

		//iterate through weights from hidden layer to single output
	}

	void CalcAverages();

	int getSub(int a, int k)
	{
		int diff = a - k;
		if (diff < 0.0)
			diff = 1000.0 + diff;
		if (diff > 1000.0)
			diff = diff - 1000.0;

		return diff;
	}

	vector<Layer> m_layers;

private:
	double m_gradient;
	double m_error;
	double m_recentAverageError;
	double m_recentAverageSmoothingFactor;
};

#endif