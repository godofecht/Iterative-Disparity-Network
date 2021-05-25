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

	void backPropagate(const vector<double> &targetVals)
	{
		// Calculate overall net error (RMS of output neuron errors)


		Layer &outputLayer = m_layers.back();
		Layer &hiddenLayer = m_layers[1];
		Layer &inputLayer = m_layers[0];
		m_error = 0.0;

		double delta = targetVals[0] - outputLayer[0].getOutputVal();
		m_error += delta * delta;

		m_error = sqrt(m_error);		   // RMS
	//	cout<<outputLayer[0].getOutputVal()<<endl;

		// Implement a recent average measurement

		m_recentAverageError =
			(m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);

		// Calculate output layer gradients



		// Calculate output layer gradients
		for (unsigned n = 0; n < outputLayer.size(); n++)
		{
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

	double dzdw = 0.0f;
	double dztdw = 0.0f;
	double dzbdw = 0.0f;

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
		if (isnan(y_bar))
			y_bar = 0.0;
		if (isnan(y))
			y = 0.0;
		double vdiff = 0.5 * pow(y_bar - y, 2.0);
		if (isnan(vdiff))
			vdiff = 0.0;
		V += vdiff;
	}

	void CalcU()
	{
		double udiff = 0.5 * pow(y_tilde - y, 2.0);

		U += udiff;
	}

	void Calcdztdw()
	{
		dztdw = lambda_s * dztdw + (1.0 - lambda_s) * y;
	}

	void Calcdzbdw()
	{
		dzbdw = lambda_l * dzbdw + (1.0 - lambda_l) * y;
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

	void calcIncrementedAverages()
	{
		y_tilde = (lambda_s)*y_tilde + (1.0 - lambda_s) * y;
		y_bar = (lambda_l)*y_bar + (1.0 - lambda_l) * y;
		dzdw = y; //PRT maybe take it out
		calcDUDW();
		calcDVDW();
	}

	double calczjt()
	{
	}

	double calcDUDW()
	{
		dudw += (y_tilde - y) * (dztdw - dzdw);
	}

	double calcDVDW()
	{
		dvdw += (y_bar - y) * (dzbdw - dzdw);
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