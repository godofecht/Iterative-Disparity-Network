
#include "disparityNet.h"

using namespace std;

int NUM_EPOCHS = 100000;

int main()
{
    DisparityNet dNet;

    vector<double> outputs;

    std::vector<std::pair<std::string, std::vector<double>>> valsd4;

    bool bShouldSaveGeneratedData = true;

    for (int i = 0; i < NUM_EPOCHS; i++)
    {
   //     dNet.Train(i, bShouldSaveGeneratedData);
        dNet.Train_MakeBackProp(i);
        if (i % 10 == 0)
        {

            if (bShouldSaveGeneratedData)
            {
                //      cout<<i<<endl;
                std::vector<std::pair<std::string,
                                      std::vector<double>>>
                    vals = {{"Values", dNet.getNetwork()->Output_Array}};
                dNet.write_csv("outputs.csv", vals);

                std::vector<std::pair<std::string,
                                      std::vector<double>>>
                    valsd = {{"Values", dNet.pruned_disp_vector}};
                dNet.write_csv("disparity.csv", valsd);

                std::vector<std::pair<std::string,
                                      std::vector<double>>>
                    valsd2 = {{"Values", dNet.getNetwork()->GetWeights()}};
                dNet.write_csv("weights.csv", valsd2);

                std::vector<std::pair<std::string, std::vector<double>>>
                    valsd3 = {{"Values", dNet.getNetwork()->GetAllActivations()}};
                dNet.write_csv("activations.csv", valsd3);

                std::vector<std::pair<std::string, std::vector<double>>> valsd4;
                for (int it = 0; it < 1000; it++)
                {
                    std::pair<std::string, std::vector<double>> vds;
                    vds = {to_string(it), dNet.BatchActivations[it]};
                    valsd4.push_back(vds);
                }
                dNet.write_csv("batch_activations.csv", valsd4);

                std::vector<std::pair<std::string, std::vector<double>>> valsd5;
                for (int it = 0; it < 1000; it++)
                {
                    std::pair<std::string, std::vector<double>> vdsl;
                    vdsl = {to_string(it), dNet.BatchDisparity[it]};

                    valsd5.push_back(vdsl);
                }
                dNet.write_csv("batch_disparity.csv", valsd5);

                std::vector<std::pair<std::string, std::vector<double>>> valsd6;
                for (int it = 0; it < 1000; it++)
                {
                    std::pair<std::string, std::vector<double>> vdsl;
                    vdsl = {to_string(it), dNet.BatchInputs[it]};
                    valsd6.push_back(vdsl);
                }
                dNet.write_csv("batch_inputs.csv", valsd6);
            }
        }
    }

    return 0;
}

void write_to_csv(vector<double> vector, string name)
{
    std::vector<std::pair<std::string,
                          std::vector<double>>>
        vals_temp = {{"Values", vector}};
    //write_csv("name", vals_temp);
}

//correct for aspect ratio
//