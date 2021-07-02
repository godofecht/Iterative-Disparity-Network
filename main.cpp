
#include "disparityNet.h"

using namespace std;


int NUM_EPOCHS = 1000000;

int main()
{
    DisparityNet dNet;

    vector<double> outputs;

    for(int i=0;i<NUM_EPOCHS;i++)
    {
        dNet.Train(i);
        if(i %10 == 0)
        {
      //      cout<<i<<endl;
                std::vector<std::pair<std::string,
                std::vector<double>>> vals = {{"Values", dNet.getNetwork()->Output_Array}};
                dNet.write_csv("outputs.csv", vals);
        }

    }




    std::vector<std::pair<std::string,
    std::vector<double>>> valsd = {{"Values", dNet.dispVec_to_write}};
    dNet.write_csv("disparity.csv", valsd);


    std::vector<std::pair<std::string,
    std::vector<double>>> valsd2 = {{"Values", dNet.getNetwork()->GetWeights()}};
    dNet.write_csv("weights.csv", valsd2);


    return 0;
}

void write_to_csv(vector<double> vector,string name)
{
    std::vector<std::pair<std::string,
    std::vector<double>>> vals_temp = {{"Values", vector}};
    //write_csv("name", vals_temp);
}





//correct for aspect ratio
//