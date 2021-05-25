
#include "disparityNet.h"

using namespace std;


int NUM_EPOCHS = 100;

int main()
{
    DisparityNet dNet;

    vector<double> outputs;

    for(int i=0;i<NUM_EPOCHS;i++)
    {
     //   cout<<i<<endl;
        dNet.Train(i);
    }


    std::vector<std::pair<std::string,
    std::vector<double>>> vals = {{"Values", dNet.getNetwork()->Output_Array}};
    dNet.write_csv("outputs.csv", vals);

    std::vector<std::pair<std::string,
    std::vector<double>>> valsd = {{"Values", dNet.disparityVector}};
    dNet.write_csv("disparity.csv", valsd);


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