
#include "disparityNet.h"
#include "filesystem.h"





using namespace std;

int NUM_EPOCHS = 2000;

int LOG_RATE = 50;

int main()
{
    vector<unsigned> topology;
    topology.push_back(11); //10 input neurons and 1 bias neuron
    topology.push_back(10);
    topology.push_back(1);
    DisparityNet dNet(topology); //Extracts signel disparity

    string time(GetTimeAsString());
    time.erase(remove(time.begin(), time.end(), ' '), time.end());
    time.erase(remove(time.begin(), time.end(), '\n'), time.end());
    time.erase(remove(time.begin(), time.end(), ':'), time.end());

    bool bShouldSaveGeneratedData = true;
    for (int i = 0; i < NUM_EPOCHS; i++)
    {
        dNet.Train(i, bShouldSaveGeneratedData);
    //    dNet.Train_MakeBackProp(i, bShouldSaveGeneratedData);
        if (i%LOG_RATE == 0)
        {

            if (bShouldSaveGeneratedData)
            {
                ghc::filesystem::create_directories("Logs/"+time+"/"+to_string(i));
                
                write_to_csv(dNet.getNetwork()->Output_Array, "Logs/"+time+"/"+to_string(i)+"/Outputs");
                write_to_csv(dNet.pruned_disp_vector, "Logs/"+time+"/"+to_string(i)+"/Disparity");
                write_to_csv(dNet.getNetwork()->GetWeights(), "Logs/"+time+"/"+to_string(i)+"/Weights");
                write_to_csv(dNet.getNetwork()->GetAllActivations(), "Logs/"+time+"/"+to_string(i)+"/Activations");          
                write_to_csv(dNet.BatchActivations, "Logs/"+time+"/"+to_string(i)+"/Batch_Activations");
                write_to_csv(dNet.BatchInputs,"Logs/"+time+"/"+to_string(i)+"/"+"Batch_Inputs");
                write_to_csv(dNet.BatchWeights,"Logs/"+time+"/"+to_string(i)+"/"+ "/Batch_Weights");
                write_to_csv(dNet.BatchF,"Logs/"+time+"/"+to_string(i)+"/"+ "/Batch_F");
            }
        }
    }
    return 0;
}


//correct for aspect ratio
//


