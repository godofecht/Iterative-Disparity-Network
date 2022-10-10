#pragma once

#include <iostream>


#ifdef __APPLE__
        #include <sys/uio.h>
#else
	#include <io.h>
#endif




#include <sys/types.h>
#include <sys/stat.h>
#include <ctime>


#define PRINT(name) print(#name, (name))



char* GetTimeAsString()
{
    time_t now = time(0);
    char* dt = ctime(&now);
    tm *gmtm = gmtime(&now);
    dt = asctime(gmtm);
    return(dt);
}


void createFolder(const char* name)
{
    mkdir (name, 0777);
}

void print(const char *name, double value)
{
    std::cout << name << " " << value << std::endl;
}

void write_csv(std::string filename, std::vector<std::pair<std::string, std::vector<double>>> dataset)
{
    // Make a CSV file with one or more columns of integer values
    // Each column of data is represented by the pair <column name, column data>
    //   as std::pair<std::string, std::vector<int>>
    // The dataset is represented as a vector of these columns
    // Note that all columns should be the same size

    // Create an output filestream object
    std::ofstream myFile = std::ofstream(filename);

    // Send column names to the stream
    for (int j = 0; j < dataset.size(); ++j)
    {
        myFile << dataset.at(j).first;
        if (j != dataset.size() - 1)
            myFile << ","; // No comma at end of line
    }
    myFile << "\n";

    // Send data to the stream
    for (int i = 0; i < dataset.at(0).second.size(); ++i)
    {
        for (int j = 0; j < dataset.size(); ++j)
        {
            myFile << dataset.at(j).second.at(i);
            if (j != dataset.size() - 1)
                myFile << ","; // No comma at end of line
        }
        myFile << "\n";
    }

    // Close the file
    myFile.close();
}





void write_to_csv(vector<vector<double>> vector, string name)
{
    std::vector<std::pair<std::string, std::vector<double>>> vals;
    for (int it = 0; it < vector.size(); it++)
    {
        std::pair<std::string, std::vector<double>> vdsl2;
        vdsl2 = {to_string(it), vector[it]};
        vals.push_back(vdsl2);
    }
    write_csv(name + ".csv", vals);
}

void write_to_csv(vector<double> vector, string name)
{
    std::vector<std::pair<std::string, std::vector<double>>>
    vals;
    std::pair<std::string, std::vector<double>> vdsl2;
    vdsl2 = {"Values", vector};
    vals.push_back(vdsl2);
    write_csv(name + ".csv", vals);
}