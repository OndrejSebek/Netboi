#include <fstream>
#include "data.h"
#include <iostream>
#include <algorithm>
#include "bits/stdc++.h"


void readData(char * file, vector < vector <double> > & input, int row, int col)
{
    ifstream inp;
    inp.open(file);

    input.resize(row);
    for(int i = 0; i < row; i++)
        input[i].resize(col);

    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            inp >> input[i][j];
        }
    }
}

void stand_Data(vector < vector <double> > & input, vector <double> & mean, vector <double> & stdev)
{
    for(int i = 0; i < input.size(); i++)
    {
        for(int j = 0; j < input[0].size(); j++)
        {
            mean[j] += input[i][j];
        }
    }
    for(int i = 0; i < input[0].size(); i++)
        mean[i] /= input.size();


    for(int i = 0; i < input.size(); i++)
    {
        for(int j = 0; j < input[0].size(); j++)
        {
            stdev[j] += (input[i][j] - mean[j])*(input[i][j] - mean[j]);
        }
    }

    for(int i = 0; i < input[0].size(); i++)
    {
        stdev[i] = sqrt(stdev[i] / input.size());
    }


    for(int i = 0; i < input.size(); i++)
    {
        for(int j = 0; j < input[0].size(); j++)
        {
            input[i][j] = (input[i][j]-mean[j])/stdev[j];
        }
    }
}

void apply_stand(vector < vector <double> > & input, vector <double> & mean, vector <double> & stdev)
{
    for(int i = 0; i < input.size(); i++)
    {
        for(int j = 0; j < input[0].size(); j++)
        {
            input[i][j] = (input[i][j]-mean[j])/stdev[j];
        }
    }
}


void stand_Data_exp(vector < vector <double> > & exp_out, vector <double> & exp_min, vector <double> & exp_max, double low, double high)
{
    for(int i = 0; i < exp_out[0].size(); i++)
    {
        exp_min[i] = 1e8;
    }

    for(int i = 0; i < exp_out.size(); i++)
    {
        for(int j = 0; j < exp_out[0].size(); j++)
        {
            if(exp_out[i][j] > exp_max[j])
                exp_max[j] = exp_out[i][j];
            else if(exp_out[i][j] < exp_min[j])
                exp_min[j] = exp_out[i][j];
        }
    }

    for(int i = 0; i < exp_out.size(); i++)
    {
        for(int j = 0; j < exp_out[0].size(); j++)
        {
            exp_out[i][j] = exp_out[i][j] * ( (high-low)/(exp_max[j]-exp_min[j]) ) + ( low*exp_max[j] - high*exp_min[j] )/( exp_max[j]-exp_min[j] );
            //cout << exp_out[i][j] << endl;
        }
    }
}

void apply_stand_exp(vector < vector <double> > & exp_out, vector <double> & exp_min, vector <double> & exp_max, double low, double high)
{
    for(int i = 0; i < exp_out.size(); i++)
    {
        for(int j = 0; j < exp_out[0].size(); j++)
        {
            exp_out[i][j] = exp_out[i][j] * ( (high-low)/(exp_max[j]-exp_min[j]) ) + ( low*exp_max[j] - high*exp_min[j] )/( exp_max[j]-exp_min[j] );
            //cout << exp_out[i][j] << endl;
        }
    }
}















