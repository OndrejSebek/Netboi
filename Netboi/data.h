#ifndef DATA_H_INCLUDED
#define DATA_H_INCLUDED
#include <vector>

using namespace std;

void readData(char * file, vector< vector<double> > & input, int row, int col);
void stand_Data(vector < vector <double> > & input, vector <double> & mean, vector <double> & stdev);
void apply_stand(vector < vector <double> > & input, vector <double> & mean, vector <double> & stdev);
void stand_Data_exp(vector < vector <double> > & exp_out, vector <double> & exp_min, vector <double> & exp_max, double low, double high);
void apply_stand_exp(vector < vector <double> > & exp_out, vector <double> & exp_min, vector <double> & exp_max, double low, double high);


#endif // DATA_H_INCLUDED
