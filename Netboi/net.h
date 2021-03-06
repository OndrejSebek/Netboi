#ifndef NET_H_INCLUDED
#define NET_H_INCLUDED

#include "mtx.h"

using namespace std;

typedef double (*p_double)(double x);

class Net
{
public:
    Net(vector<int> const &neurons, double const &learning_rate, double const &m_learning_rate);
    Net(const char *filepath);

    void load(char * INP, char * EXP_OUT, char * INP_VAL, char * EXP_OUT_VAL, int row, int row_val, double low, double high, int transfun_opt);
    void load_inp(char * INP, int row);

    void feedforward(vector<double> const &inp, p_double trans_fun);

    void backprop(vector <double> const &exp_out, p_double trans_fun_d);

    void update_weights();
    void update_weights_momentum(vector < Mtx > &dEdW_prev, vector < Mtx > &dEdB_prev);

    void learn(int n_epochs, int n_eval);
    void learn_batch(int n_epochs, int n_eval, vector< vector<double> > &inp, vector< vector<double> > &exp_out, vector< vector<double> > &inp_val, vector< vector<double> > &exp_out_val, p_double trans_fun, p_double trans_fun_d);
    void learn_momentum(int n_epochs, int n_eval, vector< vector<double> > &inp, vector< vector<double> > &exp_out, vector< vector<double> > &inp_val, vector< vector<double> > &exp_out_val, p_double trans_fun, p_double trans_fun_d);

    void dropout(int n_dropout);

    void printToFile(Mtx &m, ostream &file);
    void saveNetworkParams(const char *outf_path);

    void print_res(char * file, int res_opt);


//private:

    vector <int> n_weights;
    int total_n_weights;

    vector< Mtx > W;
    vector< Mtx > B;


    vector< Mtx > H;
    vector< Mtx > dEdW;
    vector< Mtx > dEdB;

    Mtx Y;

    Mtx bufferMtx;

    int neurons_max;
    vector <int> neurons;
    double learning_rate;
    double m_learning_rate;
    int n_hidden;

    int epoch;

    vector <int> shuffling_ind;

    // inps
    int row;
    int row_val;
    int col;
    int col_exp;

    vector< vector<double> > inp;
    vector< vector<double> > exp_out;

    vector< vector<double> > inp_val;
    vector< vector<double> > exp_out_val;

    vector <double> exp_max;
    vector <double> exp_min;

    double low;
    double high;

    int transfun_opt;

    p_double trans_fun;
    p_double trans_fun_d;

    int inp_size;
    int inp0_size;
    int inp_val_size;
    int exp_out0_size;

    vector <double> stdev;
    vector <double> mean;
};

double sigmoid(double x);

double sigmoid_d(double x);

double random_w(double x);

double tanhf(double x);

double tanh_d(double x);

double relu(double x);

double relu_d(double x);

double softplus(double x);

double softplus_d(double x);

void Hello();

int testFun();

#endif // NET_H_INCLUDED
