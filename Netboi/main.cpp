#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cmath>
#include <vector>

#include "mtx.h"
#include "net.h"
#include "data.h"


int main()
{
/*
    testFun();
    return 0;
*/

/*
    !clean code
            comp tanh(-0.9, 0.9) x sigm(0.1, 0.9)


            momentum Mtx pointers
*/

/*
        CAL
*/


    int row = 34, col = 9, col_exp = 1;

    vector< vector<double> > inp;
    vector< vector<double> > exp_out;

    vector< vector<double> > inp_val;
    vector< vector<double> > exp_out_val;

    readData("inp/INP.txt", inp, row, col);
    readData("inp/EXP_OUT.txt", exp_out, row, col_exp);

    int row_val = 34;

    readData("inp/INP.txt", inp_val, row_val, col);
    readData("inp/EXP_OUT.txt", exp_out_val, row_val, col_exp);

    // Stand

    // mean = 0, stdev = 1
    stand_Data(inp);
    stand_Data(inp_val);


    vector <double> exp_max(exp_out.size());
    vector <double> exp_min(exp_out.size());

    // Target values
    double low = 0.1;
    double high = 0.9;

    stand_Data_exp(exp_out, exp_min, exp_max, low, high);
    stand_Data_exp(exp_out_val, exp_min, exp_max, low, high);



/*
        NET   (shuffling > ff > backprop > update)
*/


    vector<int> neurons = {col, 100, col_exp};
    double l_r = 0.1, m_l_r = 0.05;

    p_double trans_fun = sigmoid;
    p_double trans_fun_d = sigmoid_d;

//
    Net net(neurons, l_r, m_l_r);                                                                               // {neurons}, lr, mlr
//
//
//  net.learn(100, 100, inp, exp_out, inp_val, exp_out_val, trans_fun, trans_fun_d);                              // n_epochs, n_eval, inp, exp_out, trans_fun, trans_fun_d


  //  Net net("out/8/100k/NETWORKPARAMS.txt");
/*
    net.inp_size = row;
    net.inp0_size = col;
    net.inp_val_size = row_val;
    net.exp_out0_size = col_exp;
*/

    net.learn(1000000, 10000, inp, exp_out, inp_val, exp_out_val, trans_fun, trans_fun_d);


/*
    //net.epoch = 0;

    //net.learning_rate = 0.01;
    //net.m_learning_rate = 0.005;


    //cout << net.learning_rate << "  " << net.m_learning_rate << " " << net.epoch << endl;
*/




/*
        VAL
*/


    ofstream outf;
    outf.open("out/out_cal.txt");

    Mtx out(1, 1);
    for(int i = 0; i < inp.size(); i++)
    {
        net.feedforward(inp[i], trans_fun);
        out.mtx_load(net.H[2].array[0], 1);                     // !! H[2] = res
        out.destand(exp_min, exp_max, low, high);

        outf << out;
    }

    ofstream outf_v;
    outf_v.open("out/out_val.txt");

    for(int i = 0; i < inp_val.size(); i++)
    {
        net.feedforward(inp_val[i], trans_fun);
        out.mtx_load(net.H[2].array[0], 1);
        out.destand(exp_min, exp_max, low, high);

        outf_v << out;
    }


    net.saveNetworkParams("out/NETWORKPARAMS.txt");
/*
*/





    return 0;
}
