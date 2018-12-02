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
            comp tanh(-0.9, 0.9) x sigm(0.1, 0.9)
            momentum Mtx pointers
*/

/*
        CAL
*/
    vector<int> neurons = {9, 8, 1};
    double l_r = 0.1, m_l_r = 0.05;


    Net net({9, 8, 1}, l_r, m_l_r);
    net.load("inp/INP.txt", "inp/EXP_OUT.txt", "inp/INP.txt", "inp/EXP_OUT.txt", 34, 34, 0.1, 0.9, sigmoid, sigmoid_d);

/*
        NET   (shuffling > ff > backprop > update)
*/


                                                                                          // {neurons}, lr, mlr


    //Net net("out/NETWORKPARAMS.txt");
/*
    net.inp_size = row;
    net.inp0_size = col;
    net.inp_val_size = row_val;
    net.exp_out0_size = col_exp;
*/

    net.learn(10000, 1000);                                 // n_epochs, n_eval, inp, exp_out, trans_fun, trans_fun_d


/*
    //net.epoch = 0;
    //net.learning_rate = 0.01;
    //net.m_learning_rate = 0.005;
    //cout << net.learning_rate << "  " << net.m_learning_rate << " " << net.epoch << endl;
*/



/*
        VAL
*/

    net.print_res("out_cal.txt");
    net.print_res("out_val.txt");

    net.saveNetworkParams("out/NETWORKPARAMS.txt");
/*
*/





    return 0;
}
