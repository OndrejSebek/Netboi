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
    vector<int> neurons = {9, 10, 1};
    double l_r = 0.1, m_l_r = 0.05;


    Net net(neurons, l_r, m_l_r);
    //Net net("out/NETWORKPARAMS.txt");

    net.load("inp/INP.txt", "inp/EXP_OUT.txt", "inp/INP_VAL.txt", "inp/EXP_OUT_VAL.txt", 34, 2, 0.1, 0.9, 1);                // INP | EXP_OUT | INP_VAL | EXP_OUT_VAL | row | row_val | low | high | transfun_opt
    //net.load_inp("inp/INP.txt", 34);                                                                                      // INP | row

/*
        NET   (shuffling > ff > backprop > update)
*/

/*
    net.inp_size = row;
    net.inp0_size = col;
    net.inp_val_size = row_val;
    net.exp_out0_size = col_exp;
*/

    net.learn(50000, 5000);                                                                                           // n_epochs, n_eval


/*
        VAL
*/

    net.print_res("out/out_cal.txt", 1);
    net.print_res("out/out_val.txt", 2);

    net.saveNetworkParams("out/NETWORKPARAMS.txt");




    return 0;
}
