#include "net.h"
#include "data.h"
#include <math.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <algorithm>



// CLASS NET

/** \brief                                      Constructor
 *
 * \param neurons vector<int>                   topology
 * \param learning_rate double                  learning rate
 *
 */
Net::Net(vector<int> const &neurons, double const &learning_rate, double const &m_learning_rate)
{
    srand (time(NULL));

    this->learning_rate = learning_rate;
    this->m_learning_rate = m_learning_rate;
    this->n_hidden = neurons.size() - 2;
    this->neurons = neurons;

    this->col = neurons[0];
    this->col_exp = neurons[neurons.size()-1];

    H = vector< Mtx >(n_hidden+2);
    W = vector< Mtx >(n_hidden+1);
    B = vector< Mtx >(n_hidden+1);
    dEdW = vector< Mtx >(n_hidden+1);
    dEdB = vector< Mtx >(n_hidden+1);

    n_weights = vector <int> (n_hidden+1);
    neurons_max = 0;


    // ALLOC

    for(int i = 0; i < n_hidden+1; i++)
    {
        H[i] = Mtx(1, neurons[i]);

        W[i] = Mtx(neurons[i], neurons[i+1]);
        dEdW[i] = Mtx(neurons[i], neurons[i+1]);

        B[i] = Mtx(1, neurons[i+1]);
        dEdB[i] = Mtx(1, neurons[i+1]);

        W[i].applyFunction(random_w);
        B[i].applyFunction(random_w);

        n_weights[i] = neurons[i] * neurons[i+1];

        if(neurons[i] > neurons_max)
            neurons_max = neurons[i];
    }

    H[n_hidden+1] = Mtx(1, neurons[n_hidden+1]);

    bufferMtx = Mtx(neurons_max, neurons_max);

    Y = Mtx(1, 1);

    epoch = 0;
}


/** \brief                                      Feedforward pass
 *
 * \param inp vector<double>                    input
 * \return Mtx                      returns     last layer
 *
 */
void Net::feedforward(vector<double> const &inp, p_double trans_fun)
{
    H[0].mtx_load(inp, col);

    for(int i = 1; i < n_hidden+2; i++)         //  H[i] = trans_fun{H[i-1] * (W[i-1]) + (B[i-1])};
    {
        H[i-1].dot(W[i-1], H[i]);

        H[i].add(B[i-1]);
        H[i].applyFunction(trans_fun);
    }
}

/** \brief                                      Backprop
 *
 * \param exp_out vector<double>                expected output
 * \return void
 *
 */
void Net::backprop(vector <double> const &exp_out, p_double trans_fun_d)              // Error E = 1/2 (exp_out - computedOutput)^2
{
    Y.mtx_load(exp_out, exp_out0_size);

    // gradients
    dEdB[n_hidden].mtx_copy(H[n_hidden+1]);            //  dEdB[n_hidden] = H[n_hidden+1] - (Y) * trans_fun_d{( H[n_hidden] * (W[n_hidden]) + (B[n_hidden]) };
    dEdB[n_hidden].subtract(Y);

    H[n_hidden].dot(W[n_hidden], bufferMtx);

    bufferMtx.add(B[n_hidden]);
    bufferMtx.applyFunction(trans_fun_d);

    dEdB[n_hidden].multiply(bufferMtx);


    for(int i = n_hidden-1; i >= 0; i--)            //  dEdB[i] = EdB[i+1] * W[i+1]T * trans_fun_d{( H[i] * (W[i]) + (B[i]) };
    {
        bufferMtx.transpose(W[i+1]);

        dEdB[i+1].dot(bufferMtx, dEdB[i]);

        H[i].dot(W[i], bufferMtx);

        bufferMtx.add(B[i]);
        bufferMtx.applyFunction(trans_fun_d);

        dEdB[i].multiply(bufferMtx);
    }


    for(int i = 0; i < n_hidden+1; i++)             // dEdW[i] = H[i].transpose().dot(dEdB[i]);
    {
        bufferMtx.transpose(H[i]);
        bufferMtx.dot(dEdB[i], dEdW[i]);
    }
}


/** \brief                                      Update weights without momentum
 *
 * \return void
 *
 */
void Net::update_weights()
{
    // learning
    for(int i = 0; i < n_hidden+1; i++)
    {
        bufferMtx.mtx_copy(dEdW[i]);
        bufferMtx.multiply(learning_rate);
        W[i].subtract(bufferMtx);

        bufferMtx.mtx_copy(dEdB[i]);
        bufferMtx.multiply(learning_rate);
        B[i].subtract(bufferMtx);

    }
}


/** \brief                                      Update weights with momentum
 *
 * \param vector < Mtx > &dEdW_prev             input
 * \param vector < Mtx > &dEdB_prev             expected output
 * \param mlr double                            momentum learning rate
 * \return void
 *
 */
void Net::update_weights_momentum(vector < Mtx > &dEdW_prev, vector < Mtx > &dEdB_prev)
{
    // learning
    for(int i = 0; i < n_hidden+1; i++)
    {
        bufferMtx.mtx_copy(dEdW[i]);
        bufferMtx.multiply(learning_rate);
        W[i].subtract(bufferMtx);

        bufferMtx.mtx_copy(dEdW_prev[i]);
        bufferMtx.multiply(m_learning_rate);
        W[i].subtract(bufferMtx);

        bufferMtx.mtx_copy(dEdB[i]);
        bufferMtx.multiply(learning_rate);
        B[i].subtract(bufferMtx);

        bufferMtx.mtx_copy(dEdB_prev[i]);
        bufferMtx.multiply(m_learning_rate);
        B[i].subtract(bufferMtx);
    }
}



/** \brief                                      Online without momentum
 *
 * \param error_crit double                     net error, stopping crit
 * \param &inp vector< vector<double>>          input
 * \param &exp_out vector< vector<double>>      expected output
 * \return void
 *
 */
void Net::learn(int n_epochs, int n_eval)     // while -> for, error pri i%xxx == 0
{
    Hello();

    inp_size = inp.size();
    inp0_size = inp[0].size();
    inp_val_size = inp_val.size();
    exp_out0_size = exp_out[0].size();


    double error, error_val;

    // shuffling
    shuffling_ind = vector <int> (inp_size);
    for(unsigned int i = 0; i < inp_size; i++)
        shuffling_ind[i] = i;

    while(epoch < n_epochs)
    {
        if(epoch%n_eval==0)
        {
            error = 0;
            error_val = 0;
        }

        // shuffling
        random_shuffle(shuffling_ind.begin(), shuffling_ind.end());
/*
        if (epoch%10 == 0)
            dropout(1);
*/
        for(unsigned int j = 0; j < inp_size; j++)
        {
            feedforward(inp[shuffling_ind[j]], trans_fun);     // take inp with index from randomly shuffled array
            //cout << exp_out[shuffling_ind[j]][0];
            //getchar();
            backprop(exp_out[shuffling_ind[j]], trans_fun_d);
            update_weights();


            if(epoch%n_eval == 0)
            {
                //error += pow((H[n_hidden+1].array[0][0] - Y.array[0][0]), 2)   ;    //H[n_hidden+1].subtract(Y);
                error += (H[n_hidden+1].array[0][0] - Y.array[0][0])*(H[n_hidden+1].array[0][0] - Y.array[0][0])   ;

                // ff inp_val
                if(j < inp_val_size)
                {
                    feedforward(inp_val[j], trans_fun);                                                                                             // j not enough if |inp| < |inp_val|
                    error_val += (H[n_hidden+1].array[0][0] - exp_out_val[j][0])*(H[n_hidden+1].array[0][0] - exp_out_val[j][0])   ;
                }

                if(j == inp_size-1) cout << "epoch:  " << epoch << "     cal:  " << error << "    val:  " << error_val << endl;
            }


        }

        epoch++;
    }
}

/** \brief                                      Batch without momentum
 *
 * \param error_crit double                     net error, stopping crit
 * \param &inp vector< vector<double>>          input
 * \param &exp_out vector< vector<double>>      expected output
 * \return void
 *
 */
void Net::learn_batch(int n_epochs, int n_eval, vector< vector<double> > &inp, vector< vector<double> > &exp_out, vector< vector<double> > &inp_val, vector< vector<double> > &exp_out_val, p_double trans_fun, p_double trans_fun_d)
{
    Hello();

    inp_size = inp.size();
    inp0_size = inp[0].size();
    inp_val_size = inp_val.size();
    exp_out0_size = exp_out[0].size();

    double error, error_val;
    int i = 0;

    // shuffling
    shuffling_ind = vector <int> (inp.size());
    for(int i = 0; i < inp.size(); i++)
        shuffling_ind[i] = i;

    while(epoch < n_epochs)
    {
        i++;
        if(i%n_eval==0)
        {
            error = 0;
            error_val = 0;
        }

        // shuffling
        random_shuffle(shuffling_ind.begin(), shuffling_ind.end());

        // reset grads
        for(int k = 0; k < dEdW.size(); k++)              // performance?
        {
            dEdW[k].multiply(0);
            dEdW[k].multiply(0);
        }


        for(int j = 0; j < inp_size; j++)
        {
            feedforward(inp[shuffling_ind[j]], trans_fun);
            backprop(exp_out[shuffling_ind[j]], trans_fun_d);

            if(i%n_eval == 0)
            {
                //error += pow((H[n_hidden+1].array[0][0] - Y.array[0][0]), 2)   ;    //H[n_hidden+1].subtract(Y);
                error += (H[n_hidden+1].array[0][0] - Y.array[0][0])*(H[n_hidden+1].array[0][0] - Y.array[0][0])   ;

                // ff inp_val
                if(j < inp_val_size)
                {
                    feedforward(inp_val[j], trans_fun);                                                                                             // j not enough if |inp| < |inp_val|
                    error_val += (H[n_hidden+1].array[0][0] - exp_out_val[j][0])*(H[n_hidden+1].array[0][0] - exp_out_val[j][0])   ;
                }

                if(j == inp_size-1) cout << "epoch:  " << epoch+1 << "     cal:  " << error << "    val:  " << error_val << endl;
            }
        }

        // update after every batch
        update_weights();
        epoch++;
    }


}



/** \brief                                      Online with momentum
 *
 * \param error_crit double                     net error, stopping crit
 * \param &inp vector< vector<double>>          input
 * \param &exp_out vector< vector<double>>      expected output
 * \return void
 *
 */
void Net::learn_momentum(int n_epochs, int n_eval, vector< vector<double> > &inp, vector< vector<double> > &exp_out, vector< vector<double> > &inp_val, vector< vector<double> > &exp_out_val, p_double trans_fun, p_double trans_fun_d)     // while -> for, error pri i%xxx == 0
{
    Hello();

    inp_size = inp.size();
    inp0_size = inp[0].size();
    inp_val_size = inp_val.size();
    exp_out0_size = exp_out[0].size();

    double error, error_val;
    int i = 0;

    // shuffling
    shuffling_ind = vector <int> (inp.size());
    for(int i = 0; i < inp.size(); i++)
        shuffling_ind[i] = i;


    // 1st ff to set dEdW, dEdB (easy)
    feedforward(inp[0], trans_fun);
    backprop(exp_out[0], trans_fun_d);


    // dEdW_prev, dEdB_prev for momentum, prev grad
    vector < Mtx > dEdW_prev(n_hidden+1);
    vector < Mtx > dEdB_prev(n_hidden+1);

    for(int i = 0; i < n_hidden+1; i++)
    {
        dEdW_prev[i] = Mtx(neurons[i], neurons[i+1]);
        dEdB_prev[i] = Mtx(1, neurons[i+1]);
    }

    for(int k = 0; k < n_hidden+1; k++)
    {
        dEdW_prev[k].mtx_copy(dEdW[k]);
        dEdB_prev[k].mtx_copy(dEdB[k]);
    }

    while(epoch < n_epochs)
    {
        // iteration counter
        i++;
        // reseting error
        if(i%n_eval==0)
        {
            error = 0;
            error_val = 0;
        }
        // shuffling
        random_shuffle(shuffling_ind.begin(), shuffling_ind.end());

        for(int j = 0; j < inp.size(); j++)
        {
            feedforward(inp[shuffling_ind[j]], trans_fun);

            // backprop sets new dEdW, dEdB
            backprop(exp_out[shuffling_ind[j]], trans_fun_d);

            // update weights with current and prev gradients
            update_weights_momentum(dEdW_prev, dEdB_prev);


            // store old grad for next iteration
            for(int k = 0; k < n_hidden+1; k++)
            {
                dEdW_prev[k].mtx_copy(dEdW[k]);
                dEdB_prev[k].mtx_copy(dEdB[k]);
            }


            // testing end crit
            if(i%n_eval == 0)
            {
                //error += pow((H[n_hidden+1].array[0][0] - Y.array[0][0]), 2)   ;    //H[n_hidden+1].subtract(Y);
                error += (H[n_hidden+1].array[0][0] - Y.array[0][0])*(H[n_hidden+1].array[0][0] - Y.array[0][0])   ;

                // ff inp_val
                if(j < inp_val_size)
                {
                    feedforward(inp_val[j], trans_fun);                                                                                             // j not enough if |inp| < |inp_val|
                    error_val += (H[n_hidden+1].array[0][0] - exp_out_val[j][0])*(H[n_hidden+1].array[0][0] - exp_out_val[j][0])   ;
                }

                if(j == inp_size-1) cout << "epoch:  " << epoch+1 << "     cal:  " << error << "    val:  " << error_val << endl;
            }
        }

        epoch++;
    }
}


/** \brief                              Print Mtx to file
 *
 * \param m Mtx&                        Mtx
 * \param outf ostream&                 path
 * \return void
 *
 */
void Net::printToFile(Mtx &m, ostream &outf)
{
    outf << m.row << endl;
    outf << m.col << endl;

    outf.precision(17);

    for (int i = 0; i < m.row; i++)
    {
        for (int j = 0; j < m.col; j++)
        {
            outf << m.array[i][j] << (j != m.col-1 ? " " : "");
        }
        outf << endl;
    }
}


/** \brief                              Save network parameters, weights + biases
 *
 * \param outf_path const char*         path
 * \return void
 *
 */
void Net::saveNetworkParams(const char *outf_path)
{
    ofstream out(outf_path);
    out.precision(17);

    out << n_hidden << endl;

    for(int i = 0; i < n_hidden+2; i++)
        out << neurons[i] << " ";

    out << endl << neurons_max;
    out << endl << learning_rate << endl;
    out << m_learning_rate << endl;
    out << epoch << endl;



    for (Mtx m : W){
        printToFile(m, out);
    }

    for (Mtx m : B){
        printToFile(m, out);
    }

/*
    for (Mtx m : H){
        printToFile(m, out);
    }

    for (Mtx m : dEdB){
        printToFile(m, out);
    }

    for (Mtx m : dEdW){
        printToFile(m, out);
    }
*/
    for(int i = 0; i < col_exp; i++)
        out << exp_max[i] << " ";

    out << endl;
    for(int i = 0; i < col_exp; i++)
        out << exp_min[i] << " ";

    out << endl << high;
    out << endl << low;

    out << endl << transfun_opt;

    out << endl;
    for(int i = 0; i < col; i++)
        out << mean[i] << " ";

    out << endl;
    for(int i = 0; i < col; i++)
        out << stdev[i] << " ";



    out.close();
}


/** \brief                              Load constructor
 *
 * \param filepath const char*          path
 *
 */
Net::Net(const char *filepath)
{
    srand(time(NULL));
    ifstream in(filepath);

    int n_row, n_col;

    if(in)
    {
        in >> n_hidden;

        neurons = vector <int>(n_hidden+2);

        for(int i = 0; i < n_hidden+2; i++)
            in >> neurons[i];

        in >> neurons_max;
        in >> learning_rate;
        in >> m_learning_rate;
        in >> epoch;

        this->col = neurons[0];
        this->col_exp = neurons[neurons.size()-1];

        H = vector< Mtx >(n_hidden+2);
        W = vector< Mtx >(n_hidden+1);
        B = vector< Mtx >(n_hidden+1);
        dEdW = vector< Mtx >(n_hidden+1);
        dEdB = vector< Mtx >(n_hidden+1);

        n_weights = vector <int> (n_hidden+1);


        // ALLOC

        for(int i = 0; i < n_hidden+1; i++)
        {
            H[i] = Mtx(1, neurons[i]);

            W[i] = Mtx(neurons[i], neurons[i+1]);
            B[i] = Mtx(1, neurons[i+1]);

            dEdW[i] = Mtx(neurons[i], neurons[i+1]);
            dEdB[i] = Mtx(1, neurons[i+1]);

            W[i].applyFunction(random_w);
            B[i].applyFunction(random_w);
            n_weights[i] = neurons[i] * neurons[i+1];

        }

        H[n_hidden+1] = Mtx(1, neurons[n_hidden+1]);

        bufferMtx = Mtx(neurons_max, neurons_max);

        Y = Mtx(1, 1);


        // VALS

        for(int i = 0; i < n_hidden+1; i++)
        {
            in >> n_row;
            in >> n_col;

            for(int j = 0; j < n_row; j++)
            {
                for(int k = 0; k < n_col; k++)
                {
                    in >> W[i].array[j][k];
                }
            }
        }

        for(int i = 0; i < n_hidden+1; i++)
        {
            in >> n_row;
            in >> n_col;

            for(int j = 0; j < n_row; j++)
            {
                for(int k = 0; k < n_col; k++)
                {
                    in >> B[i].array[j][k];
                }
            }
        }

        exp_max.resize(col_exp);
        exp_min.resize(col_exp);

        for(int i = 0; i < col_exp; i++)
            in >> exp_max[i];

        for(int i = 0; i < col_exp; i++)
            in >> exp_min[i];

        in >> high;
        in >> low;

        in >> transfun_opt;
        if(transfun_opt == 1)
        {
            trans_fun = sigmoid;
            trans_fun_d = sigmoid_d;
        }
        else if(transfun_opt == 2)
        {
            trans_fun = tanhf;
            trans_fun_d = tanh_d;
        }


        stdev.resize(col);
        mean.resize(col);
        for(int i = 0; i < col; i++)
            in >> mean[i];

        for(int i = 0; i < col; i++)
            in >> stdev[i];
/*
*/
        in.close();
    }

}

/** \brief                              Weight dropout
 *
 * \param n_dropout int                 number of weights to reset
 * \return void
 *
 */
void Net::dropout(int n_dropout)
{
    total_n_weights = 0;

    // vector<int> neurons
    for(unsigned int i = 0; i < n_weights.size(); i++)
        total_n_weights += n_weights[i];

    int target;
    int n_Mtx = 0;

    int tar_row, tar_col;

    for(int i = 0; i < n_dropout; i++)
    {
        // target random weight
        target = rand() % total_n_weights + 1;

        // find the Mtx the target is in
        while(target - n_weights[n_Mtx] > 0)
        {
            target -= n_weights[n_Mtx];
            n_Mtx++;
        }

        // set targets correctly
        target--;
        tar_row = target / W[n_Mtx].col;
        tar_col = target % W[n_Mtx].col;

        // reset with random value
        W[n_Mtx].array[tar_row][tar_col] = random_w(0);

        n_Mtx = 0;
    }
}


void Net::print_res(char * file, int res_opt)
{
    ofstream outf;
    outf.open(file);
    outf.precision(17);

    Mtx out(1, 1);

    if(res_opt == 1)
    {
        for(int i = 0; i < inp.size(); i++)
        {
            feedforward(inp[i], trans_fun);
            out.mtx_load(H[neurons.size()-1].array[0], 1);                                                                  // H[last] .. 1 neuron
            out.destand(exp_min, exp_max, low, high);

            outf << out;
        }
    }
    else
    {
        for(int i = 0; i < inp_val.size(); i++)
        {
            feedforward(inp_val[i], trans_fun);
            out.mtx_load(H[neurons.size()-1].array[0], 1);                                                                  // H[last] .. 1 neuron
            out.destand(exp_min, exp_max, low, high);

            outf << out;
        }
    }
/*
    for(int i = 0; i < inp.size(); i++)
    {
        for(int j = 0; j < inp[0].size(); j++)
        {
            outf << inp[i][j] << " ";
        }
        outf << endl;
    }
*/
}


void Net::load(char * INP, char * EXP_OUT, char * INP_VAL, char * EXP_OUT_VAL, int row, int row_val, double low, double high, int transfun_opt)
{
    this->row = row;
    this->row_val = row_val;

    // Target values
    this->high = high;
    this->low = low;
    this->transfun_opt = transfun_opt;

    if(transfun_opt == 1)
    {
        trans_fun = sigmoid;
        trans_fun_d = sigmoid_d;
    }
    else if(transfun_opt == 2)
    {
        trans_fun = tanhf;
        trans_fun_d = tanh_d;

    }

    readData(INP, inp, row, col);
    readData(EXP_OUT, exp_out, row, col_exp);

    readData(INP_VAL, inp_val, row_val, col);
    readData(EXP_OUT_VAL, exp_out_val, row_val, col_exp);

    // Stand
    // mean = 0, stdev = 1


    // find net.mean net.stdev
    if(epoch == 0)
    {
        stdev.resize(col);
        mean.resize(col);

        exp_max.resize(col_exp);
        exp_min.resize(col_exp);

        stand_Data(inp, mean, stdev);
        stand_Data_exp(exp_out, exp_min, exp_max, low, high);
    }
    else
    {
        apply_stand(inp, mean, stdev);
        apply_stand_exp(exp_out, exp_min, exp_max, low, high);
    }

    apply_stand_exp(exp_out_val, exp_min, exp_max, low, high);
    apply_stand(inp_val, mean, stdev);                                                                                   // !!!!!! CHECK same stdev mean
}

void Net::load_inp(char * INP, int inp_row)
{
    this->row = inp_row;
    readData(INP, inp, row, col);
    apply_stand(inp, mean, stdev);
}


// ACTIVATION FUNCTIONS

double sigmoid(double x)
{
    return 1/(1 + exp(-x));
}

double sigmoid_d(double x)
{
    return exp(-x)/( (1+exp(-x))*(1+exp(-x)) );
}

double tanhf(double x)
{
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

double tanh_d(double x)
{
    return ( (exp(x)+exp(-x))*(exp(x)+exp(-x)) - (exp(x)-exp(-x))*(exp(x)-exp(-x)) ) / ( (exp(x)+exp(-x))*(exp(x)+exp(-x)) );
}

double relu(double x)
{
    return x > 0 ? x : 0;
}

double relu_d(double x)
{
    return x > 0 ? 1 : 0;
}

double softplus(double x)
{
    return log(1+exp(x));
}

double softplus_d(double x)
{
    return 1/(1+exp(-x));
}

double random_w(double x)
{
    return (double)(rand() % 10000 + 1)/10000 - 0.5;
}


void Hello()
{
    cout << "    Netboi  " << endl << " - - - - - - - - - " << endl << endl;
}

int testFun()
{
    cout << "testfun";
    return 0;

}
