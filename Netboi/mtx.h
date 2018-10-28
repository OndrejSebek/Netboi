#ifndef MTX_H_INCLUDED
#define MTX_H_INCLUDED

#include <vector>
#include <iostream>

using namespace std;

class Mtx
{
public:
    Mtx();
    Mtx(int const &rows, int const &cols);

    void mtx_load(vector <double> const &arr, int const &arr_size);
    void mtx_copy(Mtx const &m);

    void multiply(double const &value);

    void add(Mtx const &m);
    void add(vector <double> const &m);
    void subtract(Mtx const &m);
    void multiply(Mtx const &m);
    void multiply(vector <double> const &value);

    void dot(Mtx const &m, Mtx &targetMtx);
    void transpose(Mtx const &m);

    void destand(vector <double> & exp_min, vector <double> & exp_max, double low, double high);

    void applyFunction(double (*function)(double));

    void print(ostream &outf);

//private:
    vector< vector<double> > array;
    int row;
    int col;

};


ostream& operator <<(ostream &out, Mtx &m);

#endif // MTX_H_INCLUDED
