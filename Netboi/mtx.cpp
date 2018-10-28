#include "mtx.h"


Mtx::Mtx(){}

Mtx::Mtx(int const &rows, int const &cols)
{
    row = rows;
    col = cols;
    this->array = vector< vector<double> >(rows, vector<double>(cols));
//    array.resize(rows);
//    for(int i = 0; i < rows; i++)
//        array[i].resize(cols);
}

void Mtx::mtx_load(vector <double> const &arr, int const &arr_size)
{
    row = 1;
    col = arr_size;

/*
    array[0] = arr;
*/
    for(int i = 0; i < col; i++)
    {
            array[0][i] = arr[i];
    }
}


void Mtx::mtx_copy(Mtx const &m)
{
    row = m.row;
    col = m.col;

    for(int i = 0; i < m.row; i++)
    {
        /*
        array[i] = m.array[i];
        */
        for(int j = 0; j < m.col; j++)
        {
            array[i][j] = m.array[i][j];
        }
    }
}

void Mtx::multiply(double const &value)
{
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            array[i][j] *= value;
        }
    }
}

void Mtx::multiply(vector <double> const &value)
{
    for(int i = 0; i < col; i++)
    {
        array[0][i] *= value[i];
    }
}

void Mtx::add(Mtx const &m)
{
    for(int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            array[i][j] += m.array[i][j];
        }
    }
}

void Mtx::add(vector <double> const &m)
{
    for(int i = 0; i < col; i++)
    {
        array[0][i] += m[i];
    }
}

void Mtx::subtract(Mtx const &m)
{
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            array[i][j] -= m.array[i][j];
        }
    }
}

void Mtx::multiply(Mtx const &m)
{
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            array[i][j] *= m.array[i][j];
        }
    }
}

void Mtx::dot(Mtx const &m, Mtx &targetMtx)
{
    register double w = 0;

    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < m.col; j++)
        {
            for(int h = 0; h < col; h++)
            {
                w += (array[i][h] * m.array[h][j]);
            }

            targetMtx.array[i][j] = w;
            w = 0;
        }
    }

    targetMtx.row = row;
    targetMtx.col = m.col;
}


void Mtx::transpose(Mtx const &m)
{
    row = m.col;
    col = m.row;
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            array[i][j] = m.array[j][i];
        }
    }
}

void Mtx::applyFunction(double (*function)(double))
{
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            array[i][j] = (*function)(array[i][j]);
        }
    }
}

void Mtx::print(ostream &outf)
{
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
            outf << array[i][j] << " ";
        outf << endl;
    }
}


void Mtx::destand(vector <double> & exp_min, vector <double> & exp_max, double low, double high)
{
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            array[i][j] = ( array[i][j] - ( low*exp_max[j] - high*exp_min[j] )/( exp_max[j]-exp_min[j] ) ) / ( (high-low)/(exp_max[j]-exp_min[j]) ) ;

        }
    }
}

ostream& operator <<(ostream &out, Mtx &m)
{
    m.print(out);
    return out;
}















