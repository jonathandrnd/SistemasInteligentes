#include <vector>
#include <iostream>
#include <random>
using namespace std;
void gravitational_search_algorithm(int fun, int N, int maxiter, vector<double>lower, vector<double>upper);
double benchmark(vector<double>x, int fun);
vector<vector<double> > solution();
void initialize();
double random();
vector<double> Fitcalc(vector<vector<double> > a);
vector<double> Calcmass(vector<double> fit);
vector<double> getminval_index(vector<double> a);
vector<double> getmaxval_index(vector<double>  a);
vector<vector<double> >Gravfield(vector<double> Mass, vector<vector<double> > XX, double GG);
vector<int> ascendingsort_index(vector<double> fitness);
vector<vector<vector<double> > > movements(vector<vector<double> > x, vector<vector<double> > a, vector<vector<double> >v);
