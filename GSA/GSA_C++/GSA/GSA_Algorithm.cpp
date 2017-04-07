#include "stdafx.h"
#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>
#include "Matrix.h"
#include<omp.h>
using namespace std;
int D;//Numero de dimensiones
int N;//Numero de particulars
int maxiter;//cantidad de iteraciones
double fmini;//minimo valor de la funcion benchmark
double fsol[10];//minimo valor de alguna particular
double Masses[100000];
double Lowerbound[10];
double Upperbound[10];
double X[100000][10];
double V[100000][10];
double A[100000][10];
double fitness[100000];
double indexed[100000];
double Lb;
double Ub;
double BESTVAL[1000];
int index;
double mini;
vector<double> dep;
double G;
int fun;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<> dis(0, 1);

double benchmark(double x[], int fun){
	if (fun == 1){
		double a = 20.0;
		double b = 0.0;
		double c = 2 * acos(-1);
		int n = D;

		double top1 = 0;
		
		#pragma omp parallel for reduction(+:top1)
		for (int i = 0; i < n; i++)
			top1 += x[i] * x[i];

		top1 = sqrt(top1 / n);
		double top2 = 0.0;
	
		#pragma omp parallel for reduction(+:top2)
		for (int i = 0; i < n; i++){
			top2 += cos(c*x[i]);
		}

		top2 /= n;
		return -a*exp(-b*top1) - exp(top2) + a + exp(1);
	}

	if (fun == 2){
		int n = D;
		double t = 0;
		#pragma omp parallel for reduction(+:t)
		for (int i = 0; i < n; i++){
			t += -x[i] * sin(sqrt(abs(x[i])));
		}
		return t + 418.9829*n;
	}

	if (fun == 3){
		int n = D;
		double t = 0.5;
		double sum = 0;

		#pragma omp parallel for reduction(+:sum)
		for (int i = 0; i < n; i++)
			sum += x[i] * x[i];

		t -= (sin(sqrt(sum))*sin(sqrt(sum)) - 0.5) / ((1 + 0.001*sum)*(1 + 0.001*sum));
		return -t;
	}
	
	return -666;
}

inline double random(){
	return dis(gen);
}


void movements() {

	//v = add(dotproduct(dummy, v), a);//Equation (11)
    //x = add(x, v);//Equation (12) 	 

	for (int i = 0; i < N; i++)
		for (int j = 0; j < D; j++)
			V[i][j] = random()*V[i][j]+A[i][j];

	#pragma omp parallel for
	for (int i = 0; i < N; i++)
		for (int j = 0; j < D; j++)
			X[i][j] = X[i][j]+V[i][j];
	return ;
}

void ascendingsort_index(double fitness[]) {
	for (int i = 0; i<N; i++)indexed[i] = i;

	for (int i = 0; i<min(30, N / 30); i++)
		for (int j = i + 1; j < N; j++){
			if (fitness[(int)indexed[i]]>fitness[(int)indexed[j]]){
				swap(indexed[i], indexed[j]);
			}
		}
}

void Gravfield(double GG) {
	double epsilon = 0.00001;
	double R = 0.0;
	
	ascendingsort_index(fitness);
	for (int i = 0; i < N; i++)
		for (int j = 0; j < D; j++)
			A[i][j] = 0;

	int j = 0;
	for (int i = 0; i < N; i++) {
		for (int tt = 0; tt < min(30, N / 30); tt++) {
			j = indexed[tt];
			if (j != i) {
				R = 0;
				//#pragma omp parallel for reduction(+:R)
				for (int k = 0; k < D; k++)R += (X[i][k] - X[j][k])*(X[i][k] - X[j][k]);
				R = sqrt(R);

				for (int k = 0; k < D; k++) {
					A[i][k] += random() * Masses[j] * ((X[j][k] - X[i][k]) / (R + epsilon));
				} //Equation 7
			}
		}
	}

	/*
	for (int i = 0; i < 4; i++)cout << A[i][0] << " "<<A[i][1]<<" "<<A[i][2]<<" "<<A[i][3];
	cout << endl;
	*/
	#pragma omp parallel for
	for (int i = 0; i < N; i++)
		for (int j = 0; j < D; j++)
			A[i][j]=A[i][j]*G;
	
	return;
}



double Gconst(int iterr) {
	return 300000 * exp(-9 * (double)iterr / (double)maxiter); //funcion 2
	//return 600 * exp(-4 * (double)iterr / (double)maxiter); //funcion 1
	//return 600 * exp(-4 * (double)iterr / (double)maxiter); //funcion 3
}

vector<double>  getmaxval_index(double a[]) {

	double m = 0.0;
	double maxval = a[0];
	for (int j = 0; j < N; j++) {
		if (a[j] > maxval) {
			maxval = a[j];
			m = j;
		}
	}
	
	vector<double>dep(2,0);
	dep[0] = maxval;
	dep[1] = m;
	return dep;
}


vector<double> getminval_index(double a[]) {

	double m = 0.0;
	double minval = a[0];

	for (int i = 0; i <N; i++) {
		if (a[i] < minval) {
			minval = a[i];
			m = i;
		}
	}

	vector<double>dep2(2);
	dep2[0] = minval;
	dep2[1] = m;
	return dep2;
}


void Calcmass(double fit[]) {

	vector<double> dpmin(2);
	vector<double> dpmax(2);
	double FFmin = 0.0;
	double FFmax = 0.0;
	double best = 0.0;
	double worst = 0.0;
	
	dpmin = getminval_index(fit);
	FFmin = dpmin[0];
	dpmax = getmaxval_index(fit);
	FFmax = dpmax[0];
	best = FFmin;
	worst = FFmax;

	#pragma omp parallel for
	for (int i = 0; i < N; i++) 
		Masses[i] = (fit[i] - worst) / (best - worst);

	double t = 0.0;
	//#pragma omp parallel for
	for (int i = 0; i <N; i++) 
		t += Masses[i];
	

	#pragma omp parallel for
	for (int i = 0; i < N; i++) 
		Masses[i] = Masses[i] / t;

}


void Fitcalc() {
	#pragma omp parallel for
	for (int i = 0; i < N; i++)
		fitness[i] = benchmark(X[i],fun);
	return;
}


void randomsimplebounds() {
	#pragma omp parallel for
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < D; j++) {
			if ((X[i][j] < Lowerbound[j]) || (X[i][j] > Upperbound[j])) {
				X[i][j] = Lowerbound[j] + (Upperbound[j] - Lowerbound[j])*random();
			}
		}
	}
}


void initialize() {
	#pragma omp parallel for
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < D; j++) {
			X[i][j] = Lowerbound[j] + ((Upperbound[j] - Lowerbound[j]) * random());
			V[i][j] = 0;
		}
	}
}

vector<vector<double> > solution() {
	omp_set_num_threads(1);
	initialize();
	cout<<"Cantidad de iteraciones " <<maxiter<<endl;

	int iter = 0;
	while (iter < maxiter) {
		randomsimplebounds();//Si algun elemento sale de los limites se ajusta
		Fitcalc();//Arreglo valor solucion en un arreglo por cada agente 
		dep = getminval_index(fitness);//dep[0]=minval dep[1]=indexdelminvalue
		mini = dep[0];  //minvalor
		index = (int)dep[1];///indice
		if (iter == 0) {
			fmini = mini;
			for (int i = 0; i < D; i++) {
				fsol[i] = X[index][i];
			}
		}
		if (mini < fmini) {
			fmini = mini;
			for (int i = 0; i < D; i++) {
				fsol[i] = X[index][i];
			}
		}

		//fsol coge el arreglo del agente q da el menor valor
		Calcmass(fitness);
		G = Gconst(iter);
		Gravfield( G);
		movements();
		BESTVAL[iter] = fmini;
		cout << "iteracion " << iter << " " << fmini << endl;
		iter++;
	}

	vector<vector<double> >d5 = vector<vector<double> >(2,vector<double>(D));
	d5[0][0] = fmini;
	for (int i = 0; i < D; i++)
		d5[1][i] = fsol[i];

	/*
	cout<<d5[0][0]<<endl;
	for (int i = 0; i < D; i++) {
		cout<<"x[" <<i <<"] =" << d5[1][i]<<endl;
	}*/

	return d5;
}


void gravitational_search_algorithm(int _fun, int _N, int _maxiter, vector<double>lower, vector<double>upper){
	fun = _fun;
	N = _N;
	maxiter = _maxiter;
	D = lower.size();
	for (int i = 0; i < D; i++){
		Lowerbound[i] = lower[i];
		Upperbound[i] = upper[i];
	}
	solution();
	return;
}