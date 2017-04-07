// GSA.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "GSA_Algorithm.h"
#include <vector>
#include <iostream>
#include <random>
#include<omp.h>

using namespace std;

int main(){
	omp_set_num_threads(1);

	int d = 10;
	vector<double>lower(d);
	vector<double>upper(d);

	#pragma omp parallel for
	for (int i = 0; i < d; i++){
		lower[i] = -500;
		upper[i] = 500;
	}

	//Cantidad de Particulas
	int N = 8000;

	//Cantidad de iteraciones
	int maxiter = 100;

	// Codigo de la funcion
	int ff = 2;
	//ff=1 -> Ackley funcion 1
	//ff=2 -> Schwefel funcion 2
	//ff=3 -> Funcion 3

	gravitational_search_algorithm(ff, N, maxiter, lower, upper);

	return 0;
}

