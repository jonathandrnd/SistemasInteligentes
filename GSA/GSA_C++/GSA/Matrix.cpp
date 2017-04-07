#include "stdafx.h"
#include <vector>
#include <iostream>
#include <random>
using namespace std;

double norm(vector<double>v) {
	double total = 0;;
	for (int i = 0; i < v.size(); i++)
		total += v[i] * v[i];

	return sqrt(total);
}

vector<double> substract(vector<double> left, vector<double>right) {
	int n = left.size();
	vector<double> b(n);

	for (int i = 0; i <n; i++)
		b[i] = left[i]-right[i];
	
	return b;
}

vector<vector<double> > multiply(double left, vector<vector<double> > right) {
	int n = right.size();
	int m = right[0].size();
	vector<vector<double> >b = right;

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			b[i][j] = right[i][j] * left;
		}
	}
	return b;
}

vector<vector<double> > dotproduct(vector<vector<double> >  left, vector<vector<double> >  right) {
	int n = left.size();
	int m = left[0].size();
	vector<vector<double> >b(n, vector<double>(m));

	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			b[i][j] = left[i][j] * right[i][j];
		
	return b;
}

vector<vector<double> > add(vector<vector<double> > left, vector<vector<double> > right) {
	int n = left.size();
	int m = left[0].size();
	
	vector<vector<double> >b(n, vector<double>(m, 0));

	for(int i = 0; i < n; i++) 
		for(int j = 0; j < m; j++) 
			b[i][j] = left[i][j] + right[i][j];

	return b;
}
