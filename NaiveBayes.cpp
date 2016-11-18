#include<iostream>
#include<cstring>
#include<cstdio>
#include<vector>
#include<cmath>
#include<sstream>
#include<algorithm>
#include<set>
#include<queue>
#include<map>
#include<ctime>
#include<string>
#include<time.h>
using namespace std;
	FILE * f1;
vector<string>dataset[5000];
double prob[10][50][10];//variable value  class
map<string,int>mclass;
string classname[10];
map<string,int>mvalues[10][10];
map<string,int>features[10];
double probClases[10];
int tamFeatures[10];
int numClases;
char separator=',';
int totalCol=0;
int totalRow=0;

void loadDataSet(){
	int count=0;
	char readLine[1024];
	
	while(fgets(readLine,1024,f1) ){
		int sizeline=strlen(readLine);
		for(int i=0;i<sizeline;i++)
			if(readLine[i]==separator)
				readLine[i]=' ';
		
		if(readLine[sizeline-1]== '\n')
			readLine[sizeline-1]=' ';
		else
			readLine[sizeline++]=' ';
		
		string aux;
		for(int i=0;i<sizeline;i++){
			if(readLine[i]!=' '){
				aux+=readLine[i];
			}else{
				dataset[count].push_back(aux);
				aux="";
			}
		}	
		count++;
	}	
	
	
	totalRow=count;
	totalCol=dataset[0].size();
}

void trainingSet(int numRow){
	int count=0;
	//variable value  class
	for(int i=0;i<10;i++)for(int j=0;j<50;j++)for(int k=0;k<10;k++)prob[i][j][k]=0;
	for(int i=0;i<10;i++)probClases[i]=0;

	for(int i=0;i<numRow;i++)
		if( mclass.find(dataset[i][totalCol-1])== mclass.end() ){
			classname[count]=dataset[i][totalCol-1];
			mclass[dataset[i][totalCol-1]]=count++;
		}
	
	numClases=count;
	memset(tamFeatures,0,sizeof(tamFeatures));
	
	for(int i=0;i<totalCol;i++){
		int count=0;
		for(int j=0;j<numRow;j++){
			if(features[i].find( dataset[j][i])== features[i].end() ){
				features[i][dataset[j][i]]=count++;
			}
		}
		tamFeatures[i]=count;
	}
	
	for(int i=0;i<totalCol;i++){
		for(int j=0;j<numRow;j++){
			int classnumber=mclass[dataset[j][totalCol-1]];
			int idfeature=features[i][dataset[j][i]];
			prob[i][classnumber][idfeature]++;
		}
		
		int tam=tamFeatures[i];
		for(int j=0;j<numClases;j++){
			int sum=0;
			for(int k=0;k<tam;k++){
				sum+=prob[i][j][k];
			}
			
			for(int k=0;k<tam;k++){
				if(sum!=0)
					prob[i][j][k]/=sum;
			}
		}
	}
	
	
	for(int i=0;i<numRow;i++){
		int idClass=mclass[dataset[i][totalCol-1]];
		probClases[idClass]++;
	}
	
	for(int i=0;i<numClases;i++){
		probClases[i]/=numRow;
	}
}

void performance(int ini,int fin){
	// P(C|X)= p(x|C)*P(C)/P(X)
	int numClases=mclass.size();
	//cout<<numClases<<endl;
	int correct=0;
	int failed=0;
	
	for(int i=ini;i<=fin;i++){
		double maxprob=-1;
		string maxclass="";
		for(int j=0;j<numClases;j++){
			double p=1;
			for(int k=0;k<totalCol-1;k++){
				string sfeatures=dataset[i][k];
				int idfeatures= features[k][sfeatures];
				p*=prob[k][j][idfeatures];		
			}
			p*=probClases[j];
			
			if(p>maxprob){
				maxprob=p;
				maxclass=classname[j];
			}
		}
		
		if(maxclass==dataset[i][totalCol-1]){
			correct++;	
		}else{
			failed++;
		}
	}
	
	double porcAciertos=(correct*100.0)/(correct+failed);
	printf("porcentaje de aciertos %.2lf\%\n",porcAciertos);
	
}

void prediccion(){
	char readLine[1024];
	vector<string>prediction[50];
	int count=0;
	
	while(fgets(readLine,1024,f1) ){
		int sizeline=strlen(readLine);
		for(int i=0;i<sizeline;i++)
			if(readLine[i]==separator)
				readLine[i]=' ';
		
		if(readLine[sizeline-1]== '\n')
			readLine[sizeline-1]=' ';
		else
			readLine[sizeline++]=' ';
		
		string aux;
		for(int i=0;i<sizeline;i++){
			if(readLine[i]!=' '){
				aux+=readLine[i];
			}else{
				prediction[count].push_back(aux);
				aux="";
			}
		}	
		count++;
	}	
	
	cout<<"---------PREDICCION---------- "<<endl;
	
	for(int i=0;i<count;i++){
		double maxprob=-1;
		string maxclass="";
		for(int j=0;j<numClases;j++){
			double p=1;
			for(int k=0;k<totalCol-1;k++){
				string sfeatures=prediction[i][k];
				int idfeatures= features[k][sfeatures];
				p*=prob[k][j][idfeatures];		
			}
			p*=probClases[j];
			
			if(p>maxprob){
				maxprob=p;
				maxclass=classname[j];
			}
		}
		
		for(int k=0;k<totalCol-1;k++)
			cout<<prediction[i][k]<<" ";
		cout<<endl;
		cout<<"Clase obtenida por Bayes ( "<<maxclass<<" )"<<endl;
	}
	
	
	
}

int main(){	
	f1 = fopen ("car.data.txt","r");
	
	clock_t t1=clock();
	loadDataSet();
	cout<<"Total de filas del dataset "<<totalRow<<endl;
    clock_t t2=clock();
	printf("carga de dataSET %.3lf segs\n",((double)t2-(double)t1)/1000);
	trainingSet( (int)(totalRow) );// tamanho del training set
	t1=clock();
	printf("carga de trainingSET %.3lf segs\n",((double)t1-(double)t2)/1000);	
	performance( 0,totalRow-1);   // inicio y fin de los datos que quiero probar su performance TestSet
	t2=clock();
	printf("Performance del Naive Bayes %.3lf segs\n",((double)t2-(double)t1)/1000);	

	fclose(f1); 
	f1 = fopen ("car-prueba.data.txt","r");
	prediccion();	
	
	t1=clock();
	cout<<"------------------------------"<<endl;
	printf("prediccion %.3lf segs\n",((double)t1-(double)t2)/1000);	
	 
	return 0;
}
