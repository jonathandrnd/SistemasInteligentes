/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package gsa;

import java.util.*;
import java.io.*;
import static java.io.FileDescriptor.out;
import java.lang.System;

/*
abstract class f_xj{
    abstract double func(double x[]);
}
*/

public class gravitional_search_algorithm{

    int D;//Number of dimension(Agent's dimension)
    int N;//Number of Agents
    int maxiter;//iteration number
    double fmin;//minimum value of a function(so far)
    double fsol[]; //minimum value of an agent(mass)
    double Masses[];
    double Lowerbound[];
    double Upperbound[];
    double Lbvec[];
    double Ubvec[];
    double X[][];
    double V[][];
    double A[][];
    double fitness[];
    double Lb;
    double Ub;
    double BESTVAL[];
    int index;
    double min;
    double[] dep;
    double G;
    f_xj ff;

    public gravitional_search_algorithm(f_xj iff, int iN, int imaxiter, double[] iLowerbound, double[] iUpperbound) {
        ff = iff;
        Lowerbound = iLowerbound;
        Upperbound = iUpperbound;
        D = Upperbound.length;
        N = iN;
        maxiter = imaxiter;
        Lbvec = new double[D];
        Ubvec = new double[D];
        X = new double[N][D];
        V = new double[N][D];
        fitness = new double[N];
        fsol = new double[D];
        Masses = new double[N];
        A = new double[N][D];
        BESTVAL = new double[maxiter];
    }

    void initialize() {
        
        System.out.println("inicializando");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < D; j++) {
                X[i][j] = Lowerbound[j] + ((Upperbound[j] - Lowerbound[j]) * Math.random());
                V[i][j]=0;
            }
        }
        
        
    }

    // if out of bounds,shift into boundary
    double[][] simplebounds(double s[][]) {
        System.out.println("simplebounds");
        
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < D; j++) {
                if (s[i][j] < Lowerbound[j]) {
                    s[i][j] = Lowerbound[j];
                }
                if (s[i][j] > Upperbound[j]) {
                    s[i][j] = Upperbound[j];
                }
            }
        }
        return s;
    }

    //if out of bounds create a random number between Lb and Ub
    double[][] randomsimplebounds(double s[][]) {
        //System.out.println("randomsimplebounds Solution()");

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < D; j++) {
                if ((s[i][j] < Lowerbound[j]) || (s[i][j] > Upperbound[j])) {
                    s[i][j] = Lowerbound[j] + (Upperbound[j] - Lowerbound[j]) * Math.random();
                }
            }
        }
        return s;
    }
    // Fitness Calculation

    double[] Fitcalc(double a[][]) {
        //System.out.println("Fitcalc() Solution()");

        for (int i = 0; i < N; i++) {
            fitness[i] = ff.func(a[i]);
        }
        return fitness;
    }

    // get minimum value and index from array
    double[] getminval_index(double[] a) {
       //System.out.println("getminval_index() Solution()");

        double m = 0.0;
        double b[] = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            b[i] = a[i];
        }
        double minval = a[0];
        for (int i = 0; i < a.length; i++) {
            if (a[i] < minval) {
                minval = a[i];
            }
        }
        for (int i = 0; i < a.length; i++) {
            if (b[i] == minval) {
                m = i;
                break;
            }
        };
        double[] dep = new double[2];
        dep[0] = minval;
        dep[1] = m;
        return dep;
    }

    // get maximum value and array from index
    double[] getmaxval_index(double a[]) {
        //System.out.println("getmaxval_index() Solution()");

        double m = 0.0;
        double b[] = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            b[i] = a[i];
        }
        double maxval = a[0];
        for (int j = 0; j < a.length; j++) {
            if (a[j] > maxval) {
                maxval = a[j];
            }
        }
        for (int i = 0; i < b.length; i++) {
            if (b[i] == maxval) {
                m = i;
                break;
            }
        }
        double dep2[] = new double[2];
        dep2[0] = maxval;
        dep2[1] = m;
        return dep2;
    }

    // Calculation of masses for mininum solution // equation 14-18
    double[] Calcmass(double fit[]) {
        //System.out.println("Calcmass() Solution()");

        double[] dpmin = new double[2];
        double[] dpmax = new double[2];
        double FFmin = 0.0;
        double FFmax = 0.0;
        double best = 0.0;
        double worst = 0.0;
        double[] Masss = new double[fit.length];
        double[] Massss = new double[fit.length];
        dpmin = getminval_index(fit);
        FFmin = dpmin[0];
        dpmax = getmaxval_index(fit);
        FFmax = dpmax[0];
        best = FFmin;
        worst = FFmax;
        for (int i = 0; i < fit.length; i++) {
            Masss[i] = (fit[i] - worst) / (best - worst);
        }
        double t = 0.0;
        for (int i = 0; i < fit.length; i++) {
            t += Masss[i];
        }
        for (int i = 0; i < fit.length; i++) {
            Massss[i] = Masss[i] / t;
        }

        return Massss;
    }

    int[] descendingsort_index(double[] Mass) {
        //System.out.println("descendingsort_index() Solution()");

        Vector<Double> vv = new Vector<Double>();
        for (int i = 0; i < Mass.length; i++) {
            vv.add(Mass[i]);
        }
        Object[] vold = vv.toArray();
        String str = "";
        double[] voldarray1 = new double[Mass.length];
        for (int i = 0; i < Mass.length; i++) {
            voldarray1[i] = Double.parseDouble(vold[i].toString());
        }
        // voldarray1 holds unsorted values

        Collections.sort(vv, Collections.reverseOrder());//sorts in descending order
        Object[] vsort = vv.toArray();
        String strsorted = "";
        double[] vsorted = new double[Mass.length];
        for (int i = 0; i < Mass.length; i++) {
            vsorted[i] = Double.parseDouble(vsort[i].toString());
        }
        int[] indexed = new int[Mass.length];
        for (int i = 0; i < Mass.length; i++) {
            for (int j = 0; j < Mass.length; j++) {
                if (voldarray1[i] == vsorted[j]) {
                    indexed[j] = i;
                }
            }
        }

        return indexed;

    }

    int[] ascendingsort_index(double[] fitness) {
        int c[]=new int[fitness.length];
        int tam=c.length;
        for(int i=0;i<tam;i++)c[i]=i;
        
        for(int i=0;i<30;i++)
            for(int j=i+1;j<tam;j++){
                if(fitness[c[i]]>fitness[c[j]]){
                    int aux=c[i];
                    c[i]=c[j];
                    c[j]=aux;
                }
            }
        
        return c;
    }
    
    // Calculation of Gravitational constant //Equation 28

    double Gconst(int iterr) {
        return 90000.0 * Math.exp(-3* (double) iterr / (double) maxiter);
    }

    public double[][] Gravfield(double[] Mass, double[][] XX, double GG) {
        //System.out.println("GG "+GG);
        double epsilon = 0.000001;
        double[][] dummy = new double[N][D];
        int[] xx = new int[N];
        double R = 0.0;
        //fitness[i]
        //int[] indexed = descendingsort_index(Mass);
        int[] indexed = ascendingsort_index(fitness);
        //for(int i=0;i<10;i++)
        //    System.out.println("ii "+i+" "+indexed[i]+" "+fitness[indexed[i]]);
        //int j = 0;

        for (int i = 0; i < N; i++) {
            for (int tt = 0; tt < 30; tt++) {
                int j=indexed[tt];
                if (j != i) {
                    R = Matrix.norm(Matrix.substract(X[i], X[j]));//Equation 8
                    for (int k = 0; k < D; k++) {
                        dummy[i][k] += Math.random() * Mass[j] * ((X[j][k] - X[i][k]) / (R  + epsilon));
                    } //Equation 7
                }
            }
            
        }
        return Matrix.multiply(GG, dummy);
    }

    double[][][] movements(double x[][], double a[][], double[][] v) {
        double[][] dummy = new double[N][D];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < D; j++) {
                dummy[i][j] = Math.random();
            }
        };
        v = Matrix.add(Matrix.dotproduct(dummy, v), a);//Equation (11)
        x = Matrix.add(x, v);//Equation (12) 	 
        /*
        System.out.println("Posicion ");
        for (int i = 0; i < 10; i++) {
            String sum="";
            for (int j = 0; j < D; j++) {
                sum+=x[i][j];
                sum+=" ";
            }
            System.out.println(sum);
        }
        
        System.out.println("Velocidad ");
        for (int i = 0; i < 10; i++) {
            String sum="";
            for (int j = 0; j < D; j++) {
                sum+=v[i][j];
                sum+=" ";
            }
            System.out.println(sum);
        }
        */
        
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < D; j++) {
                X[i][j] = x[i][j];
                V[i][j] = v[i][j];
            }
        }
        double d3[][][] = new double[2][N][D];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < D; j++) {
                d3[0][i][j] = X[i][j];
                d3[1][i][j] = V[i][j];
            }
        }
        return d3;
    }

    public double[][] solution() {
        initialize();

        double[][][] d4 = new double[2][N][D];
        System.out.println("Cantidad de iteraciones "+maxiter);
        int iter = 0;
        while (iter < maxiter) {
            X = randomsimplebounds(X);//Si algun elemento sale de los limites se ajusta
            fitness = Fitcalc(X);//Arreglo valor solucion en un arreglo por cada agente 
            dep = getminval_index(fitness);//dep[0]=minval dep[1]=indexdelminvalue
            min = dep[0];  //minvalor
            index = (int) dep[1];///indice
            if (iter == 0) {
                fmin = min;
                for (int i = 0; i < D; i++) {
                    fsol[i] = X[index][i];
                }
            }
            if (min < fmin) {
                fmin = min;
                for (int i = 0; i < D; i++) {
                    fsol[i] = X[index][i];
                }
            }
            
            //fsol coge el arreglo del agente q da el menor valor
            
            Masses = Calcmass(fitness);
            G = Gconst(iter);
            A = Gravfield(Masses, X, G);
            d4 = movements(X, A, V);
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < D; j++) {
                    X[i][j] = d4[0][i][j];
                    V[i][j] = d4[1][i][j];
                }
            }

            BESTVAL[iter] = fmin;
            System.out.println("iteracion "+iter+" "+fmin);
            iter++;
        }
        double[] iterdep = new double[maxiter];
        
        double[][] d5 = new double[2][D];
        d5[0][0] = fmin;
        for (int i = 0; i < D; i++) {
            d5[1][i] = fsol[i];
        }
        return d5;
    }

    void toStringnew() {
        double[][] dd = solution();
        System.out.println("Optimized value = " + dd[0][0]);
        for (int i = 0; i < D; i++) {
            System.out.println("x[" + i + "] =" + dd[1][i]);
        }
    }

}
