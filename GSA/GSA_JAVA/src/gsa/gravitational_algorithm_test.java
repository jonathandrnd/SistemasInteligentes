/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package gsa;
import java.util.*;

// FUNCION 1
class f1 extends f_xj //Ackley’s function 2.9        f(x)=0;      @x=(0,0,0...)     -32.768<x[i]<32.768
{
    public double func(double x[]) {
        double a = 20.0;
        double b = 0.2;
        double c = 2. * Math.PI;

        int n = x.length;

        double top1 = 0.0;
        for (int i = 0; i < n; i++) {
            top1 += x[i] * x[i];
        }
        top1 = Math.sqrt(top1 / n);
        double top2 = 0.0;
        for (int i = 0; i < n; i++) {
            top2 += Math.cos(c * x[i]);
        }
        top2 = top2 / n;
        double top = -a * Math.exp(-b * top1) - Math.exp(top2) + a + Math.exp(1);
        return top;
    }
}

// FUNCION 2
class f2 extends f_xj{
    //Corresponde al tamaño de LowerBound osea la cantidad de elementos de las particulas (Dimension)
    double func(double x[]) {
        int n = x.length;        
        double t = 418.9829*n;
        for (int i = 0; i < n; i++) {
            t -= x[i]*Math.sin(Math.sqrt(Math.abs(x[i])) );
        }
        return t;
    }
}

// FUNCION 3 
class f3 extends f_xj{
    //Corresponde al tamaño de LowerBound osea la cantidad de elementos de las particulas (Dimension)
    double func(double x[]) {
        int n = x.length;        
        double t = 0.5;
        double sum=0;
        
        for (int i = 0; i < n; i++)
            sum+=x[i]*x[i];
        
        t-=(Math.sin(Math.sqrt(sum))*Math.sin(Math.sqrt(sum))-0.5)/((1+0.001*sum)*(1+0.001*sum) );
        
        return -t;
    }
}

public class gravitational_algorithm_test {
    public static void main(String args[]){
        int d=10;// cantidad de dimensiones - numero de variables en la ecuacion
        double Lower[]=new double [d];
        double Upper[]=new double [d];
        for(int i=0;i<d;i++){
            Lower[i]=-32.768;    // minimo 
            Upper[i]=32.768;    //maximo
        }
        
        int N = 300000;// cantidad de particulas
        int maxiter = 100; // cantidad de iteraciones
        
        
        f1 ff = new f1();  // funcion a ejecutar
        System.out.println("Longitud Lower "+Lower.length);
        gravitional_search_algorithm gsa = new gravitional_search_algorithm(ff, N, maxiter, Lower, Upper);

        gsa.toStringnew();
    }
}





