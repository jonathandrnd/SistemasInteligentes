/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package gsa;

//package Lee_Kesler_Base;
//=============================================================
//     Numerical Analysis Packages in Java
//     Matrix calculations class Matrix
//     Author : Dr. Turhan Coban
//=============================================================
//Turhan Coban
import java.io.*;
import java.text.*;
import java.util.Locale;

public class Matrix {
// This class defines matrix and vector
// calculation methods

//Formatted outputs
    static String toString(int n, int w) // converts an int to a string with given width.
    {
        String s = Integer.toString(n);
        while (s.length() < w) {
            s = " " + s;
        }
        return s;
    }

    static String toString(int left) // converts an int to a string with a constant width.
    {
        return toString(left, 4);
    }

    public static String toString(double left, int w, int d) // converts a double to a string with given width and decimals.
    {
        NumberFormat df = NumberFormat.getInstance(Locale.US);
        df.setMaximumFractionDigits(d);
        df.setMinimumFractionDigits(d);
        df.setGroupingUsed(false);
        String s = df.format(left);
        while (s.length() < w) {
            s = " " + s;
        }
        if (s.length() > w) {
            s = "";
            for (int i = 0; i < w; i++) {
                s = s + "-";
            }
        }
        return s;
    }

    public static String toString(double left) {// converts a double to a string with a constant width and constant decimals.
        return toString(left, 25, 15);
    }

    public static String toString(double[][] left) {
//return a string representation of a matrix
        return toString(left, 25, 15);
    }

    public static String toString(int[][] left, int w) {
//return a string representation of a matrix
        int n, m;
        String b;
        b = "";
        n = left.length;
        m = left[0].length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                b = b + toString(left[i][j], w);
            }
            b = b + "\n";
        }
        return b;
    }

    public static String toString(int[][] left) {
        return toString(left, 6);
    }

    public static String toStringT(double[][] left, int w, int d) {
        return toString(Matrix.Transpose(left), w, d);
    }

    public static String toStringT(double[][] left) {
        return toString(Matrix.Transpose(left), 10, 10);
    }

    public static String toString(double[][] left, int w, int d) {
//return a string representation of a matrix
        int n, m;
        String b;
        b = "";
        n = left.length;
        m = left[0].length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                b = b + toString(left[i][j], w, d);
            }
            b = b + "\n";
        }
        return b;
    }

    public static String toString(complex[][] left, int w, int d) {
//return a string representation of a complex matrix
        int n, m;
        String b;
        b = "";
        n = left.length;
        m = left[0].length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                b = b + left[i][j].toString() + "\t ";
            }
            b = b + "\n";
        }
        return b;
    }

    public static String toStringT(complex[] left) {
// returns a horizontal string representation of
// a complex vector
        int n, m;
        String b;
        b = "";
        n = left.length;
        for (int i = 0; i < n; i++) {
            b = b + left[i].toString() + "\n";
        }
        return b;
    }

    public static String toString(complex[] left) {
// returns a vertical string representation of
// a complex vector
        int n, m;
        String b;
        b = "";
        n = left.length;
        for (int i = 0; i < n; i++) {
            b = b + left[i].toString() + "\t ";
        }
        b = b + "\n";
        return b;
    }

    public static String toStringT(double[] left) {
// returns a vertical string representation
// of a double vector
        int n, m;
        String b;
        b = "";
        n = left.length;
        for (int i = 0; i < n; i++) {
            b = b + toString(left[i]) + "\n";
        }
        return b;
    }

    public static String toString(double[] left) {
// returns a horizontal string representation
// of a double vector
        int n, m;
        String b;
        b = "";
        n = left.length;
        for (int i = 0; i < n; i++) {
            b = b + toString(left[i]) + "\t ";
        }
        b = b + "\n";
        return b;
    }

    public static String toString(int[] left) {
// returns a horizontal string representation
// of a integer vector
        int n, m;
        String b;
        b = "";
        n = left.length;
        for (int i = 0; i < n; i++) {
            b = b + left[i] + "\t";
        }
        b = b + "\n";
        return b;
    }

    public static double SIGN(double a, double b) {
//returns the value of double a with sign of double b;
//if a=-2, b= 3 SIGN(a,b) returns  2
//if a=-2, b=-3 SIGN(a,b) returns -2
//if a= 2, b=-3 SIGN(a,b) returns -2
//if a= 2, b= 3 SIGN(a,b) returns  2
        if (b != 0) {
            return Math.abs(a) * b / Math.abs(b);
        } else {
            return Math.abs(a);
        }
    }

    public static double[][] inv(double[][] a) {
// INVERSION OF A MATRIX
// inversion by using gaussian elimination
// with full pivoting
        int n = a.length;
        int m = a[0].length;
        double b[][];
        b = new double[n][n];
        int indxc[];
        int indxr[];
        double ipiv[];
        indxc = new int[n];
        indxr = new int[n];
        ipiv = new double[n];
        int i, j, k, l, ll, ii, jj;
        int icol = 0;
        int irow = 0;
        double big, dum, pivinv, temp;
        if (n != m) {
            System.out.println("Matrix must be square ");
            for (ii = 0; ii < n; ii++) {
                for (jj = 0; jj < n; jj++) {
                    b[ii][jj] = 0.0;
                }
            }
            return b;
        }
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                b[i][j] = a[i][j];
            }
        }
        for (i = 0; i < n; i++) {
            big = 0.0;
            for (j = 0; j < n; j++) {
                if (ipiv[j] != 1) {
                    for (k = 0; k < n; k++) {
                        if (ipiv[k] == 0) {
                            if (Math.abs(b[j][k]) >= big) {
                                big = Math.abs(b[j][k]);
                                irow = j;
                                icol = k;
                            }
                        } else if (ipiv[k] > 1) {
                            System.out.println("error : inverse of the matrix : singular matrix-1");
                            for (ii = 0; ii < n; ii++) {
                                for (jj = 0; jj < n; jj++) {
                                    b[ii][jj] = 0.0;
                                }
                            }
                            return b;
                        }
                    }
                }
            }
            ++ipiv[icol];
            if (irow != icol) {
                for (l = 0; l < n; l++) {
                    temp = b[irow][l];
                    b[irow][l] = b[icol][l];
                    b[icol][l] = temp;
                }
            }
            indxr[i] = irow;
            indxc[i] = icol;
            if (b[icol][icol] == 0.0) {
                System.out.println("error : inverse of the matrix : singular matrix-2");
                for (ii = 0; ii < n; ii++) {
                    for (jj = 0; jj < n; jj++) {
                        b[ii][jj] = 0.0;
                    }
                }
                return b;
            }
            pivinv = 1.0 / b[icol][icol];
            b[icol][icol] = 1.0;
            for (l = 0; l < n; l++) {
                b[icol][l] *= pivinv;
            }
            for (ll = 0; ll < n; ll++) {
                if (ll != icol) {
                    dum = b[ll][icol];
                    b[ll][icol] = 0.0;
                    for (l = 0; l < n; l++) {
                        b[ll][l] -= b[icol][l] * dum;
                    }
                }
            }
        }
        for (l = n - 1; l >= 0; l--) {
            if (indxr[l] != indxc[l]) {
                for (k = 0; k < n; k++) {
                    temp = b[k][indxc[l]];
                    b[k][indxc[l]] = b[k][indxr[l]];
                    b[k][indxr[l]] = temp;
                }
            }
        }
        return b;
    }

    public static double[][] inverse(double a[][]) {
//inverse of a matrix
//this method enable usage of inverse as
//name of matrix inversion method
        return Matrix.inv(a);
    }

//LU decomposition method
    public static double[][] LU(double c[][], int indx[], int d[]) {
//returns LU decomposition of matrix c and index indx
        double a[][];
        int n = c.length;
        a = new double[n][n];
        double vv[];
        vv = new double[n];
        double sum, dum, big, temp;
        int i, j, k;
        int imax;
        int nmax = 100;
        double tiny = 1.0e-40;
        imax = 0;
        for (i = 1; i <= n; i++) {
            for (j = 1; j <= n; j++) {
                a[i - 1][j - 1] = c[i - 1][j - 1];
            }
        }
        d[0] = 1;
        for (i = 1; i <= n; i++) {
            big = 0.0;
            for (j = 1; j <= n; j++) {
                if (Math.abs(a[i - 1][j - 1]) > big) {
                    big = Math.abs(a[i - 1][j - 1]);
                }
            }
            if (big == 0) {
                System.out.println("singular matrix");
                return a;
            }
            vv[i - 1] = 1.0 / big;
        }
        for (j = 1; j <= n; j++) {
            for (i = 1; i < j; i++) {
                sum = a[i - 1][j - 1];
                for (k = 1; k < i; k++) {
                    sum -= a[i - 1][k - 1] * a[k - 1][j - 1];
                }
                a[i - 1][j - 1] = sum;
            }
            big = 0;
            for (i = j; i <= n; i++) {
                sum = a[i - 1][j - 1];
                for (k = 1; k < j; k++) {
                    sum -= a[i - 1][k - 1] * a[k - 1][j - 1];
                }
                a[i - 1][j - 1] = sum;
                dum = vv[i - 1] * Math.abs(sum);
                if (dum >= big) {
                    imax = i;
                    big = dum;
                }
            } //end of i=0
            if (j != imax) {
                for (k = 1; k <= n; k++) {
                    dum = a[imax - 1][k - 1];
                    a[imax - 1][k - 1] = a[j - 1][k - 1];
                    a[j - 1][k - 1] = dum;
                }
                d[0] = -d[0];
                vv[imax - 1] = vv[j - 1];
            } //end of if
            indx[j - 1] = imax;
            if (a[j - 1][j - 1] == 0) {
                a[j - 1][j - 1] = tiny;
            }
            if (j != n) {
                dum = 1.0 / a[j - 1][j - 1];
                for (i = j + 1; i <= n; i++) {
                    a[i - 1][j - 1] *= dum;
                }
            }//endif
        } //end for j=
        return a;
    }

    public static double[] LUaxb(double a[][], double x[], int indx[]) {
//solves AX=B system of linear equation of LU decomposed matrix a
//(calculated by method LU)
        int ii = 0;
        int i, j, ll = 0;
        double sum = 0;
        int n = a.length;
        double b[];
        b = new double[n];
        for (i = 1; i <= n; i++) {
            b[i - 1] = x[i - 1];
        }
        for (i = 1; i <= n; i++) {
            ll = indx[i - 1];
            sum = b[ll - 1];
            b[ll - 1] = b[i - 1];
            if (ii != 0) {
                for (j = ii; j <= (i - 1); j++) {
                    sum -= a[i - 1][j - 1] * b[j - 1];
                }
            } else if (sum != 0) {
                ii = i;
            }
            b[i - 1] = sum;
        }
        for (i = n; i >= 1; i--) {
            sum = b[i - 1];
            if (i < n) {
                for (j = (i + 1); j <= n; j++) {
                    sum -= a[i - 1][j - 1] * b[j - 1];
                }
            }
            b[i - 1] = sum / a[i - 1][i - 1];
        }
        return b;
    }

    public static double[] AXB(double a[][], double b[]) {
//Solution of system of linear equations by LU method
// note that the same calculation can be done by divide method.
        int n = a.length;
        double c[] = new double[n];
        int d[] = {1};
        int indx[] = new int[n];
        double e[][] = new double[n][n];
        e = Matrix.LU(a, indx, d);
        c = Matrix.LUaxb(e, b, indx);
        return c;
    } //end of AXB

    public static double[][] AXB(double a[][], double b[][]) {
        if (a.length != b.length) {
            System.out.println("Boyut hatasý Matrix.AxB 471");
            System.exit(1);
        }
        double[] bin = new double[b.length];
        double[][] bout = new double[b.length][1];
        for (int i = 0; i < bin.length; i++) {
            bin[i] = b[i][0];
        }
        double[] c = AXB(a, bin);
        for (int i = 0; i < bin.length; i++) {
            bout[i][0] = c[i];
        }
        return bout;
    }

    public static double[][] LUinv(double a[][]) {
//inverse of a matrix by using LU decomposition method
//this method is more efficient than inv (or inverse)
        int n = a.length;
        double c[][] = new double[n][n];
        double b[][] = Matrix.I(n);
        int d[] = {0};
        int indx[] = new int[n];
        double e[][] = new double[n][n];
        e = Matrix.LU(a, indx, d);
        for (int i = 0; i < n; i++) {
            c[i] = Matrix.LUaxb(e, b[i], indx);
        }
        return Matrix.T(c);
    } //end of LUinv

    public static double det(double a[][]) {
//determinant of a matrix
        int n = a.length;
        int indx[] = new int[n];
        int d[] = {1};
        double e;
        double b[][] = new double[n][n];
        b = Matrix.LU(a, indx, d);
        e = d[0];
        for (int i = 0; i < n; i++) {
            e = e * b[i][i];
        }
        return e;
    }  //end of det

    public static double determinant(double a[][]) {
//determinant of a matrix
        return Matrix.det(a);
    }

    public static double D(double a[][]) {
//determinant of a matrix
        return Matrix.det(a);
    }
//*************norm methods definition****************

    public static double norm(double v[]) {
// vector norm
        double total = 0;;
        for (int i = 0; i < v.length; i++) {
            total += v[i] * v[i];
        }
        return Math.sqrt(total);
    }

    public static double norm(double v[], int p) {
        //p vector norm
        if (p == 0) {
            p = 1;
        }
        double total = 0;
        double x, y;
        for (int i = 0; i < v.length; i++) {
            x = Math.abs(v[i]);
            total += Math.pow(x, p);
        }
        return Math.pow(total, (1.0 / p));
    }

    public static double norminf(double v[]) {
        //infinite vector norm
        double x;
        double max = 0;
        for (int i = 0; i < v.length; i++) {
            x = Math.abs(v[i]);
            if (x > max) {
                max = x;
            }
        }
        return max;
    }

    public static double[] vek(double mat[][], int n) {
        double a[] = new double[mat.length];
        for (int i = 0; i < mat.length; i++) {
            a[i] = mat[i][n];
        }
        return a;
    }

    public static double norminf(double a[][]) {
        //infinite matrix norm
        double x;
        double max = 0;
        double total;
        int i, j;

        int n = a.length;
        for (i = 0; i < n; i++) {
            total = 0;
            for (j = 0; j < n; j++) {
                total += Math.abs(a[i][j]);
            }
            x = total;
            if (x > max) {
                max = x;
            }
        }
        return max;
    }

    public static double norm(double a[][]) {
        //matrix norm
        double x;
        double max = 0;
        double total;
        int i, j;
        int n = a.length;
        for (j = 0; j < n; i++) {
            total = 0;
            for (i = 0; i < n; j++) {
                total += Math.abs(a[i][j]);
            }
            x = total;
            if (x > max) {
                max = x;
            }
        }
        return max;
    }

    public static double normE(double a[][]) {
        //Euclidian matrix norm
        double x;
        double total;
        int i, j;
        total = 0;
        int n = a.length;
        for (j = 0; j < n; i++) {
            for (i = 0; i < n; j++) {
                total += a[i][j] * a[i][j];
            }
        }
        return Math.sqrt(total);
    }

    public static double normEns(double a[][]) {
        //Euclidian matrix norm non square
        double x;
        double total;
        int i, j;
        int n = a.length;
        int m = a[0].length;
        total = 0;
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++) {
                total += a[i][j] * a[i][j];
            }
        }
        return Math.sqrt(total);
    }
//************* multiply methods definitions *********************

    public static double[][] dotproduct(double[][] left, double[][] right) {
        int leftrow = left.length;
        int leftcol = left[0].length;
        int rightrow = right.length;
        int rightcol = right[0].length;
        double b[][] = new double[leftrow][leftcol];
        if (leftcol != rightcol || leftcol!=rightcol){ 
            System.out.println("inner matrix dimensions must agree");
            for (int i = 0; i < leftrow; i++) {
                for (int j = 0; j < leftcol; j++) {
                    b[i][j] = 0.0;
                }
            }
            return b;
        }
        for (int i = 0; i < leftrow; i++) {
            for (int j = 0; j < leftcol; j++) {
                b[i][j] = left[i][j] * right[i][j];
            }
        }
        return b;
    }

    public static double[] dotproduct(double left[], double right[]) {
        int leftl = left.length;
        int rightl = right.length;
        double[] b = new double[left.length];
        if (leftl != rightl) {//System.out.println("Dimensions must agree");
            for (int i = 0; i < leftl; i++) {
                b[i] = 0.0;
            }
            return b;
        }
        for (int i = 0; i < leftl; i++) {
            b[i] = left[i] * right[i];
        }
        return b;
    }

    public static double[][] multiply(double[][] left, double[][] right) {
//multiplication of two matrices
        int ii, jj, i, j, k;
        int m1 = left[0].length;
        int n1 = left.length;
        int m2 = right[0].length;
        int n2 = right.length;
//System.out.println(n1+"x"+m1+"  "+n2+"x"+m2);
        double[][] b;
        b = new double[n1][m2];
        if (m1 != n2) {
            System.out.println("inner matrix dimensions must agree");
            for (ii = 0; ii < n1; ii++) {
                for (jj = 0; jj < m2; jj++) {
                    b[ii][jj] = 0;
                }
            }
            return b;
        }
        for (i = 0; i < n1; i++) {
            for (j = 0; j < m2; j++) {
                for (k = 0; k < m1; k++) {
                    //System.out.println(i+"  "+j+"   "+k);
                    b[i][j] += left[i][k] * right[k][j];
                    //System.out.println(b[i][j]);
                }
            }
        }
        return b;
//end of multiply of two matrices
    }

    public static double multiply(double[] left, double right[]) {
        int n = left.length;
        double s = 0;
        for (int i = 0; i < n; i++) {
            s += left[i] * right[i];
        }
        return s;
    }

    public static double[] multiply(double[][] left, double[] right) {
//multiplication of one matrix with one vector
        int ii, jj, i, j, k;
        int m1 = left[0].length;
        int n1 = left.length;
        int m2 = right.length;
        double[] b;
        b = new double[m2];
        if (n1 != m2) {
            System.out.println("inner matrix dimensions must agree");
            for (ii = 0; ii < n1; ii++) {
                b[ii] = 0;
            }
            return b;
        }
        for (i = 0; i < m1; i++) {
            b[i] = 0;
            for (k = 0; k < n1; k++) {
                b[i] += left[i][k] * right[k];
            }
        }
        return b;
//end of multiply of a matrix and a vector
    }

    public static double[] multiply(double[] left, double[][] right) {
//multiplication of one vector with one matrix
        int ii, jj, i, j, k;
        int m2 = right[0].length;
        int n2 = right.length;
        int m1 = left.length;
        double[] b;
        b = new double[m1];
        if (n2 != m1) {
            System.out.println("inner matrix dimensions must agree");
            for (ii = 0; ii < n2; ii++) {
                b[ii] = 0;
            }
            return b;
        }
        for (i = 0; i < m2; i++) {
            b[i] = 0;
            for (k = 0; k < m1; k++) {
                b[i] += right[i][k] * left[k];
            }
        }
        return b;
//end of multiply of a vector and a matrix
    }

    public static double[][] multiply(double left, double[][] right) {
//multiplying a matrix with a constant
        int i, j;
        int n = right.length;
        int m = right[0].length;
        double b[][];
        b = new double[n][m];
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++) {
                b[i][j] = right[i][j] * left;
            }
        }
        return b;
//end of multiplying a matrix with a constant double
    }

    public static double[][] multiply(double[][] left, double right) {
//multiplying a matrix with a constant
        int i, j;
        int n = left.length;
        int m = left[0].length;
        double b[][];
        b = new double[n][m];
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++) {
                b[i][j] = left[i][j] * right;
            }
        }
        return b;
//end of multiplying a matrix with a constant double
    }

    public static double[] multiply(double left, double[] right) {
//multiplying a vector with a constant
        int i;
        int n = right.length;
        double b[];
        b = new double[n];
        for (i = 0; i < n; i++) {
            b[i] = left * right[i];
        }
        return b;
    }

    public static double[] multiply(double[] left, double right) {
//multiplying a vector with a constant
        int i;
        int n = left.length;
        double b[];
        b = new double[n];
        for (i = 0; i < n; i++) {
            b[i] = right * left[i];
        }
        return b;
    }
//*************** end of multiply methods definitions **************
//=============== defination of power methods pow     ==============

    public static double[][] pow(double[][] right, double left) {
// power of a matrix
        int i, j;
        double b[][];
        int n = right.length;
        int m = right[0].length;
        b = new double[n][m];
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++) {
                if (left == 0.0) {
                    b[i][j] = 1.0;
                } else {
                    b[i][j] = Math.pow(right[i][j], left);
                }
            }
        }
        return b;
//end of power of a matrix
    }

    public static double[] pow(double[] right, double left) {
// power of a vector
        int i;
        int n = right.length;
        double b[];
        b = new double[n];
        for (i = 0; i < n; i++) {
            if (left == 0.0) {
                b[i] = 1.0;
            } else {
                b[i] = Math.pow(right[i], left);
            }
        }
        return b;
//end of power of a vector
    }

//=================end of power method pow definitions =============
//***************** addition add methods  **************************
    public static double[][] add(double[][] left, double[][] right) {
//addition of two matrices
        int n1 = left.length;
        int m1 = left[0].length;
        int n2 = right.length;
        int m2 = right[0].length;
        int nMax, mMax;
        int i, j;
        if (m1 >= m2) {
            mMax = m1;
        } else {
            mMax = m2;
        }
        if (n1 >= n2) {
            nMax = n1;
        } else {
            nMax = n2;
        }
        double b[][];
        b = new double[nMax][mMax];
        for (i = 0; i < n1; i++) {
            for (j = 0; j < m1; j++) {
                b[i][j] = b[i][j] + left[i][j];
            }
        }
        for (i = 0; i < n2; i++) {
            for (j = 0; j < m2; j++) {
                b[i][j] = b[i][j] + right[i][j];
            }
        }
        return b;
//end of matrix addition method
    }

    public static double[] add(double[] left, double[] right) {
//addition of two vectors
        int n1 = left.length;
        int n2 = right.length;
        int nMax;
        int i;
        if (n1 >= n2) {
            nMax = n1;
        } else {
            nMax = n2;
        }
        double b[];
        b = new double[nMax];
        for (i = 0; i < n1; i++) {
            b[i] = b[i] + left[i];
        }
        for (i = 0; i < n2; i++) {
            b[i] = b[i] + right[i];
        }
        return b;
//end of vector addition method
    }

    public static double[][] add(double left, double[][] right) {
        int m = right.length;
        int n = right[0].length;
        double[][] b = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                b[i][j] = left + right[i][j];
            }
        }
        return b;
    }

    public static double[][] substract(double[][] left, double[][] right) {
//addition of two matrices
        int n1 = left.length;
        int m1 = left[0].length;
        int n2 = right.length;
        int m2 = right[0].length;
        int nMax, mMax;
        int i, j;
        if (m1 >= m2) {
            mMax = m1;
        } else {
            mMax = m2;
        }
        if (n1 >= n2) {
            nMax = n1;
        } else {
            nMax = n2;
        }
        double b[][];
        b = new double[nMax][mMax];
//System.out.println("Matrix den left=\n "+Matrix.toString(left));
///System.out.println("Matrix den right=\n "+Matrix.toString(right));
        for (i = 0; i < n1; i++) {
            for (j = 0; j < m1; j++) {
                b[i][j] = b[i][j] + left[i][j];
            }
        }

//System.out.println("Matrix den b1=\n "+Matrix.toString(b));
        for (i = 0; i < n2; i++) {
            for (j = 0; j < m2; j++) {
                b[i][j] = b[i][j] - right[i][j];
            }
        }//System.out.println("Matrix den  b1=\n "+Matrix.toString(b));
        return b;
//end of matrix substraction method
    }

    public static double[] substract(double[] left, double[] right) {
//addition of two vectors
        int n1 = left.length;
        int n2 = right.length;
        int nMax;
        int i;
        if (n1 >= n2) {
            nMax = n1;
        } else {
            nMax = n2;
        }
        double b[];
        b = new double[nMax];
        for (i = 0; i < n1; i++) {
            b[i] = b[i] + left[i];
        }
        for (i = 0; i < n2; i++) {
            b[i] = b[i] - right[i];
        }
        return b;
//end of vector substraction method
    }

//============== division of the matrices
    public static double[][] divide(double[][] left, double[][] right) {
//division of two matrices
        int n = right.length;
        int m = right[0].length;
        double b[][];
        b = new double[n][m];
        b = Matrix.multiply(Matrix.inv(right), left);
        return b;
    }

    public static double[][] LUdivide(double[][] left, double[][] right) {
//division of two matrices utilises LUinv method instead of inv
        int n = right.length;
        int m = right[0].length;
        double b[][];
        b = new double[n][m];
        b = Matrix.multiply(Matrix.LUinv(right), left);
        return b;
    }

    public static double[] divide(double[] left, double[][] right) {
//division of two matrices
        int n = right.length;
        int m = right[0].length;
        double b[];
        b = new double[n];
        b = Matrix.multiply(Matrix.inv(right), left);
        return b;
    }

    public static double[][] divide(double[][] left, double right) {
//division of two matrices
        int n = left.length;
        int m = left[0].length;
        double aa[][] = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                aa[i][j] = left[i][j] / right;
            }
        }
        return aa;
    }

    public static double[] LUdivide(double[] left, double[][] right) {
//division of two matrices  utilises AXB (LU decomposition method)
//in fact this method is exactly same as AXB except spacing of the
//arguments
        return AXB(right, left);
    }

//============== absolute value of a matrix===================
    public static double abs(double[][] left) {
// absoulute value of a matrix
        int i, j;
        int n = left.length;
        int m = left[0].length;
        double b = 0;
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++) {
                b = b + Math.abs(left[i][j]);
            }
        }
        return b;
    }

    public static double abs(double[] left) {
// absolute value of a vector
        int i;
        int n = left.length;
        double b = 0;
        for (i = 0; i < n; i++) {
            b = b + Math.abs(left[i]);
        }
        return b;
    }

//===============special matrices==============================
    public static double[][] Transpose(double[][] left) {
//transpose matrix (if A=a(i,j) Transpose(A)=a(j,i)
        int i, j;
        int n = left.length;
        int m = left[0].length;
        double b[][];
        b = new double[m][n];
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++) {
                b[j][i] = left[i][j];
            }
        }
        return b;
    }

    public static double[][] T(double[][] left) {
//transpose matrix (if A=a(i,j) T(A)=a(j,i)
        int i, j;
        int n = left.length;
        int m = left[0].length;
        double b[][];
        b = new double[m][n];
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++) {
                b[j][i] = left[i][j];
            }
        }
        return b;
    }

    public static complex[][] T(complex[][] left) {
//transpose matrix (if A=a(i,j) T(A)=a(j,i)
        int i, j;
        int n = left.length;
        int m = left[0].length;
        complex b[][];
        b = new complex[m][n];
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++) {
                b[j][i] = new complex(left[i][j]);
            }
        }
        return b;
    }

    public static double VT_V(double[] left) {
//multiplys a vector transpose with a vector	
        int n = left.length;
        double tot = 0;
        for (int i = 0; i < n; i++) {
            tot += left[i] * left[i];
        }
        return tot;
    }

    public static double[][] V_VT(double[] left) {
//multiplys a vector transpose with a vector	
        int n = left.length;
        double aa[][] = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                aa[i][j] = left[i] * left[j];
            }
        }
        return aa;
    }

    public static double VT_X(double[] left, double[] right) {
//multiplys a vector transpose with a vector	
        int n = left.length;
        double tot = 0;
        for (int i = 0; i < n; i++) {
            tot += left[i] * right[i];
        }
        return tot;
    }

    public static double VT_G_V(double V[], double G[][]) {
        double x1[] = multiply(V, G);
        return VT_X(V, x1);
    }

    public static double[][] I(int n) {
//unit matrix
        double b[][];
        b = new double[n][n];
        for (int i = 0; i < n; i++) {
            b[i][i] = 1.0;
        }
        return b;
    }

    public static double[] one(int n) {
//one matrix
        double b[];
        b = new double[n];
        for (int i = 0; i < n; i++) {
            b[i] = 1.0;
        }
        return b;
    }

    public static double[][] characteristic_matrix(double c[]) {
//this routine converts polynomial coefficients to a matrix
//with the same eigenvalues (roots)
        int n = c.length - 1;
        int i;
        double a[][] = new double[n][n];
        for (i = 0; i < n; i++) {
            a[0][i] = -c[i + 1] / c[0];
        }
        for (i = 0; i < n - 1; i++) {
            a[i + 1][i] = 1;
        }
        return a;
    }

//===========Eigen value calculations ==============
    public static double[][] balance(double b[][]) {
// balance of a matrix for more accurate eigenvalue
// calculations
        double radix = 2.0;
        double sqrdx = radix * radix;
        double c, r, f, s, g;
        int m, j, i, last;
        int n = b.length;
        last = 0;
        double a[][];
        a = new double[n][n];
        f = 1;
        s = 1;
        for (i = 1; i <= n; i++) {
            for (j = 1; j <= n; j++) {
                a[i - 1][j - 1] = b[i - 1][j - 1];
            }
        }
        while (last == 0) {
            last = 1;
            for (i = 1; i <= n; i++) {
                c = 0;
                r = 0;
                for (j = 1; j <= n; j++) {
                    if (j != i) {
                        c += Math.abs(a[j - 1][i - 1]);
                        r += Math.abs(a[i - 1][j - 1]);
                    } //end of if(j!=..
                } //end of for(j=1...
                if (c != 0 && r != 0) {
                    g = r / radix;
                    f = 1.0;
                    s = c + r;
                    while (c < g) {
                        f *= radix;
                        c *= sqrdx;
                    }
                    g = r * radix;
                    while (c > g) {
                        f /= radix;
                        c /= sqrdx;
                    }
                } //end of if(c != 0 && ....
                if ((c + r) / f < 0.95 * s) {
                    last = 0;
                    g = 1.0 / f;
                    for (j = 1; j <= n; j++) {
                        a[i - 1][j - 1] *= g;
                    }
                    for (j = 1; j <= n; j++) {
                        a[j - 1][i - 1] *= f;
                    }
                } //end of if( ((c+r..
            }//end of for(i=1;i<=n....
        } //end of while last==0
        return a;
    }

    public static double[][] Hessenberg(double b[][]) {
// Calculates the hessenberg matrix
// it is used in QR method to calculate eigenvalues
// of a matrix(symmetric or non-symmetric)
        int m, j, i;
        int n = b.length;
        double a[][];
        a = new double[n][n];
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                a[i][j] = b[i][j];
            }
        }
        double x, y;
        if (n > 2) {
            for (m = 2; m <= (n - 1); m++) {
                x = 0.0;
                i = m;
                for (j = m; j <= n; j++) {
                    if (Math.abs(a[j - 1][m - 2]) > Math.abs(x)) {
                        x = a[j - 1][m - 2];
                        i = j;
                    } //end of if(Math.abs(..
                }//end of for(j=m,j<=n...
                if (i != m) {
                    for (j = (m - 1); j <= n; j++) {
                        y = a[i - 1][j - 1];
                        a[i - 1][j - 1] = a[m - 1][j - 1];
                        a[m - 1][j - 1] = y;
                    }//end of for(j=(m-1)..
                    for (j = 1; j <= n; j++) {
                        y = a[j - 1][i - 1];
                        a[j - 1][i - 1] = a[j - 1][m - 1];
                        a[j - 1][m - 1] = y;
                    }//end of for(j=1;j<=n....
                } //end of if(i!=m)
                if (x != 0.0) {
                    for (i = (m + 1); i <= n; i++) {
                        y = a[i - 1][m - 2];
                        if (y != 0.0) {
                            y = y / x;
                            a[i - 1][m - 2] = y;
                            for (j = m; j <= n; j++) {
                                a[i - 1][j - 1] -= y * a[m - 1][j - 1];
                            }
                            for (j = 1; j <= n; j++) {
                                a[j - 1][m - 1] += y * a[j - 1][i - 1];
                            }
                        }//end of if(y!=0..
                    }//end of for(i=(m+1)...
                } //end of if(x != 0.0...
            }//end of for(m=2;m<=(n-1)..
        }//end of Hessenberg
        for (i = 1; i <= n; i++) {
            for (j = 1; j <= n; j++) {
                if (i > (j + 1)) {
                    a[i - 1][j - 1] = 0;
                }
            }
        }
        return a;
    }

    public static double[][] QR(double b[][], boolean fsd) {
//calculates eigenvalues of a Hessenberg matrix
        int n = b.length;
        double rm[][] = new double[2][n];
        double a[][] = new double[n + 1][n + 1];
        double wr[] = new double[n + 1];
        double wi[] = new double[n + 1];
        int nn, m, l, k, j, its, i, mmin;
        double z, y, x, w, v, u, t, s, r = 0, q = 0, p = 0, anorm;
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                a[i + 1][j + 1] = b[i][j];
            }
        }
        anorm = Math.abs(a[1][1]);
        for (i = 2; i <= n; i++) {
            for (j = (i - 1); j <= n; j++) {
                anorm += Math.abs(a[i][j]);
            }
        }
        nn = n;
        t = 0.0;
        while (nn >= 1) {
            its = 0;
            do {
                for (l = nn; l >= 2; l--) {
                    s = Math.abs(a[l - 1][l - 1]) + Math.abs(a[l][l]);
                    if (s == 0.0) {
                        s = anorm;
                    }
                    if ((double) (Math.abs(a[l][l - 1]) + s) == s) {
                        break;
                    }
                }
                x = a[nn][nn];
                if (l == nn) {
                    wr[nn] = x + t;
                    wi[nn--] = 0.0;
                } else {
                    y = a[nn - 1][nn - 1];
                    w = a[nn][nn - 1] * a[nn - 1][nn];
                    if (l == (nn - 1)) {
                        p = 0.5 * (y - x);
                        q = p * p + w;
                        z = Math.sqrt(Math.abs(q));
                        x += t;
                        if (q >= 0.0) {
                            z = p + Matrix.SIGN(z, p);
                            wr[nn - 1] = wr[nn] = x + z;
                            if (z != 0) {
                                wr[nn] = x - w / z;
                            }
                            wi[nn - 1] = wi[nn] = 0.0;
                        } else {
                            wr[nn - 1] = wr[nn] = x + p;
                            wi[nn - 1] = -(wi[nn] = z);
                        }
                        nn -= 2;
                    } else {
                        if (its == 3000) {
                            System.out.println("Matrix.java, module QR, Too many iterations in hqr");
                            //System.exit(0);
                            return err(rm);
                        }
                        if (its == 10 || its == 20) {
                            t += x;
                            for (i = 1; i <= nn; i++) {
                                a[i][i] -= x;
                            }
                            s = Math.abs(a[nn][nn - 1]) + Math.abs(a[nn - 1][nn - 2]);
                            y = x = 0.75 * s;
                            w = -0.4375 * s * s;
                        }
                        ++its;
                        for (m = (nn - 2); m >= l; m--) {
                            z = a[m][m];
                            r = x - z;
                            s = y - z;
                            p = (r * s - w) / a[m + 1][m] + a[m][m + 1];
                            q = a[m + 1][m + 1] - z - r - s;
                            r = a[m + 2][m + 1];
                            s = Math.abs(p) + Math.abs(q) + Math.abs(r);
                            p /= s;
                            q /= s;
                            r /= s;
                            if (m == l) {
                                break;
                            }
                            u = Math.abs(a[m][m - 1]) * (Math.abs(q) + Math.abs(r));
                            v = Math.abs(p) * (Math.abs(a[m - 1][m - 1])
                                    + Math.abs(z) + Math.abs(a[m + 1][m + 1]));
                            if ((double) (u + v) == v) {
                                break;
                            }
                        }
                        for (i = m + 2; i <= nn; i++) {
                            a[i][i - 2] = 0.0;
                            if (i != (m + 2)) {
                                a[i][i - 3] = 0.0;
                            }
                        }
                        for (k = m; k <= nn - 1; k++) {
                            if (k != m) {
                                p = a[k][k - 1];
                                q = a[k + 1][k - 1];
                                r = 0.0;
                                if (k != (nn - 1)) {
                                    r = a[k + 2][k - 1];
                                }
                                if ((x = Math.abs(p) + Math.abs(q) + Math.abs(r)) != 0.0) {
                                    p /= x;
                                    q /= x;
                                    r /= x;
                                }
                            }
                            if ((s = Matrix.SIGN(Math.sqrt(p * p + q * q + r * r), p)) != 0.0) {
                                if (k == m) {
                                    if (l != m) {
                                        a[k][k - 1] = -a[k][k - 1];
                                    }
                                } else {
                                    a[k][k - 1] = -s * x;
                                }
                                p += s;
                                x = p / s;
                                y = q / s;
                                z = r / s;
                                q /= p;
                                r /= p;
                                for (j = k; j <= nn; j++) {
                                    p = a[k][j] + q * a[k + 1][j];
                                    if (k != (nn - 1)) {
                                        p += r * a[k + 2][j];
                                        a[k + 2][j] -= p * z;
                                    }
                                    a[k + 1][j] -= p * y;
                                    a[k][j] -= p * x;
                                }
                                mmin = nn < k + 3 ? nn : k + 3;
                                for (i = l; i <= mmin; i++) {
                                    p = x * a[i][k] + y * a[i][k + 1];
                                    if (k != (nn - 1)) {
                                        p += z * a[i][k + 2];
                                        a[i][k + 2] -= p * r;
                                    }
                                    a[i][k + 1] -= p * q;
                                    a[i][k] -= p;
                                }
                            }
                        }
                    }
                }
            } while (l < nn - 1);
        }
        for (i = 0; i < n; i++) {
            rm[0][i] = wr[i + 1];
            rm[1][i] = wi[i + 1];
        }
        return rm;
    } //end of QR

    public static double[][] err(double[][] iA) {
        for (int i = 0; i < iA.length; i++) {
            for (int j = 0; j < iA[0].length; j++) {
                iA[i][j] = (double) Double.NaN;
            }
        }
        return iA;
    }

    public static double[][] eigenValue(double b[][]) {
// this routine input a matrix (non symetric or symmetric)
// and calculate eigen values
// method balance can be used prior to this method to balance
// the input matrix
        int n = b.length;
        double d[][] = new double[2][n];
        d = Matrix.QR(Matrix.Hessenberg(b), true);
        return d;
    }

    public static complex[] eigenValueC(double b[][]) {
// this routine input a matrix (non symetric or symmetric)
// and calculate eigen values
// method balance can be used prior to this method to balance
// the input matrix
//output eigenvalues will be in a vector of complex form
        int n = b.length;
        double d[][] = new double[2][n];
        d = Matrix.QR(Matrix.Hessenberg(b), true);
        complex c[] = new complex[n];
        for (int i = 0; i < n; i++) {
            c[i] = new complex(d[0][i], d[1][i]);
        }
        return c;
    }

//roots of a polynomial
    public static double[][] poly_roots(double c[]) {
//roots of a degree n polynomial
// P(x)=c[n]*x^n+c[n-1]*x^(n-1)+....+c[1]*x+c[0]=0;
        int n = c.length - 1;
        double a[][] = new double[n][n];
        a = characteristic_matrix(c);
        double d[][] = new double[2][n];
        d = balancedEigenValue(a);
        return d;
    }

    public static double[][] cubic_roots(double d[]) {
// roots of a degree 3 polynomial
// P(x)=d[3]*x^3+d[2]*x^2+d[1]*x+d[0]=0;
// assuming d is all real;
// if value of d will be entered bigger than d[3] 
// remaining terms will be ignored
        double x[][] = new double[2][3];
        double a, b, c;
        a = d[2] / d[3];
        b = d[1] / d[3];
        c = d[0] / d[3];
//System.out.println("a="+a+"b="+b+"c="+c);
        double Q, R, theta;
        Q = (a * a - 3.0 * b) / 9.0;
        double x1, x2, x3, x4;
        x1 = 2.0 * a * a * a;
        x2 = -9 * a * b;
        x3 = 27 * c;
        x4 = x1 + x2 + x3;
        R = x4 / 54.0;
//System.out.println("Q="+Q+"R="+R+"x1="+x1+"x2="+x2+"x3="+x3+"x4="+x4);
        double Q3 = Q * Q * Q;
        double R2 = R * R;
        double qq;
//System.out.println("Q3="+Q3+"R2="+R2+(R2<Q3));
        double A, B;
        if (R2 < Q3) {
            qq = -2.0 * Math.sqrt(Q);
            theta = Math.acos(R / Math.sqrt(Q3));
            x[0][0] = qq * Math.cos(theta / 3.0) - a / 3.0;
            x[0][1] = qq * Math.cos((theta - 2.0 * Math.PI) / 3.0) - a / 3.0;
            x[0][2] = qq * Math.cos((theta + 2.0 * Math.PI) / 3.0) - a / 3.0;
        } else {
            A = -Math.pow((R + Math.sqrt(R2 - Q3)), (1.0 / 3.0));
            if (A == 0) {
                B = 0;
            } else {
                B = Q / A;
            }
            x[0][0] = (A + B) - a / 3.0;
            x[0][1] = -0.5 * (A + B) - a / 3;
            x[1][1] = Math.sqrt(3.0) / 2.0 * (A - B);
            x[0][2] = -0.5 * (A + B) - a / 3;
            x[1][2] = -Math.sqrt(3.0) / 2.0 * (A - B);
        }
        return x;
    }

    public static complex[] cubic_rootsC(double c[]) {
// roots of a degree 3 polynomial
//return 3 complex roots
        double a[][] = new double[2][3];
        a = cubic_roots(c);
        complex e[] = new complex[3];
        for (int i = 0; i < 3; i++) {
            e[i] = new complex(a[0][i], a[1][i]);
        }
        return e;
    }

    public static complex[] poly_rootsC(double c[]) {
// roots of a degree n polynomial
// P(x)=c[n]*x^n+c[n-1]*x^(n-1)+....+c[1]*x+c[0]=0;
// roots are returned as complex variables
        int n = c.length - 1;
        double a[][] = new double[n][n];
        a = characteristic_matrix(c);
        double d[][] = new double[2][n];
        d = balancedEigenValue(a);
        complex e[] = new complex[n];
        for (int i = 0; i < n; i++) {
            e[i] = new complex(d[0][i], d[1][i]);
        }
        return e;
    }

    public static double[][] balancedEigenValue(double b[][]) {
// this routine input a matrix (non symetric or symmetric)
// and calculates eigen values
// method balance is used to balance the matrix previous to
// actual calculations
        int n = b.length;
        double d[][] = new double[2][n];
        d = Matrix.QR(Matrix.Hessenberg(Matrix.balance(b)), true);
        return d;
    }

    public static double[][] tridiagonal(double b[][], double d[], double e[]) {
//reduces matrix to tridiaonal form by using householder transformation
//this method is used by QL method to calculate eigen values
//and eigen  vectors of a symmetric matrix
        int l, k, j, i;
        int n = b.length;
        double scale, hh, h, g, f;
        double a[][] = new double[n + 1][n + 1];
        double c[][] = new double[n][n];
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                a[i][j] = b[i][j];
            }
        }
        for (i = n; i >= 2; i--) {
            l = i - 1;
            h = scale = 0.0;
            if (l > 1) {
                for (k = 1; k <= l; k++) {
                    scale += Math.abs(a[i - 1][k - 1]);
                }
                if (scale == 0.0) {
                    e[i - 1] = a[i - 1][l - 1];
                } else {
                    for (k = 1; k <= l; k++) {
                        a[i - 1][k - 1] /= scale;
                        h += a[i - 1][k - 1] * a[i - 1][k - 1];
                    }
                    f = a[i - 1][l - 1];
                    g = (f >= 0.0 ? -Math.sqrt(h) : Math.sqrt(h));
                    e[i - 1] = scale * g;
                    h -= f * g;
                    a[i - 1][l - 1] = f - g;
                    f = 0.0;
                    for (j = 1; j <= l; j++) {
                        a[j - 1][i - 1] = a[i - 1][j - 1] / h;
                        g = 0.0;
                        for (k = 1; k <= j; k++) {
                            g += a[j - 1][k - 1] * a[i - 1][k - 1];
                        }
                        for (k = j + 1; k <= l; k++) {
                            g += a[k - 1][j - 1] * a[i - 1][k - 1];
                        }
                        e[j - 1] = g / h;
                        f += e[j - 1] * a[i - 1][j - 1];
                    }
                    hh = f / (h + h);
                    for (j = 1; j <= l; j++) {
                        f = a[i - 1][j - 1];
                        e[j - 1] = g = e[j - 1] - hh * f;
                        for (k = 1; k <= j; k++) {
                            a[j - 1][k - 1] -= (f * e[k - 1] + g * a[i - 1][k - 1]);
                        }
                    }
                }
            } else {
                e[i - 1] = a[i - 1][l - 1];
            }
            d[i - 1] = h;
        }
        d[1 - 1] = 0.0;
        e[1 - 1] = 0.0;
        /* Contents of this loop can be omitted if eigenvectors not
			wanted except for statement d[i-1]=a[i-1][i-1]; */
        for (i = 1; i <= n; i++) {
            l = i - 1;
            if (d[i - 1] != 0) {
                for (j = 1; j <= l; j++) {
                    g = 0.0;
                    for (k = 1; k <= l; k++) {
                        g += a[i - 1][k - 1] * a[k - 1][j - 1];
                    }
                    for (k = 1; k <= l; k++) {
                        a[k - 1][j - 1] -= g * a[k - 1][i - 1];
                    }
                }
            }
            d[i - 1] = a[i - 1][i - 1];
            a[i - 1][i - 1] = 1.0;
            for (j = 1; j <= l; j++) {
                a[j - 1][i - 1] = a[i - 1][j - 1] = 0.0;
            }
        }
        return a;
    }

    public static double pythag(double a, double b) {
//this method is used by QL method
        double absa, absb;
        absa = Math.abs(a);
        absb = Math.abs(b);
        if (absa > absb) {
            return absa * Math.sqrt(1.0 + (absb / absa) * (absb / absa));
        } else {
            return (absb == 0.0 ? 0.0 : absb * Math.sqrt(1.0 + (absa / absb) * (absa / absb)));
        }
    }

    public static double[][] QL(double d[], double e[], double a[][]) {
// QL algorithm : eigenvalues of a symmetric matrix reduced to tridiagonal
// form by using method tridiagonal
        int n = d.length;
        int m, l, iter, i, j, k;
        double s, r, p, g, f, dd, c, b;
        for (i = 2; i <= n; i++) {
            e[i - 2] = e[i - 1];
        }
        e[n - 1] = 0.0;
        double z[][] = new double[n][n];
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                z[i][j] = a[i][j];
            }
        }
        for (l = 1; l <= n; l++) {
            iter = 0;
            do {
                for (m = l; m <= n - 1; m++) {
                    dd = Math.abs(d[m - 1]) + Math.abs(d[m]);
                    if ((double) (Math.abs(e[m - 1]) + dd) == dd) {
                        break;
                    }
                }
                if (m != l) {
                    if (iter++ == 30) {
                        System.out.println("Too many iterations in QL");
                    }
                    g = (d[l] - d[l - 1]) / (2.0 * e[l - 1]);
                    r = Matrix.pythag(g, 1.0);
                    g = d[m - 1] - d[l - 1] + e[l - 1] / (g + Matrix.SIGN(r, g));
                    s = c = 1.0;
                    p = 0.0;
                    for (i = m - 1; i >= l; i--) {
                        f = s * e[i - 1];
                        b = c * e[i - 1];
                        e[i] = (r = Matrix.pythag(f, g));
                        if (r == 0.0) {
                            d[i] -= p;
                            e[m - 1] = 0.0;
                            break;
                        }
                        s = f / r;
                        c = g / r;
                        g = d[i] - p;
                        r = (d[i - 1] - g) * s + 2.0 * c * b;
                        d[i] = g + (p = s * r);
                        g = c * r - b;
                        for (k = 1; k <= n; k++) {
                            f = z[k - 1][i];
                            z[k - 1][i] = s * z[k - 1][i - 1] + c * f;
                            z[k - 1][i - 1] = c * z[k - 1][i - 1] - s * f;
                        }
                    }
                    if (r == 0.0 && i >= l) {
                        continue;
                    }
                    d[l - 1] -= p;
                    e[l - 1] = g;
                    e[m - 1] = 0.0;
                }
            } while (m != l);
        }
        return z;
    }

    public static double[][] eigenQL(double a[][]) {
        // QL algoritm to solve eigen value problems
        // symmetric matrices only (real eigen values)
        // first column of the matrix returns eigen values
        // second..n+1 column returns eigen vectors.
        // Note : If matrix is not symmetric DO NOT use
        // this method use eigenValue method (a QR algorithm)
        int i, j;
        int n = a.length;
        double sum[] = new double[n];;
        double d[] = new double[n];
        double b[][] = new double[n][n];
        double e[] = new double[n];
        double z[][] = new double[n + 1][n];
        b = tridiagonal(a, d, e);
        b = QL(d, e, b);
        for (j = 0; j < n; j++) {
            z[0][j] = d[j];
            for (i = 0; i < n; i++) {
                z[i + 1][j] = b[i][j] / b[0][j];
                if (z[i + 1][j] < 1e-13) {
                    z[i + 1][i] = 0;
                }
            }
        }
        // result: first row is eigenvalues, the rest is eigenvector
        return z;
    }

//end of eigen value programs
//end of class Matrix
}
