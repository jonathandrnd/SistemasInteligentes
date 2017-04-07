/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package gsa;

import java.io.*;
import java.text.*;
import java.util.Locale;
import java.util.*;

import javax.swing.*; // giriþ çýkýþ

// class complex
//   complex number abstraction
//
class complex {
    // constructors

    public complex() {
        nreal = 0;
        nimag = 0;
    }

    public complex(double nre, double nim) {
        nreal = nre;
        nimag = nim;
    }

    public complex(int nre, int nim) {
        nreal = (double) nre;
        nimag = (double) nim;
    }

    public complex(double numer) {
        nreal = numer;
        nimag = 0;
    }

    public complex(complex value) {
        nreal = value.real();
        nimag = value.imaginary();
    }

// accessor functions
    public double real() {
//return real part of the complex number	
        return nreal;
    }

    public double imaginary() {
//return imaginary part of the complex number		
        return nimag;
    }

    public double R() {
// Return radius of polar coordinate equivalent of complex number	
        return Math.sqrt(nreal * nreal + nimag * nimag);
    }

    public double theta() {
// Return angle of polar coordinate equivalent of complex number in radian
        return Math.atan2(nimag, nreal);
    }

    public double dtheta() {
// Return angle of polar coordinate equivalent of complex number in degree
        return Math.atan2(nimag, nreal) * 180.0 / Math.PI;
    }

    // assignments
    public void assign(complex right) {
        nreal = right.real();
        nimag = right.imaginary();
    }

    public void add(complex right) {
        nimag = nimag + right.imaginary();
        nreal = nreal + right.real();
    }

    public void add(double nr1) {
        nreal = nreal + nr1;
    }

    public void add(double nr1, double ni1) {
        nreal = nreal + nr1;
        nimag = nimag + ni1;
    }

    public void substract(complex right) {
        nimag = nimag - right.imaginary();
        nreal = nreal - right.real();
    }

    public void multiply(complex right) {
        nreal = nreal * right.real() - nimag * right.imaginary();
        nimag = nreal * right.imaginary() + nimag * right.real();
    }

    public void divide(complex right) {
        double a = nreal * nreal + nimag * nimag;
        nreal = (nreal * right.real() + nimag * right.imaginary()) / a;
        nimag = (-nreal * right.imaginary() + nimag * right.real()) / a;
    }

    public static complex add(complex left, complex right) { // return sum of two complex numbers
        double r1 = (left.real() + right.real());
        double i1 = (left.imaginary() + right.imaginary());
        complex result;
        result = new complex(r1, i1);
        return result;
    }

    public static complex add(complex left, double right) { // return sum of two complex numbers
        double r1 = (left.real() + right);
        double i1 = left.imaginary();
        complex result;
        result = new complex(r1, i1);
        return result;
    }

    public void conjugate() {
        nimag *= -1;
    }

    public static complex conjugate(complex z) {
        complex z1 = new complex(z.real(), -z.imaginary());
        return z1;
    }

    public static complex substract(complex left, complex right) { // return substraction of two complex numbers
        complex result;
        result = new complex((left.real() - right.real()),
                (left.imaginary() - right.imaginary()));
        return result;
    }

    public static complex multiply(complex left, complex right) { // return multiplication of two complex numbers
        complex result;
        result = new complex((left.real() * right.real() - left.imaginary() * right.imaginary()),
                (left.real() * right.imaginary() + left.imaginary() * right.real()));
        return result;
    }

    public static complex multiply(complex left, double right) {
        complex result;
        result = new complex((left.real() * right), (left.imaginary() * right));
        return result;
    }

    public static complex divide(complex left, complex right) { // return division of two complex numbers
        double a = right.real() * right.real() + right.imaginary() * right.imaginary();
        complex result;
        result = new complex((left.real() * right.real() + left.imaginary() * right.imaginary()) / a,
                (-left.real() * right.imaginary() + left.imaginary() * right.real()) / a);
        return result;
    }

    public static complex divide(complex left, double right) { // return division of a complex number to a real number
        complex result;
        result = new complex((left.real() / right), (left.imaginary() / right));
        return result;
    }

    public static complex pow(complex left, double right) { // return sum of two complex numbers
        double Rad, th;
        Rad = Math.pow(left.R(), right);
        th = right * left.theta();
        complex result;
        result = new complex((Rad * Math.cos(th)),
                (Rad * Math.sin(th)));
        return result;
    }

    public static complex exp(complex left) { // exp(x+i*y)
        complex result;
        result = new complex((Math.exp(left.real()) * Math.cos(left.imaginary())),
                (Math.exp(left.real()) * Math.sin(left.imaginary())));
        return result;
    }

    public static complex exp(double left) { // exp(i*left) imaginary exponent
        complex result;
        result = new complex(Math.cos(left),
                Math.sin(left));
        return result;
    }

    public static complex sqrt(complex left) {
        return pow(left, 0.5);
    }

    public static double abs(complex left) {
        return left.R();
    }

    public boolean smaller(complex left, complex right) {
// less then comparison of two complex numbers
        return (left.R() < right.R());
    }

    public boolean smaller_equal(complex left, complex right) {
// less then and equal comparison of two complex numbers
        return (left.R() <= right.R());
    }

    public boolean greater(complex left, complex right) {
// greater then comparison of two complex numbers
        return left.R() > right.R();
    }

    public boolean greater_equal(complex left, complex right) {
// greater then and equal comparison of two complex numbers
        return left.R() >= right.R();
    }

    public boolean equal(complex left, complex right) {
// equal comparison of two complex numbers
        return left.R() == right.R();
    }

    public boolean not_equal(complex left, complex right) {
// not equal comparison of two complex numbers
        return left.R() != right.R();
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
        return toString(left, 15, 10);
    }

    public static String toString(complex value) {
        String b = "";
        if (Math.abs(value.imaginary()) != 1) {
            if (value.imaginary() >= 0) {
                b = b + "(" + toString(value.real()) + " + " + toString(value.imaginary()) + "i )";
            } else {
                b = b + "(" + toString(value.real()) + " - " + toString(-value.imaginary()) + "i )";
            }
        } else if (value.imaginary() >= 0) {
            b = b + "(" + toString(value.real()) + " + i )";
        } else {
            b = b + "(" + toString(value.real()) + " - i )";
        }
        return b;
    }

    public String toString() {
        String b = "";
        if (Math.abs(imaginary()) != 1) {
            if (imaginary() >= 0) {
                b = b + "(" + toString(real()) + " + " + toString(imaginary()) + "i )";
            } else {
                b = b + "(" + toString(real()) + " - " + toString(-imaginary()) + "i )";
            }
        } else if (imaginary() >= 0) {
            b = b + "(" + toString(real()) + " + i )";
        } else {
            b = b + "(" + toString(real()) + " - i )";
        }
        return b;
    }

    public static complex toComplex(String s) {
        //bu metod compleks sayýyý ekrandan okur.
        //StringTokanizer kütüphane sýnýfý bir stringi cümlelere ayýrýr    
        String s1 = JOptionPane.showInputDialog(s);
        StringTokenizer token = new StringTokenizer(s1);
        int n = token.countTokens() - 1;
        int m = n + 1;
        double a[] = new double[m];
        int j = 0;
        while (token.hasMoreTokens()) {
            Double ax = new Double(token.nextToken());
            a[j++] = ax.doubleValue();
        }
        complex b = new complex(a[0], a[1]);
        return b;
    }
    // data areas
    public double nreal;
    public double nimag;

};
