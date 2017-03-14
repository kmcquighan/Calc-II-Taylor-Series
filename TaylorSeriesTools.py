# -*- coding: utf-8 -*-
"""
Copyright Kelly McQuighan 2016

These tools can be used to visualize Taylor Series.
"""

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from fractions import Fraction
from sympy import *
import sympy
import decimal
mpl.rcParams['font.size'] = 17
colors = ['#A90000', '#D06728', '#D9A621', '#008040', '#0080FF', '#7B00F1']
alph = 0.4; alph2 = 0.7; alph3 = 1.0

def polynomial(f, a, x0, n, xmin, xmax, ymin, ymax, display_polynomial):     
    
    nx = 50; nmax=17
    nc = len(colors)

    if a>0:
        xtext = r'$(x-%.1f)$' %a
    elif a<0:
        xtext = r'$(x+%.1f)$' %np.abs(a)
    else:
        xtext = r'$x$'
    
    assert xmax>xmin
    
    x = sympy.symbols('x', real=True)  
    xs = np.linspace(xmin,xmax,nx)

    y = eval(f)
    func = sympy.lambdify(x, y, 'numpy')

    fxs = func(xs)
    fa = float(func(a))
    fx = fa*np.ones((nx,1))
    sn = func(a)  
    
    if fa!=0:
        nterms = 1
    else:
        nterms = 0
    
    if display_polynomial:    
        fa_frac = Fraction(fa)
        if fa_frac.denominator < 10**10:
            frac = Fraction(fa)           
            num = frac.numerator
            denom = frac.denominator
            e=0
        else:
            num = fa
            denom = 1
            e = np.min([abs(decimal.Decimal(str(num)).as_tuple().exponent),5])
        if denom!=1:
            if fa>=0:
                fxtext = r'$\frac{'+'{0:.{1}f}'.format(np.abs(num), e) +r'}{%d}$'%denom 
            else:
                fxtext = r'-$\frac{'+'{0:.{1}f}'.format(np.abs(num), e) +r'}{%d}$'%denom
        else:
            if fa>0:
                fxtext = r'{0:.{1}f} '.format(num, e)
            elif fa<0:
                fxtext = r'-{0:.{1}f} '.format(np.abs(num), e)
            else:
                fxtext = '0'
    xspow = np.ones((nx,1))
    diff = np.matrix(xs-a).T
        
    plt.figure(figsize=(20, 10))
    plt.rc('xtick', labelsize=26) 
    plt.rc('ytick', labelsize=26)

    ax0 = plt.subplot2grid((10,2), (0, 0), colspan=2)
    ax1 = plt.subplot2grid((10,2), (1, 0), rowspan=6)
    ax2 = plt.subplot2grid((10,2), (1, 1), rowspan=6)
    ax00 = plt.subplot2grid((10,2), (7, 0), colspan=2, rowspan=3)
    ax0.axis('off')
    ax00.axis('off')
    plt.tight_layout(pad=5.0, w_pad=5.0, h_pad=5.0)
    
    ax1.plot(a,func(a),'ko',markersize=15)
    ax1.plot(xs,fx,colors[0],linewidth=5, alpha=alph)
    ax1.plot(x0,sn,colors[0],marker='o', markeredgecolor=colors[0], linewidth='3', markersize=15,alpha=alph)
    ax2.plot(0,sn,colors[0],marker='o', markeredgecolor=colors[0], linewidth='3', markersize=15, alpha=alph2)
    
    ax1.set_xlim([xmin,xmax])
    ax2.set_xlim([-1,nmax])
    
    ax1.set_xlabel('x', fontsize=26)
    ax1.set_title(r'$p_n(x)$', fontsize=26)
    ax2.set_xlabel('n', fontsize=26)
    ax2.set_title(r'$p_n(%.1f)$'%x0,fontsize=26)
    ax0.text(0.5, 0.3, 'f(x) = '+f+ ', Taylor polynomials at a=%.1f' %a, ha='center', va='bottom', fontsize=36, transform=ax0.transAxes)
    
    ax1.plot(xs,fxs,'k',linewidth=5, label=r'$f(x)$')      
    ax2.plot(np.linspace(-1,nmax,nx), func(x0)*np.ones((nx,1)),'k:',linewidth=2, label=r'$f(%.1f)$' %x0)  
    
    for i in range(1,n+1):
        y = y.diff(x)
        fprime = sympy.lambdify(x, y, 'numpy')
        fpa = float(fprime(a))
        fpa_frac = Fraction(fpa)
        c = fpa/sympy.factorial(i)
        xspow = np.multiply(xspow,diff)
        fx =  fx+np.matrix(c*xspow)
        sn += c*(x0-a)**i
        
        cidx = np.mod(i,nc)
        ax1.plot(xs,fx,colors[cidx],linewidth=5, alpha=alph)
        ax1.plot(x0,sn,colors[cidx], markeredgecolor=colors[cidx], linewidth='3', marker='o',markersize=15, alpha=alph)
        ax2.plot(i,sn,colors[cidx], markeredgecolor=colors[cidx], linewidth='3', marker='o',markersize=15,alpha=alph2)
        
        if display_polynomial:
            if fpa_frac.denominator < 10**10:
                frac = Fraction(fpa)*Fraction(1, int(sympy.factorial(i)))               
                num = frac.numerator
                denom = frac.denominator
                e=0
            else:
                num = fpa
                denom = int(sympy.factorial(i))
                e = np.min([abs(decimal.Decimal(str(num)).as_tuple().exponent),5])
                
            if c>0:
                termsign = '+'
            elif c<0:
                termsign = '-'
            else:
                termsign = '0'
            
            if termsign!='0':
                nterms +=1
                if a!=0:
                    if nterms==5 or nterms == 9 or nterms==13 or nterms==16:
                        fxtext = fxtext+'\n\t\t\t\t\t     '
                elif nterms==11:
                        fxtext = fxtext+'\n\t\t\t\t  '
                if denom!=1:
                    if fxtext=='0' and termsign=='+':
                        fxtext = r'$\frac{'+'{0:.{1}f}'.format(np.abs(num), e) +r'}{%d}$'%denom +xtext
                    elif fxtext=='0' and termsign=='-':
                        fxtext = r'-$\frac{'+'{0:.{1}f}'.format(np.abs(num), e) +r'}{%d}$'%denom +xtext
                    else:
                        fxtext = fxtext+termsign+r' $\frac{'+'{0:.{1}f}'.format(np.abs(num), e) +r'}{%d}$'%denom +xtext
                elif np.abs(num)!=1:
                    if fxtext=='0' and termsign=='+':
                        fxtext = r'{0:.{1}f}'.format(num, e)+xtext
                    elif fxtext=='0' and termsign=='-':
                        fxtext = r'-{0:.{1}f}'.format(np.abs(num), e)+xtext
                    else:
                        fxtext = fxtext+termsign+r' {0:.{1}f}'.format(np.abs(num), e) +xtext
                else:
                    if fxtext=='0' and termsign=='+':
                        fxtext = xtext
                    elif fxtext=='0' and termsign=='-':
                        fxtext = '-'+xtext
                    else:
                        fxtext = fxtext+termsign+' '+xtext
                if i>1:                   
                    if termsign!='0':
                        fxtext = fxtext+r'$^{%d}$ ' %i
                else:
                    if termsign!='0':
                        fxtext = fxtext+' '

    if display_polynomial:
        ax00.text(0.0, 1., r'$p_{%d}(x)$' %n +r' = $\sum_{k=0}^{%d} \frac{f^{(k)}(%.1f)}{k!}$' %(n,a)+xtext+r'$^k$ = '+ fxtext, ha='left', va='top', fontsize=26, transform=ax00.transAxes)    
    ax1.axhline(y=0, color='k', linewidth=1)
    ax1.axvline(x=0, color='k', linewidth=1)
    ax2.axhline(y=0, color='k', linewidth=1)
    ax2.axvline(x=0, color='k', linewidth=1)
    
    ax1.set_ylim([ymin,ymax])
    ax2.set_ylim([ymin,ymax])

    cidx = np.mod(n,nc)
    
    ax1.plot(xs,fx,colors[cidx],linewidth=5,alpha=alph3,label=r'$p_{%d}(x)$' %n)  
    ax1.plot(x0,sn,colors[cidx], markeredgecolor=colors[cidx], linewidth='3', marker='o', markersize=15, alpha=1.0)   
    ax2.plot(n,sn,colors[cidx], markeredgecolor=colors[cidx], linestyle = 'None', marker='o',markersize=15, label=r'$p_{%d}(%.1f)$' %(n,x0))
          
    ax1.legend(fontsize=26, loc=0)
    ax2.legend(fontsize=26, loc=0, numpoints=1)

def remainder(f, a, x0, n, xmin, xmax, ymin, ymax):     
    
    nx = 50; nmax=17
    nc = len(colors)
    #alph = 0.4; alph2 = 0.8; alph3 = 0.9
    
    assert xmax>xmin
    
    x = sympy.symbols('x', real=True)  
    xs = np.linspace(xmin,xmax,nx)

    y = eval(f)
    func = sympy.lambdify(x, y, 'numpy')

    fxs = func(xs)
    fx0 = func(x0)
    fxsmat = np.matrix(fxs).T
    fa = float(func(a))
    fx = fa*np.ones((nx,1))
    sn = func(a)

    xspow = np.ones((nx,1))
    diff = np.matrix(xs-a).T
        
    plt.figure(figsize=(20, 7))
    plt.rc('xtick', labelsize=26) 
    plt.rc('ytick', labelsize=26)

    ax0 = plt.subplot2grid((7,2), (0, 0), colspan=2)
    ax3 = plt.subplot2grid((7,2), (1, 0), rowspan=6)
    ax4 = plt.subplot2grid((7,2), (1, 1), rowspan=6)
    ax0.axis('off')
    plt.tight_layout(pad=5.0, w_pad=5.0, h_pad=5.0)

    ax3.plot(a,0,'k*',markersize=20)
    ax3.plot(xs,fxsmat-fx,colors[0], ls='--', linewidth=5, alpha=alph)
    ax3.plot(x0,fx0-sn,colors[0],marker='*', markeredgecolor=colors[0], linewidth='3', markersize=20,alpha=alph)
    ax4.plot(0,fx0-sn,colors[0],marker='*', markeredgecolor=colors[0], linewidth='3', markersize=20, alpha=alph2)    

    ax3.set_xlim([xmin,xmax])
    ax4.set_xlim([-1,nmax])
    
    ax3.set_xlabel('x', fontsize=26)
    ax3.set_title(r'$R_n(x)$', fontsize=26)
    ax4.set_xlabel('n', fontsize=26)
    ax4.set_title(r'$R_n(%.1f)$'%x0,fontsize=26) 
    ax0.text(0.5, 0.3, 'f(x) = '+f+ ', Taylor polynomial remainders at a=%.1f' %a, ha='center', va='bottom', fontsize=36, transform=ax0.transAxes)
    
    for i in range(1,n+1):
        y = y.diff(x)
        fprime = sympy.lambdify(x, y, 'numpy')
        fpa = float(fprime(a))
        c = fpa/sympy.factorial(i)
        xspow = np.multiply(xspow,diff)
        fx =  fx+np.matrix(c*xspow)
        sn += c*(x0-a)**i

        cidx = np.mod(i,nc)
        ax3.plot(xs,fxsmat-fx,colors[cidx],ls='--', linewidth=5, alpha=alph)
        ax3.plot(x0,fx0-sn,colors[cidx], markeredgecolor=colors[cidx], linewidth='3', marker='*',markersize=20, alpha=alph)
        ax4.plot(i,fx0-sn,colors[cidx], markeredgecolor=colors[cidx], linewidth='3', marker='*',markersize=20,alpha=alph2)
    
    ax3.axhline(y=0, color='k', linewidth=1)
    ax3.axvline(x=0, color='k', linewidth=1)
    ax4.axhline(y=0, color='k', linewidth=1)
    ax4.axvline(x=0, color='k', linewidth=1)

    ax3.set_ylim([ymin,ymax])
    ax4.set_ylim([ymin,ymax])

    cidx = np.mod(n,nc)
    
    ax3.plot(xs,fxsmat-fx,colors[cidx],ls='--',linewidth=5,alpha=alph3, label=r'$R_{%d}(x)$' %n)  
    ax3.plot(x0,fx0-sn,colors[cidx], markeredgecolor=colors[cidx], linewidth='3', marker='*', markersize=20, alpha=1.0)   
    ax4.plot(n,fx0-sn,colors[cidx], markeredgecolor=colors[cidx], linestyle = 'None', marker='*',markersize=20, label=r'$R_{%d}(%.1f)$' %(n,x0))
    
    ax3.legend(fontsize=26, loc=0)
    ax4.legend(fontsize=26, loc=0, numpoints=1)

def error_bound(f, a, x0, n, xmin, xmax):     
    
    nx = 50;
    nc = len(colors)

    if a>0:
        absxtext = r'$|x-%.1f|$' %a
    elif a<0:
        absxtext = r'$|x+%.1f|$' %np.abs(a)
    else:
        absxtext = r'$|x|$'     
    
    assert xmax>xmin
    
    x = sympy.symbols('x', real=True)  
    xs = np.linspace(xmin,xmax,nx)

    y = eval(f)
    func = sympy.lambdify(x, y, 'numpy')

    fxs = func(xs)
    fxsmat = np.matrix(fxs).T
    fa = float(func(a))
    fx = fa*np.ones((nx,1))
    
    xspow = np.ones((nx,1))
    diff = np.matrix(xs-a).T
        
    plt.figure(figsize=(20, 7))
    plt.rc('xtick', labelsize=26) 
    plt.rc('ytick', labelsize=26)

    ax0 = plt.subplot2grid((7,2), (0, 0), colspan=2)
    ax5 = plt.subplot2grid((7,2), (1,0), rowspan=6, colspan=2)
    ax0.axis('off')
    plt.tight_layout(pad=5.0, w_pad=5.0, h_pad=5.0)
    
    ax5.set_xlim([xmin,xmax])
    
    ax5.set_xlabel('x', fontsize=26)
    ax0.text(0.5, 0.3, 'f(x) = '+f+ ', Error bound for Taylor polynomials at a=%.1f' %a, ha='center', va='bottom', fontsize=36, transform=ax0.transAxes)
    
    for i in range(1,n+1):
        y = y.diff(x)
        fprime = sympy.lambdify(x, y, 'numpy')
        fpa = float(fprime(a))
        c = fpa/sympy.factorial(i)
        xspow = np.multiply(xspow,diff)
        fx =  fx+np.matrix(c*xspow)        
        
    z = y.diff(x)
    fprimeN = sympy.lambdify(x,z,'numpy')
    # can only evaluate a lambda function at one x at a time, so I need to implement the maximum myself
    maxfpNxs = np.abs(float(fprimeN(xs[0])))
    for i in range(1,nx):
        fpNx = np.abs(float(fprimeN(xs[i])))
        if fpNx > maxfpNxs:
            maxfpNxs = fpNx
    RNbound = maxfpNxs*np.power(np.abs(diff),(n+1))/sympy.factorial(n+1)
    
    ax5.axhline(y=0, color='k', linewidth=1)
    ax5.axvline(x=0, color='k', linewidth=1)
    
    cidx = np.mod(n,nc)
    
    ax5.plot(xs,np.abs(fxsmat-fx),colors[cidx],ls='--',linewidth=5,alpha=1.0, label=r'$\left|R_{%d}(x)\right|$' %n)  
    if n==0:
        ax5.set_title(r'upper bound on $\left|R_{%d}(x)\right|$, $M=\max_{x\in[%.1f,%.1f]}\left(\left|f\right|\right)\approx$%.5f' 
                      %(n,xmin, xmax, maxfpNxs), fontsize=26, y=1.1)
        ax5.plot(xs,RNbound,'k-.',linewidth=5,alpha=1.0, label=r'$M$' +absxtext)  
    else:
        ax5.set_title(r'upper bound on $\left|R_{%d}(x)\right|$, $M=\max_{x\in[%.1f,%.1f]}\left(\left|f^{(%d)}\right|\right)\approx$%.5e' 
                      %(n+1,xmin, xmax, n+1,maxfpNxs), fontsize=26, y=1.1)
        ax5.plot(xs,RNbound,'k-.',linewidth=5,alpha=1.0, label=r'$\frac{M}{%d!}$' %(n+1)+absxtext+r'$^{%d}$' %(n+1))  
          
    ax5.legend(fontsize=26, loc=0)

def all_tools(f, a, x0, n, xmin, xmax, ymin, ymax, ymin_remainder, ymax_remainder):     

    nx = 50; nmax=17
    nc = len(colors)
    #alph = 0.4; alph2 = 0.8; alph3 = 0.9
    if a>0:
        absxtext = r'$|x-%.1f|$' %a
    elif a<0:
        absxtext = r'$|x+%.1f|$' %np.abs(a)
    else:
        absxtext = r'$|x|$'     
    
    assert xmax>xmin
    
    x = sympy.symbols('x', real=True)  
    xs = np.linspace(xmin,xmax,nx)

    y = eval(f)
    func = sympy.lambdify(x, y, 'numpy')

    fxs = func(xs)
    fx0 = func(x0)
    fxsmat = np.matrix(fxs).T
    fa = float(func(a))
    fx = fa*np.ones((nx,1))
    sn = func(a)
        
    xspow = np.ones((nx,1))
    diff = np.matrix(xs-a).T
        
    plt.figure(figsize=(20, 20))
    plt.rc('xtick', labelsize=26) 
    plt.rc('ytick', labelsize=26)

    ax0 = plt.subplot2grid((20,2), (0, 0), colspan=2)
    ax1 = plt.subplot2grid((20,2), (1, 0), rowspan=6)
    ax2 = plt.subplot2grid((20,2), (1, 1), rowspan=6)
    ax3 = plt.subplot2grid((20,2), (7, 0), rowspan=6)
    ax4 = plt.subplot2grid((20,2), (7, 1), rowspan=6)
    ax000 = plt.subplot2grid((20,2), (13, 0), colspan=2)
    ax5 = plt.subplot2grid((20,2), (14,0), rowspan=6, colspan=2)

    ax0.axis('off')
    ax000.axis('off')
    plt.tight_layout(pad=5.0, w_pad=5.0, h_pad=5.0)
    
    ax1.plot(a,func(a),'ko',markersize=15)
    ax1.plot(xs,fx,colors[0],linewidth=5, alpha=alph)
    ax1.plot(x0,sn,colors[0],marker='o', markeredgecolor=colors[0], linewidth='3', markersize=15,alpha=alph)
    ax2.plot(0,sn,colors[0],marker='o', markeredgecolor=colors[0], linewidth='3', markersize=15, alpha=alph2)
    ax3.plot(a,0,'k*',markersize=20)
    ax3.plot(xs,fxsmat-fx,colors[0], ls='--', linewidth=5, alpha=alph)
    ax3.plot(x0,fx0-sn,colors[0],marker='*', markeredgecolor=colors[0], linewidth='3', markersize=20,alpha=alph)
    ax4.plot(0,fx0-sn,colors[0],marker='*', markeredgecolor=colors[0], linewidth='3', markersize=20, alpha=alph2)
    
    ax1.set_xlim([xmin,xmax])
    ax2.set_xlim([-1,nmax])
    ax3.set_xlim([xmin,xmax])
    ax4.set_xlim([-1,nmax])
    ax5.set_xlim([xmin,xmax])
    
    ax1.set_xlabel('x', fontsize=26)
    ax1.set_title(r'$p_n(x)$', fontsize=26)
    ax2.set_xlabel('n', fontsize=26)
    ax2.set_title(r'$p_n(%.1f)$'%x0,fontsize=26)
    ax3.set_xlabel('x', fontsize=26)
    ax3.set_title(r'$R_n(x)$', fontsize=26)
    ax4.set_xlabel('n', fontsize=26)
    ax4.set_title(r'$R_n(%.1f)$'%x0,fontsize=26) 
    ax5.set_xlabel('x', fontsize=26)
    ax0.text(0.5, 0.3, 'f(x) = '+f+ ', Taylor polynomials at a=%.1f' %a, ha='center', va='bottom', fontsize=36, transform=ax0.transAxes)
    
    ax1.plot(xs,fxs,'k',linewidth=5, label=r'$f(x)$')      
    ax2.plot(np.linspace(-1,nmax,nx), func(x0)*np.ones((nx,1)),'k:',linewidth=5, label=r'$f(%.1f)$' %x0)  
    
    for i in range(1,n+1):
        y = y.diff(x)
        fprime = sympy.lambdify(x, y, 'numpy')
        fpa = float(fprime(a))
        c = fpa/sympy.factorial(i)
        xspow = np.multiply(xspow,diff)
        fx =  fx+np.matrix(c*xspow)
        sn += c*(x0-a)**i
           
        cidx = np.mod(i,nc)
        ax1.plot(xs,fx,colors[cidx],linewidth=5, alpha=alph)
        ax1.plot(x0,sn,colors[cidx], markeredgecolor=colors[cidx], linewidth='3', marker='o',markersize=15, alpha=alph)
        ax2.plot(i,sn,colors[cidx], markeredgecolor=colors[cidx], linewidth='3', marker='o',markersize=15,alpha=alph2)
        ax3.plot(xs,fxsmat-fx,colors[cidx],ls='--', linewidth=5, alpha=alph)
        ax3.plot(x0,fx0-sn,colors[cidx], markeredgecolor=colors[cidx], linewidth='3', marker='*',markersize=20, alpha=alph)
        ax4.plot(i,fx0-sn,colors[cidx], markeredgecolor=colors[cidx], linewidth='3', marker='*',markersize=20,alpha=alph2)

    z = y.diff(x)
    fprimeN = sympy.lambdify(x,z,'numpy')
    maxfpNxs = np.abs(float(fprimeN(xs[0])))
    for i in range(1,nx):
        fpNx = np.abs(float(fprimeN(xs[i])))
        if fpNx > maxfpNxs:
            maxfpNxs = fpNx
    RNbound = maxfpNxs*np.power(np.abs(diff),(n+1))/sympy.factorial(n+1)
    
    ax1.axhline(y=0, color='k', linewidth=1)
    ax1.axvline(x=0, color='k', linewidth=1)
    ax2.axhline(y=0, color='k', linewidth=1)
    ax2.axvline(x=0, color='k', linewidth=1)
    ax3.axhline(y=0, color='k', linewidth=1)
    ax3.axvline(x=0, color='k', linewidth=1)
    ax4.axhline(y=0, color='k', linewidth=1)
    ax4.axvline(x=0, color='k', linewidth=1)
    ax5.axhline(y=0, color='k', linewidth=1)
    ax5.axvline(x=0, color='k', linewidth=1)
    
    ax1.set_ylim([ymin,ymax])
    ax2.set_ylim([ymin,ymax])
    ax3.set_ylim([ymin_remainder,ymax_remainder])
    ax4.set_ylim([ymin_remainder,ymax_remainder])

    cidx = np.mod(n,nc)
    
    ax1.plot(xs,fx,colors[cidx],linewidth=5,alpha=alph3,label=r'$p_{%d}(x)$' %n)  
    ax1.plot(x0,sn,colors[cidx], markeredgecolor=colors[cidx], linewidth='3', marker='o', markersize=15, alpha=1.0)   
    ax2.plot(n,sn,colors[cidx], markeredgecolor=colors[cidx], linestyle = 'None', marker='o',markersize=15, label=r'$p_{%d}(%.1f)$' %(n,x0))
    ax3.plot(xs,fxsmat-fx,colors[cidx],ls='--',linewidth=5,alpha=alph3, label=r'$R_{%d}(x)$' %n)  
    ax3.plot(x0,fx0-sn,colors[cidx], markeredgecolor=colors[cidx], linewidth='3', marker='*', markersize=20, alpha=1.0)   
    ax4.plot(n,fx0-sn,colors[cidx], markeredgecolor=colors[cidx], linestyle = 'None', marker='*',markersize=20, label=r'$R_{%d}(%.1f)$' %(n,x0))
    ax5.plot(xs,np.abs(fxsmat-fx),colors[cidx],ls='--',linewidth=5,alpha=1.0, label=r'$\left|R_{%d}(x)\right|$' %n)  
    if n==0:
        ax5.set_title(r'upper bound on $\left|R_{%d}(x)\right|$, $M=\max_{x\in[%.1f,%.1f]}\left(\left|f\right|\right)\approx$%.3e' 
                      %(n,xmin, xmax, maxfpNxs), fontsize=26, y=1.1)
        ax5.plot(xs,RNbound,'k-.',linewidth=5,alpha=1.0, label=r'$M$' +absxtext)  
    else:
        ax5.set_title(r'upper bound on $\left|R_{%d}(x)\right|$, $M=\max_{x\in[%.1f,%.1f]}\left(\left|f^{(%d)}\right|\right)\approx$%.3e' 
                      %(n+1,xmin, xmax, n+1,maxfpNxs), fontsize=26, y=1.1)
        ax5.plot(xs,RNbound,'k-.',linewidth=5,alpha=1.0, label=r'$\frac{M}{%d!}$' %(n+1)+absxtext+r'$^{%d}$' %(n+1))  
          
    ax1.legend(fontsize=26, loc=0)
    ax2.legend(fontsize=26, loc=0, numpoints=1)
    ax3.legend(fontsize=26, loc=0)
    ax4.legend(fontsize=26, loc=0, numpoints=1)
    ax5.legend(fontsize=26, loc=0)