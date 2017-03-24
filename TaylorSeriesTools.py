# -*- coding: utf-8 -*-
"""
Kelly McQuighan 2017

These tools can be used to visualize Taylor Series and convergence of a Taylor series.
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

"""
These are various helper functions used to correctly format the text if the
polynomial is to be displayed in function polynomial.
"""
def get_xtext(a):
    
    if a>0:
        xtext = r'$(x-%.1f)$' %a
    elif a<0:
        xtext = r'$(x+%.1f)$' %np.abs(a)
    else:
        xtext = r'$x$'
    
    return xtext

def get_absxtext(a):
    
    if a>0:
        absxtext = r'$|x-%.1f|$' %a
    elif a<0:
        absxtext = r'$|x+%.1f|$' %np.abs(a)
    else:
        absxtext = r'$|x|$'
    
    return absxtext

def get_poly_values(fa,i):
    fa_frac = Fraction(fa)
    if fa_frac.denominator < 10**10:
        frac = Fraction(fa)*Fraction(1, int(sympy.factorial(i)))               
        num = frac.numerator
        denom = frac.denominator
        e=0
    else:
        num = fa
        denom = int(sympy.factorial(i))
        e = np.min([abs(decimal.Decimal(str(num)).as_tuple().exponent),5])
    
    return [num,denom,e]

def get_fxtext(num,denom,e,fa):
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
    
    return fxtext

def get_termsign(c):
    if c>0:
        termsign = '+'
    elif c<0:
        termsign = '-'
    else:
        termsign = '0'
        
    return termsign

# Outputs a new line for certain terms. I determined empiracally that these
# were reasonable conditions for outputting a new line. This does not compute
# anything dynamically. Improvements to the program could be made here.
def new_line(nterms):
    if nterms==5 or nterms == 9 or nterms==13 or nterms==16:
            return '\n\t\t\t\t  '
    else:
        return ''

# Outputs everything in the next term: +/- sign, coefficient and the x variable (x-a). 
# The only thing missing is the power of (x-a) which is taken care of in a different
# function. Outputting the coefficient properly is tricky. The cases are:
# - numerator/denominator if denominator!=1
# - numerator if numerator!=1
# - otherwise nothing since this means the coefficient is exactly 1 and
#    it looks ugly to write 1*(x-a)...
def format_polynomial(num,denom,e,zero_fxtext,termsign,xtext):
    
    if denom!=1:
        if zero_fxtext and termsign=='+':
            return r'$\frac{'+'{0:.{1}f}'.format(np.abs(num), e) +r'}{%d}$'%denom +xtext
        elif zero_fxtext and termsign=='-':
            return r'-$\frac{'+'{0:.{1}f}'.format(np.abs(num), e) +r'}{%d}$'%denom +xtext
        else:
            return termsign+r' $\frac{'+'{0:.{1}f}'.format(np.abs(num), e) +r'}{%d}$'%denom +xtext
    elif np.abs(num)!=1:
        if zero_fxtext and termsign=='+':
            return r'{0:.{1}f}'.format(num, e)+xtext
        elif zero_fxtext and termsign=='-':
            return r'-{0:.{1}f}'.format(np.abs(num), e)+xtext
        else:
            return termsign+r' {0:.{1}f}'.format(np.abs(num), e) +xtext
    else:
        if zero_fxtext and termsign=='+':
            return xtext
        elif zero_fxtext and termsign=='-':
            return '-'+xtext
        else:
            return termsign+' '+xtext

# ouputs ^i provided i>1
def format_power(i,termsign):
    if i>1:                   
        if termsign!='0':
            return r'$^{%d}$ ' %i
        else:
            return ''
    else:
        if termsign!='0':
            return ' '
        else: 
            return ''

# relies on the previous 3 functions to update the text properly
def update_text(fxtext,nterms,fa,xtext,i):
    
    [num,denom,e] = get_poly_values(fa,i)
        
    termsign = get_termsign(fa)
    
    if termsign!='0':
        nterms +=1
        
        fxtext += new_line(nterms)
        
        if (fxtext=='0'):
            fxtext = format_polynomial(num,denom,e,True,termsign,xtext)
        else:
            fxtext += format_polynomial(num,denom,e,False,termsign,xtext)
            
        fxtext += format_power(i,termsign)
    
    return [fxtext,nterms]

"""
The way the Notebook is set up the user can run visualize the Taylor series convergence
by looking either just at the polynomial, looking just at the remainder, looking
just at the remainder bound, or looking at all 3. In each of these functions 
the initialization step and successive build steps for the polynomial are the same.
Thus these subprograms handle the building of a Taylor polynomial. In this way,
the only code that is unique to each subprogram is the code that define the axes
and plots the correct functions.

The maximum_Nth_derivative function is used ONLY for computing the error bound.

"""
def init_Taylor(f, a, x0, n, xmin, xmax,nx):
    
    x = sympy.symbols('x', real=True)  
    xs = np.linspace(xmin,xmax,nx)

    y = eval(f)
    func = sympy.lambdify(x, y, 'numpy')

    fxs = np.matrix(func(xs)).T # needed to ensure that the dimensions are correct
    fa = float(func(a))
    fx0 = func(x0)
    pn_xs = fa*np.ones((nx,1))
    sn = func(a)
    xspow = np.ones((nx,1))
    xs_diff = np.matrix(xs-a).T
    
    return [x,xs,y,fxs,fa,fx0,pn_xs,sn,xspow,xs_diff]

def update_Taylor(x,y,a,i,pn_xs,xspow,xs_diff,sn,x0):
    y = y.diff(x)
    func = sympy.lambdify(x, y, 'numpy')
    fa = float(func(a))
    
    c = fa/sympy.factorial(i)
    xspow = np.multiply(xspow,xs_diff)
    pn_xs = pn_xs+ np.matrix(c*xspow)
    sn += c*(x0-a)**i
                
    return [y,fa,xspow,pn_xs,sn]

def maximum_Nplus1th_derivative(x,y,xs):
    # need to take the derivative one more time to 
    z = y.diff(x)
    fprimeN = sympy.lambdify(x,z,'numpy')
    
    # can only evaluate a lambda function at one x at a time, 
    #so I need to implement the maximum myself
    max_fNplus1 = 0
    for this_x in xs:
        fpNx = np.abs(float(fprimeN(this_x)))
        if fpNx > max_fNplus1:
            max_fNplus1 = fpNx
    
    return max_fNplus1

"""
The functions starting here are the functions that the user interacts with in the
IPython Notebook. In all functions the variable names are as follows:
    
Variable names:
    x: symbolic x used for lamdifying f(x) and for differentiating y
    y: the function f(x) (note: f(x) is inputted as text, needs to be converted to a function)
        as the program progresses y turns into successive derivatives of f(x); see the line
        y = y.diff(x)
    a: The number to center the Taylor series about
    fa: stands for the fucntion f evaluated at the point a (a number). As the program 
        progresses this turns into successive f^(i)(a) (the ith derivative of f 
        evaluated at a)
    xs: vector of xs, used for creating the points (x,f(x)) to plot
    fxs: the vector created when evaluating f at the vector xs. This is used to
        plot the actual function for comparison with the Taylor polynomial
    xs_diff: the vector created when subtracting (xs-a)
    xspow: the vector created when raising (xs-a) to the power i 
           It is created by successively multiplying the term xs_diff i times.
    pn_xs: the vector determined by evaluating the Taylor polynomial
          (in mathematics typically denoted p_n(x)) evaluated at the points in
          the vector xs
    sn: the value of pn_xs evaluated at a SPECIFIC choice of x=x_0. The name comes from
        the fact that this notebook is all about understanding pointwise convergence
        of the Taylor series; this means that we want to understand the convergence
        of the infinite series obtained when evaluating the Taylor series at a point.
        In mathematics the notation for the partial sums in an infinite series is s_n.
    x0: the point that the Taylor series is evaluated at in order to visualize
        poitnwise convergence at
    fx0: the function f(x) evaluated at x0

"""

"""
This function plots the Taylor polynomial approximation as well as the evaluation of
the approximation at a specific value of x. When "display_polynomial" is set to True,
it also constructs and displays the Taylor polynomial. 
"""
def polynomial(f, a, x0, n, xmin, xmax, ymin, ymax, display_polynomial):     
    
    nx = 50; nmax=17
    nc = len(colors)

    assert xmax>xmin
    xtext = get_xtext(a)
    
    [x,
     xs,
     y,
     fxs,
     fa,
     fx0,
     pn_xs,
     sn,
     xspow,
     xs_diff] = init_Taylor(f, a, x0, n, xmin, xmax,nx)
    
    
    if display_polynomial:    
        [num,denom,e] = get_poly_values(fa,0)        
        fxtext = get_fxtext(num,denom,e,fa)
        # this is for pretty output: don't print a "0"
        if fa!=0:
            nterms = 1
        else:
            nterms = 0
        
    plt.figure(figsize=(20, 10))
    plt.rc('xtick', labelsize=26) 
    plt.rc('ytick', labelsize=26)

    axTitle = plt.subplot2grid((10,2), (0, 0), colspan=2)
    ax1 = plt.subplot2grid((10,2), (1, 0), rowspan=6)
    ax2 = plt.subplot2grid((10,2), (1, 1), rowspan=6)
    axPoly = plt.subplot2grid((10,2), (7, 0), colspan=2, rowspan=3)
    axTitle.axis('off')
    axPoly.axis('off')
    plt.tight_layout(pad=5.0, w_pad=5.0, h_pad=5.0)
    
    ax1.plot(a,fa,'ko',markersize=15)
    ax1.plot(xs,pn_xs,colors[0],linewidth=5, alpha=alph)
    ax1.plot(x0,sn,colors[0],marker='o', markeredgecolor=colors[0], linewidth=3, markersize=15,alpha=alph)
    ax2.plot(0,sn,colors[0],marker='o', markeredgecolor=colors[0], linewidth=3, markersize=15, alpha=alph2)
    
    ax1.set_xlim([xmin,xmax])
    ax2.set_xlim([-1,nmax])
    
    ax1.set_xlabel('x', fontsize=26)
    ax1.set_title(r'$p_n(x)$', fontsize=26)
    ax2.set_xlabel('n', fontsize=26)
    ax2.set_title(r'$p_n(%.1f)$'%x0,fontsize=26)
    axTitle.text(0.5, 0.3, 'f(x) = '+f+ ', Taylor polynomials at a=%.1f' %a, ha='center', va='bottom', fontsize=36, transform=axTitle.transAxes)
    
    ax1.plot(xs,fxs,'k',linewidth=5, label=r'$f(x)$')      
    ax2.plot(np.linspace(-1,nmax,nx), fx0*np.ones((nx,1)),'k:',linewidth=2, label=r'$f(%.1f)$' %x0)  
    
    for i in range(1,n+1):
        [y,fa,xspow,pn_xs,sn] = update_Taylor(x,y,a,i,pn_xs,xspow,xs_diff,sn,x0)
        
        cidx = np.mod(i,nc)
        ax1.plot(xs,pn_xs,colors[cidx],linewidth=5, alpha=alph)
        ax1.plot(x0,sn,colors[cidx], markeredgecolor=colors[cidx], linewidth=3, marker='o',markersize=15, alpha=alph)
        ax2.plot(i,sn,colors[cidx], markeredgecolor=colors[cidx], linewidth=3, marker='o',markersize=15,alpha=alph2)
        
        # Only construct the polynomial if the "display polynomial" option is turned on
        if display_polynomial:            
            [fxtext,nterms] = update_text(fxtext,nterms,fa,xtext,i)

    if display_polynomial:
        axPoly.text(0.0, 1., r'$p_{%d}(x)$' %n +r' = $\sum_{k=0}^{%d} \frac{f^{(k)}(%.1f)}{k!}$' %(n,a)+xtext+r'$^k$ = '+ fxtext, ha='left', va='top', fontsize=26, transform=axPoly.transAxes)    
    ax1.axhline(y=0, color='k', linewidth=1)
    ax1.axvline(x=0, color='k', linewidth=1)
    ax2.axhline(y=0, color='k', linewidth=1)
    ax2.axvline(x=0, color='k', linewidth=1)
    
    ax1.set_ylim([ymin,ymax])
    ax2.set_ylim([ymin,ymax])

    cidx = np.mod(n,nc)
    
    ax1.plot(xs,pn_xs,colors[cidx],linewidth=5,alpha=alph3,label=r'$p_{%d}(x)$' %n)  
    ax1.plot(x0,sn,colors[cidx], markeredgecolor=colors[cidx], linewidth=3, marker='o', markersize=15, alpha=1.0)   
    ax2.plot(n,sn,colors[cidx], markeredgecolor=colors[cidx], linestyle = 'None', marker='o',markersize=15, label=r'$p_{%d}(%.1f)$' %(n,x0))
          
    ax1.legend(fontsize=26, loc=0)
    ax2.legend(fontsize=26, loc=0, numpoints=1)

"""
This function plots the remainder function as well as the value of the remainder
for a specific choice of x.
"""
def remainder(f, a, x0, n, xmin, xmax, ymin, ymax):     
    
    nx = 50; nmax=17
    nc = len(colors)
    
    assert xmax>xmin
    
    [x,
     xs,
     y,
     fxs,
     fa,
     fx0,
     pn_xs,
     sn,
     xspow,
     xs_diff] = init_Taylor(f, a, x0, n, xmin, xmax,nx)
        
    plt.figure(figsize=(20, 7))
    plt.rc('xtick', labelsize=26) 
    plt.rc('ytick', labelsize=26)

    axTitle = plt.subplot2grid((7,2), (0, 0), colspan=2)
    ax3 = plt.subplot2grid((7,2), (1, 0), rowspan=6)
    ax4 = plt.subplot2grid((7,2), (1, 1), rowspan=6)
    axTitle.axis('off')
    plt.tight_layout(pad=5.0, w_pad=5.0, h_pad=5.0)

    ax3.plot(a,0,'k*',markersize=20)
    ax3.plot(xs,fxs-pn_xs,colors[0], ls='--', linewidth=5, alpha=alph)
    ax3.plot(x0,fx0-sn,colors[0],marker='*', markeredgecolor=colors[0], linewidth=3, markersize=20,alpha=alph)
    ax4.plot(0,fx0-sn,colors[0],marker='*', markeredgecolor=colors[0], linewidth=3, markersize=20, alpha=alph2)    

    ax3.set_xlim([xmin,xmax])
    ax4.set_xlim([-1,nmax])
    
    ax3.set_xlabel('x', fontsize=26)
    ax3.set_title(r'$R_n(x)$', fontsize=26)
    ax4.set_xlabel('n', fontsize=26)
    ax4.set_title(r'$R_n(%.1f)$'%x0,fontsize=26) 
    axTitle.text(0.5, 0.3, 'f(x) = '+f+ ', Taylor polynomial remainders at a=%.1f' %a, ha='center', va='bottom', fontsize=36, transform=axTitle.transAxes)
    
    for i in range(1,n+1):
        [y,fa,xspow,pn_xs,sn] = update_Taylor(x,y,a,i,pn_xs,xspow,xs_diff,sn,x0)

        cidx = np.mod(i,nc)
        ax3.plot(xs,fxs-pn_xs,colors[cidx],ls='--', linewidth=5, alpha=alph)
        ax3.plot(x0,fx0-sn,colors[cidx], markeredgecolor=colors[cidx], linewidth=3, marker='*',markersize=20, alpha=alph)
        ax4.plot(i,fx0-sn,colors[cidx], markeredgecolor=colors[cidx], linewidth=3, marker='*',markersize=20,alpha=alph2)
    
    ax3.axhline(y=0, color='k', linewidth=1)
    ax3.axvline(x=0, color='k', linewidth=1)
    ax4.axhline(y=0, color='k', linewidth=1)
    ax4.axvline(x=0, color='k', linewidth=1)

    ax3.set_ylim([ymin,ymax])
    ax4.set_ylim([ymin,ymax])

    cidx = np.mod(n,nc)
    
    ax3.plot(xs,fxs-pn_xs,colors[cidx],ls='--',linewidth=5,alpha=alph3, label=r'$R_{%d}(x)$' %n)  
    ax3.plot(x0,fx0-sn,colors[cidx], markeredgecolor=colors[cidx], linewidth=3, marker='*', markersize=20, alpha=1.0)   
    ax4.plot(n,fx0-sn,colors[cidx], markeredgecolor=colors[cidx], linestyle = 'None', marker='*',markersize=20, label=r'$R_{%d}(%.1f)$' %(n,x0))
    
    ax3.legend(fontsize=26, loc=0)
    ax4.legend(fontsize=26, loc=0, numpoints=1)
    

"""
This function plots the error bound of the remainder as well as the remainder 
itself. The error bound is computed using Taylor's Remainder Theorem. 

|R_n(x)| <= M/(n+1)! * (x-a)^(n+1) where M = sup |f^(n+1)(x)|

Thus, in addition to the Taylor series stuff used in the previous two functions
which computes the TRUE error, I need an additional variable for the bound.
This is the variable RNbound. M in the formula is the varible max_fNplus1
"""
def error_bound(f, a, x0, n, xmin, xmax):     
    
    nx = 50;
    nc = len(colors)
    
    assert xmax>xmin
    
    [x,
     xs,
     y,
     fxs,
     fa,
     fx0,
     pn_xs,
     sn,
     xspow,
     xs_diff] = init_Taylor(f, a, x0, n, xmin, xmax,nx)
        
    plt.figure(figsize=(20, 7))
    plt.rc('xtick', labelsize=26) 
    plt.rc('ytick', labelsize=26)

    axTitle = plt.subplot2grid((7,2), (0, 0), colspan=2)
    ax5 = plt.subplot2grid((7,2), (1,0), rowspan=6, colspan=2)
    axTitle.axis('off')
    plt.tight_layout(pad=5.0, w_pad=5.0, h_pad=5.0)
    
    ax5.set_xlim([xmin,xmax])
    
    ax5.set_xlabel('x', fontsize=26)
    axTitle.text(0.5, 0.3, 'f(x) = '+f+ ', Error bound for Taylor polynomials at a=%.1f' %a, ha='center', va='bottom', fontsize=36, transform=axTitle.transAxes)
    
    for i in range(1,n+1):
        [y,fa,xspow,pn_xs,sn] = update_Taylor(x,y,a,i,pn_xs,xspow,xs_diff,sn,x0)      
        
    max_fNplus1 = maximum_Nplus1th_derivative(x,y,xs)
    RNbound = max_fNplus1*np.power(np.abs(xs_diff),(n+1))/sympy.factorial(n+1)
    
    ax5.axhline(y=0, color='k', linewidth=1)
    ax5.axvline(x=0, color='k', linewidth=1)
    
    cidx = np.mod(n,nc)
    
    ax5.plot(xs,np.abs(fxs-pn_xs),colors[cidx],ls='--',linewidth=5,alpha=1.0, label=r'$\left|R_{%d}(x)\right|$' %n)  
    
    absxtext = get_absxtext(a) 
    if n==0:
        ax5.set_title(r'upper bound on $\left|R_{%d}(x)\right|$, $M=\max_{x\in[%.1f,%.1f]}\left(\left|f\right|\right)\approx$%.5f' 
                      %(n,xmin, xmax, max_fNplus1), fontsize=26, y=1.1)
        ax5.plot(xs,RNbound,'k-.',linewidth=5,alpha=1.0, label=r'$M$' +absxtext)  
    else:
        ax5.set_title(r'upper bound on $\left|R_{%d}(x)\right|$, $M=\max_{x\in[%.1f,%.1f]}\left(\left|f^{(%d)}\right|\right)\approx$%.5e' 
                      %(n+1,xmin, xmax, n+1,max_fNplus1), fontsize=26, y=1.1)
        ax5.plot(xs,RNbound,'k-.',linewidth=5,alpha=1.0, label=r'$\frac{M}{%d!}$' %(n+1)+absxtext+r'$^{%d}$' %(n+1))  
          
    ax5.legend(fontsize=26, loc=0)

"""
This function plots everything: The Taylor series approximation, the error, and
the error bound, all on one plot. This makes it easier to visualize how each
object is related to the others. However, due to the number of number of computations
the interactive update in the IPython notebook is slow. 
"""
def all_tools(f, a, x0, n, xmin, xmax, ymin, ymax, ymin_remainder, ymax_remainder):     

    nx = 50; nmax=17
    nc = len(colors)
    
    assert xmax>xmin
    
    [x,
     xs,
     y,
     fxs,
     fa,
     fx0,
     pn_xs,
     sn,
     xspow,
     xs_diff] = init_Taylor(f, a, x0, n, xmin, xmax,nx)
        
    plt.figure(figsize=(20, 20))
    plt.rc('xtick', labelsize=26) 
    plt.rc('ytick', labelsize=26)

    axTitle = plt.subplot2grid((20,2), (0, 0), colspan=2)
    ax1 = plt.subplot2grid((20,2), (1, 0), rowspan=6)
    ax2 = plt.subplot2grid((20,2), (1, 1), rowspan=6)
    ax3 = plt.subplot2grid((20,2), (7, 0), rowspan=6)
    ax4 = plt.subplot2grid((20,2), (7, 1), rowspan=6)
    axPoly0 = plt.subplot2grid((20,2), (13, 0), colspan=2)
    ax5 = plt.subplot2grid((20,2), (14,0), rowspan=6, colspan=2)

    axTitle.axis('off')
    axPoly0.axis('off')
    plt.tight_layout(pad=5.0, w_pad=5.0, h_pad=5.0)
    
    ax1.plot(a,fa,'ko',markersize=15)
    ax1.plot(xs,pn_xs,colors[0],linewidth=5, alpha=alph)
    ax1.plot(x0,sn,colors[0],marker='o', markeredgecolor=colors[0], linewidth=3, markersize=15,alpha=alph)
    ax2.plot(0,sn,colors[0],marker='o', markeredgecolor=colors[0], linewidth=3, markersize=15, alpha=alph2)
    ax3.plot(a,0,'k*',markersize=20)
    ax3.plot(xs,fxs-pn_xs,colors[0], ls='--', linewidth=5, alpha=alph)
    ax3.plot(x0,fx0-sn,colors[0],marker='*', markeredgecolor=colors[0], linewidth=3, markersize=20,alpha=alph)
    ax4.plot(0,fx0-sn,colors[0],marker='*', markeredgecolor=colors[0], linewidth=3, markersize=20, alpha=alph2)
    
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
    axTitle.text(0.5, 0.3, 'f(x) = '+f+ ', Taylor polynomials at a=%.1f' %a, ha='center', va='bottom', fontsize=36, transform=axTitle.transAxes)
    
    ax1.plot(xs,fxs,'k',linewidth=5, label=r'$f(x)$')      
    ax2.plot(np.linspace(-1,nmax,nx), fx0*np.ones((nx,1)),'k:',linewidth=5, label=r'$f(%.1f)$' %x0)  
    
    for i in range(1,n+1):
        [y,fa,xspow,pn_xs,sn] = update_Taylor(x,y,a,i,pn_xs,xspow,xs_diff,sn,x0)
           
        cidx = np.mod(i,nc)
        ax1.plot(xs,pn_xs,colors[cidx],linewidth=5, alpha=alph)
        ax1.plot(x0,sn,colors[cidx], markeredgecolor=colors[cidx], linewidth=3, marker='o',markersize=15, alpha=alph)
        ax2.plot(i,sn,colors[cidx], markeredgecolor=colors[cidx], linewidth=3, marker='o',markersize=15,alpha=alph2)
        ax3.plot(xs,fxs-pn_xs,colors[cidx],ls='--', linewidth=5, alpha=alph)
        ax3.plot(x0,fx0-sn,colors[cidx], markeredgecolor=colors[cidx], linewidth=3, marker='*',markersize=20, alpha=alph)
        ax4.plot(i,fx0-sn,colors[cidx], markeredgecolor=colors[cidx], linewidth=3, marker='*',markersize=20,alpha=alph2)

    max_fNplus1 = maximum_Nplus1th_derivative(x,y,xs)
    RNbound = max_fNplus1*np.power(np.abs(xs_diff),(n+1))/sympy.factorial(n+1)
    
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
    
    ax1.plot(xs,pn_xs,colors[cidx],linewidth=5,alpha=alph3,label=r'$p_{%d}(x)$' %n)  
    ax1.plot(x0,sn,colors[cidx], markeredgecolor=colors[cidx], linewidth=3, marker='o', markersize=15, alpha=1.0)   
    ax2.plot(n,sn,colors[cidx], markeredgecolor=colors[cidx], linestyle = 'None', marker='o',markersize=15, label=r'$p_{%d}(%.1f)$' %(n,x0))
    ax3.plot(xs,fxs-pn_xs,colors[cidx],ls='--',linewidth=5,alpha=alph3, label=r'$R_{%d}(x)$' %n)  
    ax3.plot(x0,fx0-sn,colors[cidx], markeredgecolor=colors[cidx], linewidth=3, marker='*', markersize=20, alpha=1.0)   
    ax4.plot(n,fx0-sn,colors[cidx], markeredgecolor=colors[cidx], linestyle = 'None', marker='*',markersize=20, label=r'$R_{%d}(%.1f)$' %(n,x0))
    ax5.plot(xs,np.abs(fxs-pn_xs),colors[cidx],ls='--',linewidth=5,alpha=1.0, label=r'$\left|R_{%d}(x)\right|$' %n)  
    
    absxtext = get_absxtext(a)   
    if n==0:
        ax5.set_title(r'upper bound on $\left|R_{%d}(x)\right|$, $M=\max_{x\in[%.1f,%.1f]}\left(\left|f\right|\right)\approx$%.3e' 
                      %(n,xmin, xmax, max_fNplus1), fontsize=26, y=1.1)
        ax5.plot(xs,RNbound,'k-.',linewidth=5,alpha=1.0, label=r'$M$' +absxtext)  
    else:
        ax5.set_title(r'upper bound on $\left|R_{%d}(x)\right|$, $M=\max_{x\in[%.1f,%.1f]}\left(\left|f^{(%d)}\right|\right)\approx$%.3e' 
                      %(n+1,xmin, xmax, n+1,max_fNplus1), fontsize=26, y=1.1)
        ax5.plot(xs,RNbound,'k-.',linewidth=5,alpha=1.0, label=r'$\frac{M}{%d!}$' %(n+1)+absxtext+r'$^{%d}$' %(n+1))  
          
    ax1.legend(fontsize=26, loc=0)
    ax2.legend(fontsize=26, loc=0, numpoints=1)
    ax3.legend(fontsize=26, loc=0)
    ax4.legend(fontsize=26, loc=0, numpoints=1)
    ax5.legend(fontsize=26, loc=0)
