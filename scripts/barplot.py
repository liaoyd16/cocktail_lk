
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import math
import scipy
import scipy.stats
from scipy.optimize import curve_fit

## Matplotlib setting 
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.style': 'normal'})
plt.rcParams.update({'font.size': 15})
arfont = {'fontname':'Arial'}
plt.style.use('default')

## 計算兩張圖之間 correlation 
def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r

def _bar_vals(modelname, jsonname):
    cond_specs_4 = json.load(open(os.path.join(modelname, jsonname)))
    # [a1,a2,s1,s2]
    sp1_sp1 = [] # attend SP1, original SP1
    sp1_sp2 = [] # attend SP1, original SP2
    sp2_sp2 = [] # attend SP2, original SP2
    sp2_sp1 = [] # attend SP2, original SP1

    for i in range(len(cond_specs_4[0])):
        sp1_sp1.append(corr2(cond_specs_4[0][i], cond_specs_4[2][i]))
        sp1_sp2.append(corr2(cond_specs_4[0][i], cond_specs_4[3][i]))

        sp2_sp2.append(corr2(cond_specs_4[1][i], cond_specs_4[3][i]))
        sp2_sp1.append(corr2(cond_specs_4[1][i], cond_specs_4[2][i]))

    return sp1_sp1, sp1_sp2, sp2_sp1, sp2_sp2


def _stars_p(pv):
    if pv > .05: return 'ns'
    if pv > .01: return '*'
    if pv > .001: return '**'
    return '***'

# plot 2x2 bars figure
def ann(ax, text, Xs, Ys):
    x = (Xs[0]+Xs[1])/2
    y = 1.1*max(Ys[0], Ys[1])
    dx = abs(Xs[0]-Xs[1])

    props = {'connectionstyle':'bar','arrowstyle':'-',\
                 'shrinkA':5,'shrinkB':5,'linewidth':2}
    # ax.annotate("{:.2f}".format(Ys[0]), xy=(Xs[0], Ys[0]+.02))
    # ax.annotate("{:.2f}".format(Ys[1]), xy=(Xs[1], Ys[1]+.02))
    ax.annotate(text, xy=((Xs[0]+Xs[1])/2, y+.1), zorder=10, fontsize=15)
    ax.annotate('', xy=(Xs[0],y), xytext=(Xs[1],y), arrowprops=props)

def do_bar(bar_vals): # size 4
    a1r1 = np.mean(bar_vals[0])
    a1r2 = np.mean(bar_vals[1])
    a2r1 = np.mean(bar_vals[2])
    a2r2 = np.mean(bar_vals[3])
    means_a1 = [a1r1, a2r1]
    means_a2 = [a1r2, a2r2]
    x = np.array([0,1])

    total_width = 0.8
    width = total_width / 2
    x = x - (total_width - width) / 2

    fig, ax = plt.subplots()
    fig.set_size_inches(3.5, 3.5)

    plt.xticks(np.arange(2), ["SP1", "SP2"], fontsize=15)
    plt.bar(x, means_a1, width=width, label='attend SP1', color='red')
    plt.bar(x + width, means_a2, width=width, label='attend SP2', color='blue')
    plt.legend(fontsize=15)

    _, pval1 = scipy.stats.ttest_ind(bar_vals[0], bar_vals[1])
    _, pval2 = scipy.stats.ttest_ind(bar_vals[2], bar_vals[3])
    print(pval1, pval2)
    stars1 = _stars_p(pval1)
    stars2 = _stars_p(pval2)

    ann(ax, stars1, [x[0],x[0]+width], [a1r1, a1r2])
    ann(ax, stars2, [x[1],x[1]+width], [a2r1, a2r2])

    plt.ylim([0,1.5])
    plt.yticks(fontsize=15)

def bar(modelname, jsonname):
    bar_vals = _bar_vals(modelname, jsonname) # [a1_r1, a1_r2, a2_r1, a2_r2]
    do_bar(bar_vals)


# plot 2x6 bars and a curve
def sigmoid(x, a, b, c, d):
    return c / (1 + np.exp(- (x - b)/a)) + d

def line(x, a):
    return a

def do_many_bars(bar_vals, f1, f2, p01, p02): #size 2x6
    fig = plt.figure(figsize = (3.5, 3.5))
    plt.subplots_adjust(left=.17, right=.95, bottom=.13, top=0.90)
    ax = fig.add_subplot(111)

    r_1x = [np.mean(vals) for vals in bar_vals[0]]
    r_2x = [np.mean(vals) for vals in bar_vals[1]]
    d_1x = [np.std(vals)  for vals in bar_vals[0]]
    d_2x = [np.std(vals)  for vals in bar_vals[1]]

    # print(r_1x)

    N = len(r_1x)
    ind = np.arange(N)                # the x locations for the groups
    width = 0.35                      # the width of the bars

    ## the bars
    rects1 = ax.bar(ind, r_1x, width,
                    color='red', alpha=0.6,
                    yerr=d_1x,
                    error_kw=dict(elinewidth=1,ecolor='black', capsize=3, alpha=1.))

    rects2 = ax.bar(ind+width, r_2x, width,
                    color='blue', alpha=0.6,
                    yerr=d_2x,
                    error_kw=dict(elinewidth=1,ecolor='black', capsize=3,alpha=1.))

    maxi = max(max(r_1x), max(r_2x))
    mini = min(min(r_1x), min(r_2x))
    popt1, pcov1 = curve_fit(f1, np.arange(6), r_1x)#, p0=[1, 3, maxi, mini])
    popt2, pcov2 = curve_fit(f2, np.arange(6), r_2x)#, p0=[1, 3, maxi, mini])
    xdata = np.linspace(0,5,30)
    print(xdata)
    y1 = [f1(x, *popt1) for x in xdata]
    y2 = [f2(x, *popt2) for x in xdata]
    plt.plot(xdata, y1, 'r--')
    plt.plot(xdata+width, y2, 'b--')

    # axes and labels
    ax.set_xlim(-width,len(ind))
    ax.set_ylim(0, 1)
    # ax.set_ylabel('Correlation with {}')
    # ax.set_title('Correlation by attended layer')
    xTickMarks = ['att L'+str(i) for i in np.arange(N+1,1,-1)]
    ax.set_xticks(ind+width/2)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=15)
    ytickNames = ax.set_yticklabels([0, .2, .4, .6, .8, 1.])
    plt.setp(ytickNames, fontsize=15)

    ## add a legend
    ax.legend( (rects1[0], rects2[0]), ('SP1', 'SP2') , fontsize=15)


if __name__ == '__main__':
    """
    bar("../results/dumped/n=7_en=3/de=en/", "condition_single-tpdwn_6_spec.json")
    plt.savefig("../results/bar.png", dpi=300, bbox_inches='tight')
    """
    
    cumu_single = 'single'
    modelname = "../results/dumped_{}/n=7_en=3_resblock_3x3conv_5shortcut/de=en".format(cumu_single)

    # """
    bar_vals_1x = [[],[]]
    bar_vals_2x = [[],[]]
    for ijson in range(1,7):
        print(ijson)
        r11, r12, r21, r22 = _bar_vals(modelname, "condition_{}-tpdwn_{}_spec.json".format(cumu_single, ijson)) ## 1~6
        bar_vals_1x[0].append(r11)
        bar_vals_1x[1].append(r12)
        bar_vals_2x[0].append(r21)
        bar_vals_2x[1].append(r22)

    json.dump(bar_vals_1x, open(os.path.join(modelname, "bar_vals_1x.json"), "w"))
    json.dump(bar_vals_2x, open(os.path.join(modelname, "bar_vals_2x.json"), "w"))

    bar_vals_1x = json.load(open(os.path.join(modelname, "bar_vals_1x.json")))
    bar_vals_2x = json.load(open(os.path.join(modelname, "bar_vals_2x.json")))

    p0_sigmoid = [1, 4, np.random.randn(), 0.5]
    p0_line = [np.random.randn()]
    do_many_bars(bar_vals_1x, sigmoid, line, p01=p0_sigmoid, p02=p0_line)
    plt.savefig("../results/barplot_1x_{}.png".format(cumu_single), dpi = 300, bbox_inches='tight')
    do_many_bars(bar_vals_2x, line, sigmoid, p01=p0_line, p02=p0_sigmoid)
    plt.savefig("../results/barplot_2x_{}.png".format(cumu_single), dpi = 300, bbox_inches='tight')
    # """
