import numpy as np
import matplotlib.pyplot as plt
import json
import os
import math

# matplotlib setting
plt.rcParams['figure.dpi'] = 600
plt.rcParams.update({'font.style': 'normal'})
plt.rcParams.update({'font.size': 12})
arfont = {'fontname':'Arial'}
plt.style.use('default')


# functions
def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r


def scat(param1, param2):
    sp1_sp1, sp1_sp2, sp2_sp2, sp2_sp1 = ami_cal(param1, param2)
    plt.scatter(sp1_sp1, sp2_sp1, label = 'Target SP1', s = 10, color = 'red')
    plt.scatter(sp1_sp2, sp2_sp2, label = 'Target SP2', s = 10, color = 'blue')
    plt.plot([0,1], [0,1], color= "black", ls = "--", alpha = 0.55)

    plt.gca().set_aspect('equal', adjustable='box')
    axes = plt.gca()
    axes.set_xlim([-0.1,1.01])
    axes.set_ylim([-0.1,1.01])

    plt.tick_params(
        axis= 'both',          
        bottom= True,      # ticks along the bottom edge are off
        top= False,         # ticks along the top edge are off
        labelbottom= True,
        labelleft = True,
        left = True,
        right = False) # labels along the bottom edge are off

    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.title("{}-{}".format(param1, param2))

    plt.legend(loc = 3)


def ami_cal(param1, param2):
    ami_path = '/home/liaoyuanda/data3/cocktail_new/results/dumped/{}/{}/'.format(param1, param2)
    sample_list = os.listdir(ami_path)

    with open(ami_path + "condition_0.json") as fil:
        spec = np.array(json.load(fil))


    sp1_sp1 = [] # attend SP1, original SP1
    sp1_sp2 = [] # attend SP1, original SP2
    sp2_sp2 = [] # attend SP2, original SP2
    sp2_sp1 = [] # attend SP2, original SP1
    
    ami_list = []
    for i in range(240):
        sp1_sp1.append(corr2(spec[0][i], spec[2][i]))
        sp1_sp2.append(corr2(spec[0][i], spec[3][i]))

        sp2_sp2.append(corr2(spec[1][i], spec[3][i]))
        sp2_sp1.append(corr2(spec[1][i], spec[2][i]))
        
        #print (sp1_sp1, sp1_sp2, sp2_sp2, sp2_sp1)
        
    return np.array(sp1_sp1), np.array(sp1_sp2), np.array(sp2_sp2), np.array(sp2_sp1)



# 產生 ｎ＝５ 之下所有條件的 scatter plot

plt.figure(figsize = (12,8))
plt.subplot(231)
p1 = "n=5_en=1"
p2 = "de=0"

scat(p1, p2)

plt.subplot(232)
p1 = "n=5_en=1"
p2 = "de=en"
scat(p1, p2)

plt.subplot(233)
p1 = "n=5_en=2"
p2 = "de=0"
scat(p1, p2)

plt.subplot(234)
p1 = "n=5_en=2"
p2 = "de=en"
scat(p1, p2)

plt.subplot(235)
p1 = "n=5_en=3"
p2 = "de=0"
scat(p1, p2)

plt.subplot(236)
p1 = "n=5_en=3"
p2 = "de=en"
scat(p1, p2)
plt.savefig("/home/liaoyuanda/data3/cocktail_new/results/figs/n=5.png")
print ("n=5 done")


plt.figure(figsize = (12,8))
plt.subplot(231)
p1 = "n=7_en=1"
p2 = "de=0"
scat(p1, p2)

plt.subplot(232)
p1 = "n=7_en=1"
p2 = "de=en"
scat(p1, p2)

plt.subplot(233)
p1 = "n=7_en=2"
p2 = "de=0"
scat(p1, p2)

plt.subplot(234)
p1 = "n=7_en=2"
p2 = "de=en"
scat(p1, p2)

plt.subplot(235)
p1 = "n=7_en=3"
p2 = "de=0"
scat(p1, p2)

plt.subplot(236)
p1 = "n=7_en=3"
p2 = "de=en"
scat(p1, p2)
plt.savefig("/home/liaoyuanda/data3/cocktail_new/results/figs/n=7.png")
print ("n=7 done")

plt.figure(figsize = (12,8))
plt.subplot(231)
p1 = "n=10_en=1"
p2 = "de=0"
scat(p1, p2)

plt.subplot(232)
p1 = "n=10_en=1"
p2 = "de=en"
scat(p1, p2)

plt.subplot(233)
p1 = "n=10_en=2"
p2 = "de=0"
scat(p1, p2)

plt.subplot(234)
p1 = "n=10_en=2"
p2 = "de=en"
scat(p1, p2)

plt.subplot(235)
p1 = "n=10_en=3"
p2 = "de=0"
scat(p1, p2)

plt.subplot(236)
p1 = "n=10_en=3"
p2 = "de=en"
scat(p1, p2)
plt.savefig("/home/liaoyuanda/data3/cocktail_new/results/figs/n=10.png")
print ("n=10 done")


n_option = ["5", "7", "10"]
en_option = ["1", "2", "3"]
p2_option = ["de=0", "de=en"]

box_list = []

for L1 in n_option: 
    for L2 in en_option:
        for L3 in p2_option:
            p1 = "n={}_en={}".format(L1, L2)
            p2 = L3
            
            sp1_sp1, sp1_sp2, sp2_sp2, sp2_sp1 = ami_cal(p1, p2)
            box_list.append(sp1_sp1 + sp2_sp2 - sp1_sp2 - sp2_sp1)
            
plt.figure(figsize = (12, 6))
box_plot(box_list)
plt.savefig("/home/liaoyuanda/data3/cocktail_new/results/figs/box.png")
print ("box plot done")

