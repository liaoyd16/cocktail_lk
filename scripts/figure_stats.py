# created: 2021.7.19
# provide stats for figure-3{a,b,c}

from scatter import ami_cal as corr_cal
scatter.cumu_single = 'cumu'

from barplot import _bar_vals

import numpy as np

if __name__ == '__main__':
    # fig3a: scatter.py::ami_cal
    sp1_sp1, sp1_sp2, sp2_sp2, sp2_sp1 = corr_cal("condition_cumu-tpdwn_6_spec")
    N = len(sp1_sp1)
    sp1_diff = sp1_sp1 - sp1_sp2
    sp2_diff = sp2_sp2 - sp2_sp1
    print("sp1: {}\% recoveries on right side".format(100*np.sum(np.where(sp1_diff > 0, 1, 0)) / N))
    print("sp2: {}\% recoveries on right side".format(100*np.sum(np.where(sp2_diff > 0, 1, 0)) / N))

    # fig3b: from spx_spx <- corr_cal
    print("stds: {}, {}, {}, {}"
        .format(np.std(sp1_sp1), np.std(sp1_sp2), np.std(sp2_sp1), np.std(sp2_sp2)))
    print("means: {}, {}, {}, {}"
        .format(np.mean(sp1_sp1), np.mean(sp1_sp2), np.mean(sp2_sp1), np.mean(sp2_sp2)))

    # fig3c: from spx_spx <- corr_cal
    ami_all = sp1_sp1 + sp2_sp2 - sp1_sp2 - sp2_sp1
    print("AMI histogram: mean={}, std={}".format(np.mean(ami_all), np.std(ami_all)))

