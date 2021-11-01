
import os
import sys
sys.path.append("../")
import Meta

def clean_results_in_(phase_no, test_or_train):
    ret = os.system("rm -rf " + os.path.join(Meta.PROJ_ROOT, "../../data2/results/phase{}/{}/*".format(phase_no, test_or_train)))
    print(ret)

def clean_log_dir():
    os.system("rm " + os.path.join(Meta.PROJ_ROOT, "scripts/log/event*"))

def mkdir_in(path_parent, path_child):
    assert(os.path.isdir(path_parent) and ('/' not in path_child))
    if os.path.isdir(os.path.join(path_parent, path_child)):
        os.system("rm -rf " + os.path.join(path_parent, path_child, "*"))
    elif os.path.isfile(os.path.join(path_parent, path_child)):
        os.system("rm " + os.path.join(path_parent, path_child))
    else:
        os.system("mkdir " + os.path.join(path_parent, path_child))

def combine_tops_data():
    for i_pair in range(Meta.data_meta['pairs']):
        sp1 = Meta.data_meta['using_speakers'][i_pair]
        sp2 = Meta.data_meta['using_speakers'][2*Meta.data_meta['pairs'] - 1 - i_pair]
        sp_pair_dir = os.path.join("../../data2/results/tops/sp{}-sp{}/".format(sp1, sp2))
        jsons = os.listdir("../../data2/results/tops/sp{}-sp{}/".format(sp1, sp2))

        # attend: sp1
        tops_sp1 = []
        for js in jsons:
            if "attend{}".format(sp1) in js:
                tops = json.load(open(os.path.join(sp_pair_dir, js)))
                tops_sp1.extend(tops)
        json.dump(tops_sp1, open(os.path.join(sp_pair_dir, "tops_sp{}.json".format(sp1)), "w"))

        # attend: sp2
        tops_sp2 = []
        for js in jsons:
            if "attend{}".format(sp2) in js:
                tops = json.load(open(os.path.join(sp_pair_dir, js)))
                tops_sp2.extend(tops)
        json.dump(tops_sp2, open(os.path.join(sp_pair_dir, "tops_sp{}.json".format(sp2)), "w"))

        # attend: none
        tops_none = []
        for js in jsons:
            if "attendnone" in js:
                tops = json.load(open(os.path.join(sp_pair_dir, js)))
                tops_none.extend(tops)
        json.dump(tops_none, open(os.path.join(sp_pair_dir, "tops_none.json"), "w"))

        os.system("rm {}".format(os.path.join(sp_pair_dir, "attend*.json")))
