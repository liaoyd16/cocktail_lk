import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import os

TOT = "test"
PHASE = 3

if __name__ == '__main__':
    os.system("rm -rf ./images/*")
    sample_dirs = []

    # sample_k - pair_i - [sp1, sp2, mix, re1, re2]
    all_samples_path = "results/phase{}/{}/".format(PHASE, TOT)
    for file in os.listdir(all_samples_path):
        if os.path.isdir(os.path.join(all_samples_path, file)): sample_dirs.append(file)

    """
    # phase 2
    for sample_dir in sample_dirs:
        os.system("mkdir {}".format(os.path.join("./images/", sample_dir)))
        print("processing {}".format(os.path.join(all_samples_path, sample_dir)))
        ground_truth = np.array(json.load(
            open(os.path.join(all_samples_path, sample_dir, "ground_truth.json"))
        ))
        recover = np.array(json.load(
            open(os.path.join(all_samples_path, sample_dir, "recover.json"))
        ))

        cv2.imwrite(os.path.join("./images/", sample_dir, "ground_truth.png"), ground_truth * 100)
        cv2.imwrite(os.path.join("./images/", sample_dir, "recover.png"), recover * 100)

    """
    # phase 3
    sample_dirs.sort()
    for sample_dir in sample_dirs:
        os.system("mkdir {}".format(os.path.join("./images/", sample_dir)))
        for pair_dir in os.listdir(os.path.join(all_samples_path, sample_dir)):
            if not os.path.isdir(os.path.join(all_samples_path, sample_dir, pair_dir)):
                continue
            if not os.path.isfile(os.path.join(all_samples_path, sample_dir, pair_dir, "sp1.json")):
                continue

            print( "processing {}".format( os.path.join(all_samples_path, sample_dir, pair_dir) ) )
            sp1_spec = np.array(json.load(
                open(os.path.join(all_samples_path, sample_dir, pair_dir, "sp1.json"))
            ))
            sp2_spec = np.array(json.load(
                open(os.path.join(all_samples_path, sample_dir, pair_dir, "sp2.json"))
            ))
            mixed = np.array(json.load(
                open(os.path.join(all_samples_path, sample_dir, pair_dir, "mix.json"))
            ))
            recover_1 = np.array(json.load(
                open(os.path.join(all_samples_path, sample_dir, pair_dir, "re1.json"))
            ))
            recover_2 = np.array(json.load(
                open(os.path.join(all_samples_path, sample_dir, pair_dir, "re2.json"))
            ))
            recover_none = np.array(json.load(
                open(os.path.join(all_samples_path, sample_dir, pair_dir, "re_none.json"))
            ))

            os.system("mkdir {}".format(os.path.join("./images/", sample_dir, pair_dir)))
            cv2.imwrite(os.path.join("./images/", sample_dir, pair_dir, "sp1.png"), sp1_spec * 100)
            cv2.imwrite(os.path.join("./images/", sample_dir, pair_dir, "sp2.png"), sp2_spec * 100)
            cv2.imwrite(os.path.join("./images/", sample_dir, pair_dir, "mix.png"), mixed * 100)
            cv2.imwrite(os.path.join("./images/", sample_dir, pair_dir, "re1.png"), recover_1 * 100)
            cv2.imwrite(os.path.join("./images/", sample_dir, pair_dir, "re2.png"), recover_2 * 100)
            cv2.imwrite(os.path.join("./images/", sample_dir, pair_dir, "re_none.png"), recover_none * 100)
    # """