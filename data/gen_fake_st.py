import numpy as np
import random

with open("./roc.txt", "r") as fin:
    with open("./roc_shuffle.txt", "w") as fout:
        tmp = []
        for k, line in enumerate(fin):
            i = k + 1
            if i % 6 == 0:
                idx = [0] + np.random.permutation(range(1,5)).tolist()
                for sen in np.take(tmp, idx).tolist():
                    fout.write(sen+"\n")
                tmp = []
                fout.write(line.strip()+"\n")
            else:
                tmp.append(line.strip())
with open("./roc.txt", "r") as fin:
    with open("./roc_repeat.txt", "w") as fout:
        tmp = []
        for k, line in enumerate(fin):
            i = k + 1
            if i % 6 == 0:
                idx = random.randint(1,4)
                tmp[idx] = tmp[idx][:-1] + tmp[idx]
                for sen in tmp:
                    fout.write(sen+"\n")
                tmp = []
                fout.write(line.strip()+"\n")
            else:
                tmp.append(line.strip())
with open("./roc.txt", "r") as fin:
    with open("./roc_replace.txt", "w") as fout:
        post, tmp = [], []
        for k, line in enumerate(fin):
            i = k + 1
            if i % 6 == 0:
                post.append(tmp)
                tmp = []
            else:
                tmp.append(line.strip().split())
        data = {"1":[], "2":[], "3":[], "4":[], "5":[]}
        for p in post:
            for i in range(5):
                data["%d"%(i+1)].append(p[i])
        random_data = data.copy()
        for i in range(5):
            random_data["%d"%(i+1)] = np.random.permutation(random_data["%d"%(i+1)])

        for k in range(len(post)):
            idx = np.random.permutation(range(1,5))[0]
            for i in range(5):
                if i == idx:
                    fout.write(' '.join(random_data["%d"%(i+1)][k])+"\n")
                else:
                    fout.write(' '.join(data["%d"%(i+1)][k])+"\n")
            fout.write("------\n")