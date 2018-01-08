import sys
import os

from python.gitchrome.crohmelib.bin.inkml import *


def generate_seg(ink, nb_strk_max=4):
    nb_strokes = len(ink.strokes)
    print("Nb strokes: ", nb_strokes)
    strokes_list = range(nb_strokes)
    all_hyp_matrix = []
    if nb_strk_max > nb_strokes:
        nb_strk_max = nb_strokes
    for itNbMaxOfStrkPerObj in range(nb_strk_max):
        itNbMaxOfStrkPerObj += 1
        # add all possible segments
        # all_hyp_matrix.extend(itertools.combinations(strokes_list,itNbMaxOfStrkPerObj))
        # or add only seg without time jump :
        for i in strokes_list:
            if i + itNbMaxOfStrkPerObj < nb_strokes:
                r = range(i, i + itNbMaxOfStrkPerObj)
                # get real id of the strokes (strings)
                seg = []
                for s in r:
                    seg.append(ink.strkOrder[s])
                all_hyp_matrix.append(seg)
    print(all_hyp_matrix)

    # écrire dans fichier LG qui a le même nom que le fichier .ikml
    file_path = '/'.join(ink.fileName.split('/')[:-1])
    filename = ink.fileName.split('/')[-1].split('.')[0]
    with open(file_path + '/' + filename + '.LG', 'w') as file:
        hyp_id = 0
        for hypo in all_hyp_matrix:
            hyp_id += 1
            file.write("O, hyp{}, *, 1.0, {}\n".format(hyp_id, ', '.join(hypo)))
    return len(all_hyp_matrix)


def main():
    if len(sys.argv) < 1:
        print("Usage: [[python]] segmenter.py <file.inkml>")
        print("Usage: [[python]] segmenter.py <list.txt>")
        print("")
        print("Extract all hypothesis symbols from <file.inkml> or from the list of inkml <list.txt> "
              "and saved to files .LG")
        sys.exit(0)
    nb_hypo = -1
    file_list = []
    if ".inkml" in sys.argv[1]:
        file_list.append(sys.argv[1])
    else:
        fl = open(sys.argv[1])
        file_list = fl.readlines()
        fl.close()
    for fname in file_list:
        try:
            f = Inkml(fname.strip())
            nb_hypo = generate_seg(f)
        except IOError:
            print("Can not open " + fname.strip())
        except ET.ParseError:
            print("Inkml Parse error " + fname.strip())

    print(str(nb_hypo) + " hypothesis extracted.")
