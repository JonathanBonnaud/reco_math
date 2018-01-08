################################################################
# segGenerator.py
#
# Program that reads in inkml ground-truthed files and generate 
# right or wrong segmented symbols.
#
#
# Author: H. Mouchere, Feb. 2014
# Copyright (c) 2014, Harold Mouchere
################################################################
import random
import sys

from python.gitchrome.crohmelib.bin.inkml import *


def generate_right_seg(ink, seg_name, k=0):
    """
    generate all one inkml file per symbol. Return the number of generated files.
    :param ink:
    :param seg_name:
    :param k:
    :return:
    """
    # print "size seg ="+str(len(ink.segments.values()))
    output_g_tfile = open(seg_name + "_GT.txt", 'a')
    for seg in ink.segments.values():
        symb = Inkml()
        symb.UI = ink.UI + "_" + str(k)
        # symb.truth = seg.label
        lab = seg.label
        if lab == ",":
            lab = "COMMA"
        output_g_tfile.write(symb.UI + "," + lab + "\n")
        for s in seg.strId:
            symb.strokes[s] = ink.strokes[s]
        symb.segments["0"] = seg
        symb.segments["0"].id = "0"
        symb.segments["0"].label = ""
        symb.getInkML(seg_name + str(k) + ".inkml")
        k += 1
    output_g_tfile.close()
    return k


def generate_wrong_seg(ink, seg_name, nb=-1, nb_strk_max=4, k=0):
    """
    generate nb wrong segmentation from the ink. If nb=-1, it will generate all wrong seg.
    Hypothesis are generated with continuous index in the ink file (no time jump)
    :param ink:
    :param seg_name:
    :param nb:
    :param nb_strk_max: maximum size of the generated hypothesis
    :param k:
    :return:
    """
    nbs = len(ink.strokes)
    strokes_list = range(nbs)
    all_hyp_matrix = []
    if nb_strk_max > nbs:
        nb_strk_max = nbs
    for itNbMaxOfStrkPerObj in range(nb_strk_max):
        itNbMaxOfStrkPerObj += 1
        # add all possible segments
        # all_hyp_matrix.extend(itertools.combinations(strokes_list,itNbMaxOfStrkPerObj))
        # or add only seg without time jump
        for i in strokes_list:
            if i + itNbMaxOfStrkPerObj < nbs:
                r = range(i, i + itNbMaxOfStrkPerObj)
                # get real id of the strokes (strings)
                seg = []
                for s in r:
                    seg.append(ink.strkOrder[s])
                # check if it is not a symbol
                # print str(seg)
                if not ink.isRightSeg(set(seg)):
                    # print "JUNK"
                    all_hyp_matrix.append(seg)
    if -1 < nb < len(all_hyp_matrix):
        all_hyp_matrix = random.sample(all_hyp_matrix, nb)
    symb = Inkml()
    # symb.truth = "junk"
    output_g_tfile = open(seg_name + "_GT.txt", 'a')
    for hyp in all_hyp_matrix:
        symb.UI = ink.UI + "_" + str(k)
        symb.strokes = {}
        for s in hyp:
            symb.strokes[s] = ink.strokes[s]
        symb.segments["0"] = Segment("0", "", hyp)
        symb.getInkML(seg_name + str(k) + ".inkml")
        output_g_tfile.write(symb.UI + ", junk\n")
        k += 1
    output_g_tfile.close()
    return k


def main():
    if len(sys.argv) < 3:
        print("Usage: [[python]] segGenerator.py <file.inkml> symbol_file_name [JUNK|BOTH [NB]]")
        print("Usage: [[python]] segGenerator.py <list.txt> symbol_file_name [JUNK|BOTH [NB]]")
        print("")
        print("Extract all symbols from <file.inkml> or from the list of inkml <list.txt> and saved to files named:")
        print("symbol_file_name_0.inkml, symbol_file_name_1.inkml, symbol_file_name_2.inkml ...")
        print("if JUNK is set, wrong segmentations are generated (with time consecutive strokes)")
        print("if BOTH is set, both wrong segmentations and symbols are generated")
        print("	   NB is the number of junk generated per inkml file, randomly chosen (default = all) ")
        sys.exit(0)
    n = -1
    gen_symb = True
    gen_junk = False
    # print str(len(sys.argv))+ sys.argv[4]
    if "JUNK" in sys.argv:
        gen_symb = False
        gen_junk = True
        print("Extract only junks")
    elif "BOTH" in sys.argv:
        gen_symb = True
        gen_junk = True
        print("Extract symbols and junks")
    else:
        print("Extract only symbols")
    if gen_junk:
        try:
            n = int(sys.argv[-1])
            print("extract " + str(n) + " junks per expression")
        except ValueError:
            n = -1
            print("extract all of junk per expression\n")
    file_list = []
    if ".inkml" in sys.argv[1]:
        file_list.append(sys.argv[1])
    else:
        fl = open(sys.argv[1])
        file_list = fl.readlines()
        fl.close()
    nb = 0
    for fname in file_list:
        try:
            f = Inkml(fname.strip())
            if gen_symb:
                nb = generate_right_seg(f, sys.argv[2], k=nb)
            if gen_junk:
                nb = generate_wrong_seg(f, sys.argv[2], n, k=nb)
        except IOError:
            print("Can not open " + fname.strip())
        except ET.ParseError:
            print("Inkml Parse error " + fname.strip())

    print(str(nb) + " symbols or junks  extracted")


main()
