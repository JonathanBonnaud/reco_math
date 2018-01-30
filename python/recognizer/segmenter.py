import sys
import os

from python.gitchrome.crohmelib.bin.inkml import *


def generate_seg(ink, nb_strk_max=2):
    nb_strokes = len(ink.strokes)
    print("Nb strokes: ", nb_strokes)
    strokes_list = ink.strokes
    all_hyp_matrix = []
    if nb_strk_max > nb_strokes:
        nb_strk_max = nb_strokes
    for itNbMaxOfStrkPerObj in range(nb_strk_max):
        itNbMaxOfStrkPerObj += 1
        # add all possible segments
        # all_hyp_matrix.extend(itertools.combinations(strokes_list,itNbMaxOfStrkPerObj))
        # or add only seg without time jump :
        for i in strokes_list:
            stroke_id = int(i)
            if stroke_id + itNbMaxOfStrkPerObj <= nb_strokes:
                r = range(stroke_id, stroke_id + itNbMaxOfStrkPerObj)
                # get real id of the strokes (strings)
                seg = []
                for s in r:
                    seg.append(ink.strkOrder[s])
                all_hyp_matrix.append(seg)
    # print(all_hyp_matrix)

    # écrire dans fichier LG qui a le même nom que le fichier .ikml
    output_folder = "LGs/segments_hypo"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filename = ink.fileName.split('/')[-1].split('.')[0]
    hyp_id = 0
    for hypo in all_hyp_matrix:
        with open(output_folder + '/' + filename + '_hypo_' + str(hyp_id) + '.inkml', 'w') as file:
            hyp_id += 1
            file.write(
                "<ink xmlns=\"http://www.w3.org/2003/InkML\">\n<traceFormat>\n<channel name=\"X\" type=\"decimal\"/>\n"
                "<channel name=\"Y\" type=\"decimal\"/>\n</traceFormat>\n")
            for stroke_id in hypo:
                file.write("<trace id=\"{}\">\n{}\n</trace>\n".format(stroke_id, strokes_list[stroke_id]))
            file.write("</ink>")

    with open(output_folder + '/' + filename + '.lg', 'w') as file:
        hyp_id = 0
        for hypo in all_hyp_matrix:
            file.write("O, hyp{}, *, 1.0, {}\n".format(hyp_id, ', '.join(hypo)))
            hyp_id += 1
    return len(all_hyp_matrix)


def main(args):
    if len(args) < 1:
        print("Usage: [[python]] segmenter.py <file.inkml>")
        print("Usage: [[python]] segmenter.py <list.txt>")
        print("")
        print("Extract all hypothesis symbols from <file.inkml> or from the list of inkml <list.txt> "
              "and saved to files .LG")
        sys.exit(0)
    nb_hypo = -1
    file_list = []
    if ".inkml" in args[0]:
        file_list.append(args[0])
    else:
        fl = open(args[0])
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
