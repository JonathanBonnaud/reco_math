def write(file_name):
    with open(file_name, 'r') as file:
        r = file.readlines()
    _dict = dict()
    for line in r:
        res = line.strip('\n').split('\t')
        if res[0].split('.')[0].split('_hypo_')[0] in _dict:
            _dict[res[0].split('.')[0].split('_hypo_')[0]].append(('hyp'+res[0].split('.')[0].split('_hypo_')[1], res[1]))
        else:
            _dict[res[0].split('.')[0].split('_hypo_')[0]] = [('hyp'+res[0].split('.')[0].split('_hypo_')[1], res[1])]
    print(_dict['18_em_0'])

    for filename, hypos in _dict.items():
        with open('LGs/segments_hypo/' + filename + '.lg', 'r+') as file:
            lines = file.readlines()
            file.seek(0)
            file.truncate()
            for line in lines:
                for hypo in hypos:
                    if hypo[0] in line:
                        line = line.replace('*', hypo[1])
                file.write(line)
        print("File {}.lg updated.".format(filename))


