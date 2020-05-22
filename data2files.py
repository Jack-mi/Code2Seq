from argparse import ArgumentParser

def read_args():
    parser = ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-d", "--data", dest="data_path",
                       help="path to preprocessed dataset")
    return parser.parse_args()

if __name__ == '__main__':
    # arg = read_args()
    f = open('data/java-small/java-small.train.c2s', 'r')
    cnt = 0
    data_dir = 'data/java-small/train'
    for line in f:
        content = f.readline()
        if content == '' or content == '\n':
            continue
        fnew = open(data_dir + '/{:0>6d}.txt'.format(cnt), 'w')
        fnew.write(content)
        cnt += 1
        print("Now :{}".format(cnt))
        # if cnt == 5:
        #     break
