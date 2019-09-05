import os

def abc(labelpath,basepath='examples/test'):
    with open(labelpath, 'r') as fs:
        while True:
            line = fs.readline().strip()
            if line == '': break

            name = line
            dirname,xx=name.split('/')
            if not os.path.exists(os.path.join(basepath,dirname)):
                os.mkdir(os.path.join(basepath,dirname))

            with open(os.path.join(basepath,name[:-3]+'txt'),'w') as ff:
                ff.write(name+'\n')
                line = fs.readline().strip()
                ff.write(line+'\n')
                count = int(line)

                for i in range(count):
                    row = fs.readline()
                    splits = row.split()[:4]
                    face = list(map(int, splits[0:4]))+[1.0]
                    ff.write(' '.join(map(str,face))+'\n')
                    print(' '.join(map(str,face)))
abc('examples/widerface/wider_face_val_bbx_gt.txt')