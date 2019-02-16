import sys
def progess_print(info):
    sys.stdout.write('\r>>'+info)
    sys.stdout.flush()