import os, signal
import argparse
import cfai.config as config
def check_process(pstring):
    query = "ps aux | grep " + pstring + " | grep -v grep"

    for line in os.popen(query):
        fields = line.split()
        pid = fields[1]
        target=fields[-1]
        if 'target' in target:
            target=target.replace('--target=','')
        else:
            target=config.defaultConfig()['target']
        return (pid, target)
    return None

def stopProcess(pstring):
    pidinfo=check_process(pstring)
    if pidinfo==None:
        print('服务没有开启')
    else:
        pid=pidinfo[0]
        os.kill(int(pid), signal.SIGKILL)
        print('服务pid(%s)关闭成功' % pid)


def startProcess(pstring,target):
    '''
    
    :param pstring: 进程名
    :param target: 进程需要的参数,输出的目录地址
    :return: 
    '''
    pidinfo=check_process(pstring)
    if pidinfo==None:
        # os.system('nohup python3 -m cfai.cluster &')
        program = 'nohup'
        arguments = ('python3','-m','cfai.cluster',)
        if target is not  None:
            arguments=arguments+('--target='+target,)
        print(arguments)
        os.execvp(program, (program,) + arguments)
        print('服务开启成功')
    else:
        pid=pidinfo[0]
        print('服务pid(%s)已经启动'%pid)
def showProcess(pstring):
    pidinfo = check_process(pstring)
    if pidinfo==None:

        print('服务没启动')
    else:
        pid=pidinfo[0]
        print('服务pid(%s)已经启动'%pid)
        print('配置路径:',pidinfo[1])

def parse():
    parser=argparse.ArgumentParser()

    parser.add_argument('--start',action='store_true')
    parser.add_argument('--stop',action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--target',default=None,type=str)

    args=parser.parse_args()
    return args
if __name__ == '__main__':
    processname="'python3 -m cfai.cluster'"

    p=parse()



    if p.start:
        startProcess(processname,p.target)
    elif p.stop:
        stopProcess(processname)
    elif p.show:
        showProcess(processname)
    else:
        print('nothing to do')

