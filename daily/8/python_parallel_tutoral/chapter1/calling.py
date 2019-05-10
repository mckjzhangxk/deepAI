import os
import sys

program='python3'
print('Process calling')

arguments=('called.py',)

os.execvp(program,(program,)+arguments)
print('goodby')
