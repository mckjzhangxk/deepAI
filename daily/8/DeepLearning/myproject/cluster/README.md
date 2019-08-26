1.关闭以后服务
ps aux |grep python3，查看下所有python3 cluster.py的进程pid记录
然后删除kill -9 上面的pid
2.vi .bashrc
  A.删除掉 alias myai对应的一行
  B.退出ssh 重新进入
3.pip install myai-1.1-py3-none-any.whl
4.启动myai --start &

其他命令:
myai --start & //注意 + &表示后台执行
myai --show //查询是否启动
myai --stop //查询关闭
