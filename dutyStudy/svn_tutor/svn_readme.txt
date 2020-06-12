video:
	https://www.youtube.com/watch?v=DzTcrmYNRYw
安装svn
sudo apt install subversion openssh-server

服务器配置

1.svnadmin create repos //创建一个仓库,repos必须是不存在的目录
2.vi repos/conf/svnserve.conf //修改以下3行
	
	anon-access = none
	auth-access = write
	password-db = passwd
3.vi repos/conf/passwd
	加入认证
	zxk=123
4.启动服务器
	svnserver -d -r repos //表示repos是主目录,后台启动运行


客户端1
1.导入默认的项目
svn import client svn://192.168.0.1/repos //把client目录的东西导入到repos

客户端2
svn checkout svn://192.168.0.1/repos client1 //把repos checkout到本地client1 文件夹
#常用命令
svn update
svn add
svn revert
svn commit
svn rm

高级应用
把repos目录划分成
trunk //表示主干,当前版本
branches //表示分支,各种开发版本
tags  //表示release  版本

把repos的trunk的内容copy 到branches中
svn copy svn://127.0.0.1/repos/trunk svn://127.0.0.1/repos/branches/b1
客户端更新程序
svn update
然后你可能修改branches/b1中的模块,测试后与trunck合并


