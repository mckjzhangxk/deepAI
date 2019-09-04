对.data文件的说明
由于程序全部是是使用darknet,下面路径都是相对于darknet的目录而言，
    例如要查看配置文件到{darknet_root}/winderface/train.txt
    要保存weight保存到{darknet_root}winderface/backup/

classes=1
train=winderface/train.txt
valid=winderface/test.txt
names=winderface/winderface.names
backup=winderface/backup/