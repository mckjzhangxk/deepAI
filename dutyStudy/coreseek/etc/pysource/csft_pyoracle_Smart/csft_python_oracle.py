#! /usr/bin/env python
# coding=utf-8
# coreseek4.1 从oracle数据库读取数据


import cx_Oracle
import logging
import os
import sys 

default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
        reload(sys)
        sys.setdefaultencoding(default_encoding)

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
reload(sys) # Python2.5 may remove sys.setdefaultencoding() method after init, so here reload it
sys.setdefaultencoding('utf-8')



#from misc import setLoggerHandler,loadConfig
#print  [x.lower() for x in sys.path]
class MainSource(object):
	def __init__(self,conf):
		self.conf = conf
		self._couCount=0
		self._rowCount=0
		self.m_cursor = None
		self.m_dbconn = None
                        
		os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
		reload(sys) # Python2.5 may remove sys.setdefaultencoding() method after init, so here reload it
		sys.setdefaultencoding('utf-8')
	def GetScheme(self):  #获取id,标题、正文内容
                
		return [
		
			('id' , {'docid':True, } ),	
			('content', { 'type':'string_text'} ),
			('title', { 'type':'string_text'} ),
			('code', { 'type':'string_text'} ), 
			('typeflag', { 'type':'string_text'} ), 
		]

	def GetFieldOrder(self): #字段的优先顺序
                
		return [('content','title','code')]

	def Connected(self):  #如果是数据库，则在此处做数据库连接 
		try:
			self.m_dbconn = cx_Oracle.connect("lxnsfw", "S4VMz2fQwh", "10.1.1.51:1521/platform")

		except cx_Oracle.Error, e:
			# logginf.error( "connect Error %d: %s" , e.args[0], e.args[1])
			# print "conn oracle err."
			print e
			return False
		return True
		 
		  
	def NextDocument(self):    #取得每一个文档记录的调用
		if self._rowCount==0:   #do fetch 
			try:
				self.m_cursor = self.m_dbconn.cursor() 
				sql="SELECT ROWNUM BROSE_NUM, Q.CONTENT, Q.CODE, Q.TITLE, Q.TYPEFLAG FROM (SELECT T.BUS_ITEM_CODE CODE, I.BUS_NAME TITLE, T.ADVERT CONTENT, '1' TYPEFLAG FROM SAC_BUSITEM_ADVERT T, SAM_CODE_BUS I WHERE I.BUS_CODE = T.BUS_ITEM_CODE UNION ALL SELECT T.NOTICE_UID CODE, T.TITLE TITLE, T.CONTENT_TEXT CONTENT, '2' TYPEFLAG FROM SAR_NOTICE T UNION ALL SELECT H.ITEM_UID CODE, H.TITLE TITLE, H.CONTENT CONTENT, H.QUES_TYPE_CODE TYPEFLAG FROM SAR_HOTQUE H UNION ALL SELECT Q.ITEM_UID CODE, Q.TITLE TITLE, Q.CONTENT_CLOB CONTENT, 'Q' TYPEFLAG FROM SAR_QUESTIONS Q UNION ALL SELECT A.PRO_UID CODE, (SELECT SQ.TITLE FROM SAR_QUESTIONS SQ WHERE SQ.ITEM_UID = A.PRO_UID) TITLE, A.CONTENT_CLOB CONTENT, 'Q' TYPEFLAG FROM SAR_ANWSERS A) Q"
				self._rowCount = self.m_cursor.execute (sql)
				return self._getRow()
			except cx_Oracle.Error, e:
				#logging.error( "NextDocument Error %d:%s",e.args[0],e.args[1])
				print "nextdocment errsss."
				return False
		else:
			#self._rowCount-=1
			return self._getRow()

	def _getRow(self):
		m_row=self.m_cursor.fetchone()
		if m_row:
			self.id=m_row[0]
			self.content = m_row[1].read()
			self.code = m_row[2]
			self.title = m_row[3]
			self.typeflag = m_row[4]
			return True
		else:
			return False

	def OnAfterIndex(self):
		self.m_dbconn.close()
		#self.m_cursor.close()
if __name__ == "__main__":    #直接访问演示部分
	conf = {}
	source = MainSource(conf)
	source.Connected()

	while source.NextDocument():
		print 'ok'
		print "===brose_num=%d, content=%s, title=%s,code=%s,typeflag=%s" % (source.id, source.content, source.title, source.code, source.typeflag)
	pass


