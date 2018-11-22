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
			('zcfg_xh', { 'type':'string_text'} ),
			('zcfg_bt', { 'type':'string_text'} ),
			('zcfg_zw', { 'type':'string_text'} ),
			('zcfg_wh', { 'type':'string_text'} ), 
			('zcfg_fwrq', { 'type':'string_text'} ),
			('zcfg_fwdw', { 'type':'string_text'} ),
		]

	def GetFieldOrder(self): #字段的优先顺序
		return [('zcfg_bt','zcfg_zw','zcfg_wh','zcfg_fwrq','zcfg_fwdw')]

	def Connected(self):  #如果是数据库，则在此处做数据库连接 
		try:
			self.m_dbconn = cx_Oracle.connect("smartagent", "sdsx621", "124.128.231.182") 
		except cx_Oracle.Error, e:
			#logginf.error( "connect Error %d: %s" , e.args[0], e.args[1])
			print "conn oracle err."
			return False
		return True
		
	def OnBeforeIndex(self):
		try:
			self.m_cursor = self.m_dbconn.cursor()
			sql="""select id,index_name,update_time from index_counter where id=1"""
			self._couCount = self.m_cursor.execute(sql)
			self.data = [ row for row in self.m_cursor] 
			if(len(self.data)==0):
				sql="""insert into index_counter (id,index_name) values(1,'fgk')"""
				self.m_cursor.execute (sql)
				self.m_dbconn.commit()
			else:
				sql="""update index_counter set update_time=sysdate where id=1"""
				self.m_cursor.execute (sql)
				self.m_dbconn.commit()
			return True
		except cx_Oracle.Error, e:
			#logging.error( "NextDocument Error %d:%s",e.args[0],e.args[1])
			print "nextdocment err."
			return False
	
	def NextDocument(self, err):    #取得每一个文档记录的调用
		if self._rowCount==0:   #do fetch 
			try:
				self.m_cursor = self.m_dbconn.cursor() 
				sql="""SELECT ID,TO_CHAR(B.ZCFG_XH) AS ZCFG_XH,B.ZCFG_BT,B.ZCFG_ZW,B.ZCFG_WH,TO_CHAR(B.ZCFG_FWRQ, 'YYYY-MM-DD') AS ZCFG_FWRQ,B.FWDW_DM AS ZCFG_FWDW,TO_CHAR(B.ZCFG_DJRQ, 'YYYY-MM-DD') AS ZCFG_DJRQ FROM SSFG_MX B WHERE B.ZCFG_ZW IS NOT NULL"""
				self._rowCount = self.m_cursor.execute (sql)
				return self._getRow()
			except cx_Oracle.Error, e:
				#logging.error( "NextDocument Error %d:%s",e.args[0],e.args[1])
				print "nextdocment err."
				return False
		else:
			#self._rowCount-=1
			return self._getRow()

	def _getRow(self):
		m_row=self.m_cursor.fetchone()
		if m_row: 
			self.id = m_row[0]
			self.zcfg_xh = m_row[1]
			self.zcfg_bt = m_row[2]
			self.zcfg_zw = m_row[3].read()
			self.zcfg_wh = m_row[4]
			self.zcfg_fwrq = m_row[5]
			self.zcfg_fwdw = m_row[6]
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

	while source.NextDocument(None):
		print "===id=%d, zcfg_xh=%s, zcfg_bt=%s, zcfg_zw=%s, zcfg_wh=%s, zcfg_fwrq=%s,zcfg_fwdw=%s" % (source.id, source.zcfg_xh, source.zcfg_bt, source.zcfg_zw, source.zcfg_wh, source.zcfg_fwrq,source.zcfg_fwdw)
	pass

#################################################################################################	
class DeltaSource(object):
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
			('zcfg_xh', { 'type':'string_text'} ),
			('zcfg_bt', { 'type':'string_text'} ),
			('zcfg_zw', { 'type':'string_text'} ),
			('zcfg_wh', { 'type':'string_text'} ),
			('zcfg_fwrq', { 'type':'string_text'} ),
			('zcfg_fwdw', { 'type':'string_text'} ),
		]

	def GetFieldOrder(self): #字段的优先顺序
		return [('zcfg_bt','zcfg_zw','zcfg_wh','zcfg_fwrq','zcfg_fwdw')]

	def Connected(self): #如果是数据库，则在此处做数据库连接 
		try: 
			self.m_dbconn = cx_Oracle.connect("smartagent", "sdsx621", "124.128.231.182") 
		except cx_Oracle.Error, e:
			#logginf.error( "connect Error %d: %s" , e.args[0], e.args[1])
			print "conn oracle err."
			return False
		return True
	
	def NextDocument(self, err):    #取得每一个文档记录的调用
		if self._rowCount==0:   #do fetch
			
			try:
				self.m_cursor = self.m_dbconn.cursor() 
				sql="""SELECT ID,TO_CHAR(ZCFG_XH) AS ZCFG_XH,ZCFG_BT,ZCFG_ZW,ZCFG_WH,TO_CHAR(ZCFG_FWRQ,'YYYY-MM-DD') AS ZCFG_FWRQ,FWDW_DM AS ZCFG_FWDW,TO_CHAR(ZCFG_DJRQ,'YYYY-MM-DD') AS ZCFG_DJRQ FROM SSFG_MX WHERE ZCFG_DJRQ>(SELECT UPDATE_TIME FROM INDEX_COUNTER WHERE ID=1)"""
				self._rowCount = self.m_cursor.execute (sql)
				return self._getRow()
			except cx_Oracle.Error, e:
				#logging.error( "NextDocument Error %d:%s",e.args[0],e.args[1])
				print "nextdocment err."
				return False
		else:
			#self._rowCount-=1
			return self._getRow()

	def _getRow(self):
		m_row=self.m_cursor.fetchone()
		if m_row: 
			self.id = m_row[0]
			self.zcfg_xh = m_row[1]
			self.zcfg_bt = m_row[2]
			self.zcfg_zw = m_row[3].read()
			self.zcfg_wh = m_row[4]
			self.zcfg_fwrq = m_row[5]
			self.zcfg_fwdw = m_row[6]
			return True
		else:
			return False

	def OnAfterIndex(self):
		try:
			self.m_cursor = self.m_dbconn.cursor()
			sql="""update index_counter set update_time=sysdate where id=1"""
			self.m_cursor.execute (sql)
			self.m_dbconn.commit()
			return True
		except cx_Oracle.Error, e:
			#logging.error( "NextDocument Error %d:%s",e.args[0],e.args[1])
			print "nextdocment err."
			return False
		self.m_dbconn.close()
		#self.m_cursor.close()
if __name__ == "__main__":    #直接访问演示部分
	conf = {}
	source = MainSource(conf)
	source.Connected()

	while source.NextDocument(None):
		print "#####id=%d,zcfg_xh=%s, zcfg_bt=%s, zcfg_zw=%s, zcfg_wh=%s, zcfg_fwrq=%s,zcfg_fwdw=%s" % (source.id, source.zcfg_xh, source.zcfg_bt, source.zcfg_zw, source.zcfg_wh, source.zcfg_fwrq, source.zcfg_fwdw)
	pass

