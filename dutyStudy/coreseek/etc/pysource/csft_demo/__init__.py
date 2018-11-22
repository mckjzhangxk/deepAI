# -*- coding:utf-8 -*-
# coreseek3.2 python source演示
# author: HonestQiao
# date: 2010-06-03 11:46

class MainSource(object):
    def __init__(self, conf):
        self.conf =  conf
        self.idx = 0
        self.data = [
            {'id':1, 'subject':u"愚人节最佳蛊惑爆料 谷歌300亿美元收购百度", 'context':u'据国外媒体报道，谷歌将巨资收购百度，涉及金额高达300亿美元。谷歌借此重返大陆市场。　　该报道称，目前谷歌与百度已经达成了收购协议，将择机对外公布。百度的管理层将100%保留，但会将项目缩减，包括有啊商城，以及目前实施不力的凤巢计划。正在进行测试阶段的视频网站qiyi.com将输入更多的Youtube资源。(YouTube在大陆区因内容审查暂不能访问)。　　该消息似乎得到了谷歌CEO施密特的确认，在其twitter上用简短而暧昧的文字进行了表述：“ Withdraw from that market? u\'ll also see another result, just wait... ” 意思是：从那个市场退出?你还会看到另外一个结果。毫无疑问，那个市场指的就是中国大陆。而另外的结果，对应此媒体报道，就是收购百度，从而曲线返回大陆搜索市场。　　在最近刚刚结束的深圳IT领袖峰会上，李彦宏曾言，“谷歌没有退出中国，因为还在香港”。也似乎在验证被收购的这一事实。　　截止发稿，百度的股价为597美元，市值为207亿美元。谷歌以高达300亿美元的价格，实际溢价高达50%。而谷歌市值高达1796亿美元，而且手握大量现金，作这样的决策也在情理之中。    近日，很多媒体都在报道百度创始人、CEO李彦宏的两次拒购：一次是百度上市前夕，李彦宏拒绝谷歌的并购，这个细节在2月28日央视虎年首期对话节目中得到首次披露﹔一次是在百度国际化战略中，拒绝采用海外并购的方式，而是采取了从日本市场开始的海外自主发展之路。这也让笔者由此开始思考民族品牌的发展之路。 　　收购是打压中国品牌的惯用伎俩　　2010年2月28日，央视经济频道《对话》节目昨晚推出虎年首期节目，百度董事长兼CEO李彦宏作为嘉宾做客节目。李彦宏首度谈及2005年百度上市前夕，谷歌CEO施密特曾秘密造访百度时秘密谈话的内容，主要是劝阻百度上市，李彦宏断然拒绝了施密特的“好意”。今天看来，施密特当日也许已有不祥的预感，这个几百人的小公司终有一日会成为他们的大麻烦。　　本期《对话》一经播出，便引发了业界讨论。　　外资品牌通过收购打压中国品牌的案例不胜枚举。从以往跨国企业并购的中国品牌来看，真正让其活下来的品牌并不多，要么被雪藏，要么被低端化。　　因此，2005年百度没有接受Google的收购邀请，坚持自主发展，这对于保护中国品牌，维护中国网民信息安全有着至关重要的作用。当前百度市场份额高达76%，并持续增长，这也充分验证了李彦宏拒绝收购决策的正确性。　　今天看来，“百度一下”已经成为3亿多中国网民的网络生存法则，而直到今天环视全球，真正能像中国一样，拥有自己独立搜索引擎的只有4个国家！我们也许应该庆幸当时李彦宏的选择。这个故事也告诉我们，中国企业做品牌还要靠自己！　　收购也可能是中国企业走出去的陷阱　　同样在2月28日，亚布力第十届年会上，李彦宏在论坛上指出：“我们和很多其它公司的国际化路子是很不一样的，我们不是去买一个国外的公司，”，李彦宏解释了百度率先选择日本作为走出去的对象的原因，因为日本和中国一衣带水的近邻优势，日本的市场规模，在日本也没有一家独大的搜索引擎。　　中国企业收购这些外资品牌目的是“借船出海”。外资品牌进入中国是收购中国优质品牌，而中国企业进入国外市场的收购策略恰恰相反，这也是中国企业借船出海屡屡失败的原因所在。　　笔者认为，中国互联网公司走出去要依靠自身发展，并不能单纯依靠收购。李彦宏在百度成立伊始就抱定了国际化决心，使百度真正在面对国际化机遇时，更加冷静和具有前瞻力。李彦宏也承认当前百度在日本还处于初级发展阶段，但他也预言“2012年，百度与Google划洋而治”，对此我们拭目以待！', 'published':1270131607, 'author_id':1},
            {'id':2, 'subject':u'Twitter主页改版 推普通用户消息增加趋势话题', 'context':u'4月1日消息，据国外媒体报道，Twitter本周二推出新版主页，目的很简单：帮助新用户了解Twitter和增加用户黏稠度。　　新版Twittter入口处的内容眼花缭乱，在头部下方有滚动的热门趋势话题，左边列出了普通用户账户和他们最新的消息。　　另一个显著的部分是“Top Tweets”，它采用了新算法推选出最热门的话题，每个几秒刷新一次。Twitter首席科学家Abdur Chowdhury表示，这种算法选出了所有用户的信息，而不是拥有大量追随者所发的信息。　　首页对于首次访问网站的用户非常重要，因为这决定了用户的第一印象。研究发现，多达60%的Twittter用户在注册后的一个月内不再访问网站。Twittter希望能更好地展现网站的面貌，帮助游客找到感兴趣的东西', 'published':1270135548, 'author_id':1},
            {'id':3, 'subject':u'死都要上！Opera Mini 体验版抢先试用', 'context':u'Opera一直都被认为是浏览速度飞快，同时在移动平台上更是占有不少的份额。不久前，Opera正式向苹果提交了针对iPhone设计的Opera Mini。日前，台湾IT网站放出了Opera Mini和Safari的评测文章，下面让我们看看Opera和Safari到底谁更好用更快吧。　　Opera Mini VS Safari，显示方式很不相同和Safari不同的是，Opera Mini会针对手机对网页进行一些调整　　Opera Mini与Safari的运作原理不大相同。网页会通过Opera的服务器完整压缩后再发送到手机上，不像Safari可通过Multi-Touch和点击的方式自由缩放，Opera Mini会预先将文字照iPhone的宽度做好调整，点击区域后自动放大。如果习惯了Safari的浏览方式，会感觉不大顺手，不过对许多宽度太宽，缩放后文字仍然显示很小的网页来说，Opera Mini的显示方式比较有优势。　　打开测试网站首页所花费的流量，Safari和Opera Mini的差距明显可见。这个在国内移动资费超高的局面来说，Opera Mini估计会比较受欢迎和省钱。Opera Mini的流量少得惊人，仅是Safari的十分之一　　兼容性相比，Safari完胜打开Google首页，Safari上是iPhone专用界面，Opera则是一般移动版本　　Opera Mini的速度和省流量还是无法取代Safari成为iPhone上的主要浏览器。毕竟iPhone的高占有率让许多网站，线上服务都为Safari设计了专用页面。光Google的首页为例子就看出了明显的差别。另外，像Google Buzz这样线上应用，就会出现显示错误。Google Buzz上，Opera无法输入内容　　Opera Mini其他专属功能页面内搜索和关键字直接搜索相当人性化　　除了Opera独创的Speed Dial九宫格快速启动页面外，和Opera Link和电脑上的Opera直接同步书签、Speed Dial设定外。Opera Mini还能够直接搜索页面中的文字，查找资料时相当方便。另外也能选取文字另开新分页搜索，比起Safari还要复制、开新页、粘贴简单许多。同时还能将整个页面打包存储，方便离线浏览。　　现在Opera Mini想要打败Safari还剩下一个很严重的问题-苹果何时会或者会不会通过Opera Mini的审核。', 'published':1270094460, 'author_id':2},
        ]

    def GetScheme(self):  #获取结构，docid、文本、整数
        return [
            ('id' , {'docid':True, } ),
            ('subject', { 'type':'text'} ),
            ('context', { 'type':'text'} ),
            ('published', {'type':'integer'} ),
            ('author_id', {'type':'integer'} ),
        ]

    def GetFieldOrder(self): #字段的优先顺序
        return [('subject', 'context')]
        
    def Connected(self):   #如果是数据库，则在此处做数据库连接
	if self.conn==None:       
            self.conn = connect(host='localhost', user='sddslznk', password='sddslznk', database='sdlznk40', as_dict=True,charset='cp936')
            self.cur = self.conn.cursor()
            sql = 'SELECT id,title FROM documents'
            self.cur.execute(sql)
            self.data = [ row for row in self.cur]
        pass
 
    def NextDocument(self, err):   #取得每一个文档记录的调用
        if self.idx < len(self.data):
            item = self.data[self.idx]
            self.docid = self.id = item['id'] #'docid':True
            self.subject = item['subject'].encode('utf-8')
            self.context = item['context'].encode('utf-8')
            self.published = item['published']
            self.author_id = item['author_id']
            self.idx += 1
            return True
        else:
            return False

if __name__ == "__main__":    #直接访问演示部分
    conf = {}
    source = MainSource(conf)
    source.Connected()

    while source.NextDocument():
        print "id=%d, subject=%s" % (source.docid, source.subject)
    pass
#eof
