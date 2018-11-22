# -*- coding: utf-8 -*-

"""

"""

import jieba.posseg as pseg
import re


def preprocess_text(value):
    """
        文本预处理操作

        :param
            - value: 原始文本

    """

    # 位置信息和受害人
    words_lst = pseg.cut(value)
    location_count = 0
    name_count = 0
    name = ""
    location_infos = ""
    temp_name = re.search("(报警人|报案人|被害人).{2,3}",value)
    if temp_name != None:
        name_count = 1
        name = temp_name.group(0)
        name_infos = pseg.cut(name)
        temp_name_info = ""
        for name_info, name_flag in name_infos:
            if name_flag == "p" or name_flag == "v" or name_flag == "uj"  or name_flag == "c":
                temp_name_info = name_info
        if temp_name_info != "" and name.endswith(temp_name_info):
            name = name.split(temp_name_info)[0]
        name = re.sub("报警人|报案人|被害人", "", name)

    last_flag = ""
    for word, flag in words_lst:
        if word == "张" and (last_flag == "x" or last_flag == ""):
            name = word
        if flag.startswith('nr') and len(word) >= 2 and name_count == 0:
            name = name + word
            name_count = 1
        if flag == 'ns' and last_flag == "p" and location_count == 0:
            location_infos = location_infos + word
            location_count = 1
        if location_count > 0 and name_count > 0:
            break
        last_flag = flag


    if location_infos == "":
        location = "无位置信息"
    else:
        # test = re.search(location_infos + '.*[\\,，.。]$', value).group(0)
        # print(test)
         tmp_location = re.search(location_infos + '.*[\\,，.。]$', value)
         if tmp_location == None:
             location = "无位置信息"
         else:
             location = re.split('[,，.。]', tmp_location.group(0))[0]
             locations_lst = pseg.cut(location)
             for word, flag in locations_lst:
                 if flag == "v" and word != "示范":
                     location = re.split(word, location)[0]
                     if location.find("公司") > 0 :
                         if location.endswith("公司"):
                             break
                         else:
                             location = location.split("公司")[0]+"公司"
                     else:
                         break
    # 被骗金额
    # 返款及无累加金额暂未计算

    total_info = re.search(r'(共|损失).*[0-9]+(.[0-9]{1,2})?元', value)
    # if total_info == None:
    #     total_info = re.search(r'.*[0-9]+(.[0-9]{1,2})?元', value)
    #     if total_info != None:
    #         total_info = total_info.group(0)
    # else:
    #     total_info = total_info.group(0)
    amount = 0
    if total_info == None:
        total_com = re.compile(r'[0-9]+(.[0-9]{1,2})?元')
        infos = re.findall(total_com, value)
        if infos != None:
            for info in infos:
                if info != "":
                    info_amount = info.strip("元")
                    print(info_amount)
                    amount = amount + int(info_amount)
    else:
        amount = re.search(r'[0-9]+(.[0-9]{1,2})?元', total_info.group(0)).group(0)

    # 时间

    time = re.search(
        r'(?P<Year>[0-9零一二三四五六七八九]{4,4}|同|当)年(?P<Imprecise_month>[末底终初中])?(?:(?P<Season>[春夏秋冬])[天季])?(?P<Month>[0-9零一二三四五六七八九]{1,2}|同|当)月份?(?P<Mo>左右)?(?P<Imprecise_day>(?P<Id1>上旬|中旬|下旬|[末底终初中])?(?P<Id2>的一天|一天)?)?(?:(?P<Day>(?:[0-9零一二三四五六七八九]{1,2}|同|当|某))[日号](?P<D1>左右|前后)?(?P<D2>的?一天)?)?(?P<Imprecise_hour>凌晨|早晨|早上|晚上|傍晚|上午|中午|下午|深夜|半夜|夜间|夜晚|夜里|夜|早|中|晚)?(?:(?P<Hour>[0-9零一二三四五六七八九]{1,2}|某)(?:时|点钟?)(?P<H>左右|前后|许)?)?(?:(?P<Minute>[0-9零一二三四五六七八九]{1,2}|某)分(?P<Mi>左右|前后|许)?)?(?:(?P<Second>[0-9零一二三四五六七八九]{1,2}|某)秒(?P<S>左右|前后|许)?)?\b',
        value)
    if time == None:
        time = re.search('\d{4}年\d{1,2}月\d{1,2}日\d{1,2}时\d{1,2}分',value)
    if time == None:
        time = "无时间信息"
    else:
        time = time.group(0)

    # check_time = re.search(r'\w*年\w*月\w*至\w*[年月日]\w*[月日]?\w*日?', value)
    # if (check_time == None):
    #     # 带中文数字版
    #     time = re.search(
    #         r'(?P<Year>[0-9零一二三四五六七八九]{4,4}|同|当)年(?P<Imprecise_month>[末底终初中])?(?:(?P<Season>[春夏秋冬])[天季])?(?P<Month>[0-9零一二三四五六七八九]{1,2}|同|当)月份?(?P<Mo>左右)?(?P<Imprecise_day>(?P<Id1>上旬|中旬|下旬|[末底终初中])?(?P<Id2>的一天|一天)?)?(?:(?P<Day>(?:[0-9零一二三四五六七八九]{1,2}|同|当|某))[日号](?P<D1>左右|前后)?(?P<D2>的?一天)?)?(?P<Imprecise_hour>凌晨|早晨|早上|晚上|傍晚|上午|中午|下午|深夜|半夜|夜间|夜晚|夜里|夜|早|中|晚)?(?:(?P<Hour>[0-9零一二三四五六七八九]{1,2}|某)(?:时|点钟?)(?P<H>左右|前后|许)?)?(?:(?P<Minute>[0-9零一二三四五六七八九]{1,2}|某)分(?P<Mi>左右|前后|许)?)?(?:(?P<Second>[0-9零一二三四五六七八九]{1,2}|某)秒(?P<S>左右|前后|许)?)?\b',
    #         value).group(0)
    # else:
    #     time = check_time.group(0)


    # 报案类型
    type = re.search(".*骗|转账", value)
    if type != None:
        type = '诈骗'
    else:
        type = '其他'


    # 诈骗信息
    fraud_info = ""
    weixins = re.findall("[a-zA-Z][a-zA-Z0-9_-]{5,19}",value)
    for weixin in weixins:
        fraud_info = fraud_info + "微信号:" + weixin + "\n\t\t"
    numbers = re.findall(r'\d+',value)
    for number in numbers:
        if len(number) == 19:
            fraud_info = fraud_info+"银行卡号:"+ number +"\n\t\t"
        elif len(number) >= 9 and len(number) <= 10:
            fraud_info = fraud_info + "QQ号:" + number + "\n\t\t"
        elif len(number) == 11 and re.match("1\d{10}",number):
            fraud_info = fraud_info + "手机号:" + number + "\n\t\t"

    print("*****************************\n")
    print("报案人:" + name)
    print("\n报案地点:" + location)
    print("\n报案时间:" + time)
    print("\n报案金额:" + str(amount))
    print("\n报案类型:" + type)
    print("\n涉案账号:" + fraud_info)


if __name__ == '__main__':
    # 标准信息
    value = "2018年7月3日14时，报警人顾香的QQ（562543875，昵称：明明很爱你）上收到一条消息（858087058，昵称：梁静依）称：招聘刷单员，佣金5%-8%。于是顾香添加对方QQ开始聊天，并自2018年7月3日16时41分至7月4日12时05分期间，分别以点击链接付款、扫二维码付款、向对方银行卡转账（顾香农业银行卡：6228480393713373610，对方农业银行卡：6230520210024496674，开户人：霍涛涛）的方式，先后14次向对方付款共计34621.4元。后报警人发现被骗，遂报案。损失人民币共计34621.4元。"
    # 无报警人信息
    value1 = "2018年8月24日10时许，报警人郭桥在南京市经济技术开发区新尧路28号南京人示范公寓2幢502室玩手机时，其手机陌陌（248011813，昵称：小情绪理不清）上收到（638571458，昵称：确认过眼神）的消息问其是否刷单，该郭称可以试一试，随后对方给其刷单客服QQ号码（2969911689，昵称：吴娜娜11），该郭加对方好友后，通过QQ（605573485，昵称：不离不弃）与对方商谈刷单事宜，对方先后10次让其通过支付宝（绑定的中国银行卡，卡号：621790610000760034，开户人：郭桥）、QQ（绑定的中国银行卡，卡号：621790610000760034，开户人：郭桥）、云闪付（绑定的中国银行卡，卡号：621790610000760034，开户人：郭桥）向对方转账人民币14376.42元。于2018年8月24日23时50左右发现被骗，遂报案。损失人民币14376元。"
    # 多时间西信息
    value2 = "报案人柏丽君称：其2018年5月初在南京经济技术开发区恒通大道1号南京熊猫科技实业有限公司上班时通过微信添加了一名男性好友（微信号：prr00254），之后对方多次向报警人推荐一个博彩平台，报警人于2018年9月9日通过扫对方提供的二维码进入该平台，对方让其在平台中充值彩金（一元彩金等于一元人民币），报警人于9月9日22时、9月10日23时通过微信转账分别充值100元、1000元，并成功从中连本带赢提现1230元。后对方又向报警人介绍充值返利的服务，报警人于9月11日至9月13日期间先后11次通过微信绑定的工商银行卡（6212264301009842947）、支付宝绑定的建设银行卡（6217001370018409722）向对方提供的建行卡（6217003890008670551）转账60000元。后发现无法提现，意识到被骗，遂报警。"

    # preprocess_text(value)
    f = open("./data/test.txt", "r", encoding='utf8')
    files = f.read()
    f.close()
    files = re.split(r'\n', files);  # 分割不同的文书
    n = len(files)  # 一共有n个文书

    for file in files:
        if file != "":
            preprocess_text(file)


