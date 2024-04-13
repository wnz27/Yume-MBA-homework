'''
Author: 27
LastEditors: 27
Date: 2024-04-13 18:28:42
LastEditTime: 2024-04-13 22:35:07
FilePath: /Yume-MBA-homework/yumi/src/decision_tree.py
description: type some description
'''
import csv
from typing import Dict, List, Tuple

from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn import preprocessing
# 该库已剔除，更改为 six 库
# from sklearn.externals.six import StringIO
from six import StringIO
from pydantic import BaseModel, Field

channel_type = {
    "Ajio": 1,
    "Amazon": 2,
    "Flipkart": 3,
    "Meesho": 4,
    "Myntra": 5,
    "Nalli": 6,
    "Others": 7,
}

product_type_map = {
    "Blouse": 1,
    "Bottom": 2,
    "Ethnic Dress": 3,
    "kurta": 4,
    "Saree": 5,
    "Set": 6,
    "Top": 7,
    "Western Dress": 8,
}

gender_type = {
    "Women": 1,
    "Men": 2
}

order_status_map = {
    "Cancelled": 1,
    "Delivered": 2,
    "Refunded": 3,
    "Returned": 4
    }

def calculate_is_buy(status: int) -> bool:
    if status == 2:
        return True
    else: 
        return False


age_group_type = {
    "Teenager": 1,
    "Adults" : 2,
    "Senior": 3
}

def calculate_age_group(age: int) -> int:
    if age < 30:
        return 1
    elif age >= 30 and age < 50:
        return 2
    else:
        return 3

def get_gender_int(gender: str) -> int:
    return gender_type.get(gender, 1)

def get_channel_int(channel: str) -> int:
    return channel_type.get(channel, 7)

def get_product_type_int(product_type: str) -> int:
    return product_type_map.get(product_type, 7)

class OrderItem(BaseModel):
    gender: int = Field(title="性别")
    # age: int = Field(title="年龄")
    """
    Adults >=30 < 50
    Senior >= 50
    Teenager <30
    """
    age_group: int = Field(title="年龄组")
    channel: int = Field(title="购物渠道")
    product_type: int = Field(title="产品类型")
    price: float = Field(title="价格")
    qty: int = Field(title="数量")
    is_buy: bool = Field(title="是否购买")  
    
    @classmethod
    def BuildOrderItem(cls, gender: str, age: int, channel: str, product_type: str, price: float, qty: int, status: str) -> "OrderItem":
        return cls(
            gender=get_gender_int(gender),
            age_group=calculate_age_group(age),
            channel=get_channel_int(channel),
            product_type=get_product_type_int(product_type),
            price=price,
            qty=qty,
            is_buy=calculate_is_buy(order_status_map.get(status, 1))
        )

    def to_feature_dict(self) -> Dict:
        d = self.model_dump()
        d.pop("is_buy")
        return d


def get_feature_list_and_label_List(
        order_list: List[OrderItem],
    ) -> Tuple[List[Dict], List[bool]]:
    '''
    将其特征值存储在列表featureList中，
    将预测的目标值存储在labelList中
    '''
    feature_list = []
    label_list = []
    for order in order_list:
        feature_list.append(order.to_feature_dict())
        label_list.append(order.is_buy)
    return feature_list, label_list
    

def extract_csv_data(csv_file_url: str) -> List[OrderItem]:


    row_c = 1
    order_list = []
    # 打开 csv 文件
    with open(csv_file_url, "r") as csvfile:
        # 读取 csv 文件
        csvreader = csv.reader(csvfile)
        # 获取 csv 文件的第一行
        header = next(csvreader)
        # 遍历 csv 文件的每一行
        for row in csvreader:
            # if row_c == 1:
            #     print(row)
            #     break
            gender = row[3]
            age = row[4]
            channel = row[9]
            product_type = row[11]
            price = row[15]
            qty = row[13]
            status = row[8]
            order_list.append(OrderItem.BuildOrderItem(
            gender, int(age), channel, product_type, float(price), int(qty), status
            ))
            row_c += 1
    # 遍历 csv 每一行
    print("------->", len(order_list))
    return order_list

def decision_tree_demo():
    order_list = extract_csv_data("/Users/f27/self_biz/Yume-MBA-homework/yumi/src/data_set1.csv")
    feature_list, label_list = get_feature_list_and_label_List(order_list)
    
    # Vetorize features:将特征值数值化
    vec = DictVectorizer()    #整形数字转化
    dummyX = vec.fit_transform(feature_list) .toarray()   #特征值转化是整形数据
 
    print("dummyX: " + str(dummyX))
    print(vec.get_feature_names_out())
 
    # print("labelList: " + str(label_list))
 
    # vectorize class labels
    lb = preprocessing.LabelBinarizer()
    dummyY = lb.fit_transform(label_list)
    print("dummyY: \n" + str(dummyY))
    
    # 使用决策树进行分类预测处理
    # clf = tree.DecisionTreeClassifier()
    #自定义采用信息熵的方式确定根节点
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(dummyX, dummyY)
    print("clf: " + str(clf))
    
    # Visualize model
    with open("/Users/f27/self_biz/Yume-MBA-homework/yumi/src/allElectronicInformationGainOri.dot", 'w') as f:
        f = tree.export_graphviz(clf, feature_names=vec.get_feature_names_out(), out_file=f)
    



