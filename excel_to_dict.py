import pandas as pd
file_path = 'test_one_table.xlsx'
df = pd.read_excel(file_path, None)
first_sheet_name = list(df.keys())[0]  # 获取第一个工作表的名称
first_sheet = df[first_sheet_name]

#去除空行和空列
df = first_sheet.dropna(how='all').dropna(axis=1, how='all')
df = df.reset_index(drop=True)# 重新编写列索引
df.columns = range(df.shape[1])  # 使用数字作为列名


#行列key所在位置
print(df)
for col in range(0, len(df.columns)):
    if not pd.isna(df.iloc[0, col]):
        col_key_location=col
        break
for row in range(0, len(df)):
    if not pd.isna(df.iloc[row, 0]):
        row_key_location=row
        break


keys_location_dict={}
#遍历索引获取索引和位置的键值对
for i in range(0, row_key_location):
    for j in range(col_key_location, len(df.columns)):
        keys_location_dict[(i,j)]=df.iloc[i,j]
for i in range(0, col_key_location):
    for j in range(row_key_location, len(df)):
        keys_location_dict[(j,i)]=df.iloc[j,i]
print(keys_location_dict)


#遍历excel获取键值对
def find_all_keys(row, col) :
    all_keys=[]
    for i in range(0,row_key_location) :
        all_keys.append(df.iloc[i,col])
    for i in range(0, col_key_location):
        all_keys.append(df.iloc[row,i])
    return all_keys

dict={}
for i in range(row_key_location, len(df)):
    for j in range(col_key_location, len(df.columns)):
        dict[tuple(find_all_keys(i,j))]=df.iloc[i,j]


print(dict)