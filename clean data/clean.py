import pandas as pd

chunk_size = 100000
source = "ust_email.csv"
out_put = "ust_data.csv"

select = [
    "主题","正文","发件人: (姓名)","发件人: (地址)","发件人: (类型)",
    "敏感度","重要性"
]

new_name = {
    "主题" : "head", "正文": "text" , "发件人: (姓名)": "name" , 
    "发件人: (地址)" : "address" , "发件人: (类型)" : "sender_type",
    "敏感度" :"sensitivity",
    "重要性" : "importance"
} 

swap_order = [
    "主题",
    "发件人: (姓名)",
    "发件人: (地址)",
    "敏感度","重要性",
    "发件人: (类型)",
    "正文"
]

open("ust_data.csv" , "w").close()  ###empty

with pd.read_csv(source , chunksize=chunk_size , usecols=select) as f:
    for i , chunk in enumerate(f):
        
        chunk = chunk[swap_order]
        chunk["正文"] = chunk["正文"].str.replace("\n" , " " , regex=False)

        chunk.to_csv(out_put , mode="a" , index=False , header=(i == 0) )
        print(f"processed {i}")
    #f.rename(columns=new_name , inplace=True)

df = pd.read_csv("ust_data.csv")
df.columns = [
    "head" , "name" , "address" , "sensitivity" ,"importance" , "type" , "text"
]
df["text"] = df["text"].fillna("").str.replace(r"[\r\t\n]" , " " , regex=True)

df.to_csv("ust_data.csv" , index=False)


