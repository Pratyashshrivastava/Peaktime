from datetime import datetime
now=[]
while True:
    curr = now.append(datetime.now())
for i in now:
    date_time_str = i.strftime("%Y-%m-%d %H:%M:%S")
    print('DateTime String:', date_time_str)