# Data-Wrangling

- **time to complete a task**

```python
import time

# Record the start time
start = time.time()

#do some data manipulation

# Record the end time
end = time.time()

print('Command took ', end - start, ' seconds')
```

- **view a dataframe in excel**

```python
import pandas as pd
import xlwings as xw

xw.view(df)
```

- **append df to existing worksheet in excel**

```python
import pandas as pd
from openpyxl import load_workbook

workbook = load_workbook(r'C:\Users\georgiosc\Downloads\MiFIR_2022.xlsx')
writer = pd.ExcelWriter(r'C:\Users\georgiosc\Downloads\MiFIR_2022.xlsx', engine='openpyxl')
writer.book = workbook
writer.sheets = dict((ws.title, ws) for ws in workbook.worksheets)
df.to_excel(writer,sheet_name= 'MiFIR_2022',startrow=len(Olddf)+1, index=False, header=False)
writer.save()
writer.close()
```

- **Combine multiple Excel files**

```python
import glob
import os
  
#list with all .xlsx files in the folder
all_files = [i for i in glob.glob(r'C:\Users\*.xlsx')]

#combine all files in the list
combined_excel = pd.concat([pd.read_excel(f, header = 0).assign(FileName=os.path.basename(f)) for f in all_files],ignore_index=True)

#write file
combined_excel.to_excel(r'C:\Users\file.xlsx', index=False)
```

- **Combine Multiple Excel Worksheets**

```python
#Combine Multiple Excel Worksheets Into a Single Pandas Dataframe
df = pd.concat(pd.read_excel(r'C:\Users\file.xlsx', sheet_name=None), ignore_index=True)

#Combine Multiple Excel Worksheets Into a Single Pandas Dataframe with a new column for each sheet name
dfs = pd.read_excel(r'C:\Users\file.xlsx', sheet_name=None)

dfs = {k: v.loc[:, ~v.columns.str.contains('Unnamed')] for k, v in dfs.items()}

df = (pd.concat(dfs)
        .reset_index(level=1, drop=True)
        .reset_index())
```

- **Read last column from Excel**

```python
xl = pd.ExcelFile(r'C:\Users\file.xlsx')
ncols = xl.book.sheets()[0].ncols
df = xl.parse(0, usecols=[0, 1, ncols-1])
```

- **Replace file names in a folder**

```python
import os

path = os.chdir(r'N:\...') # set current working directory above to the required path

filenames = os.listdir(path)
for filename in filenames:
    os.rename(filename, filename.replace("nm", "NM").replace("mt4 NM", "NM").replace("mt5 NM", "NM").replace("NM mt4", "NM").replace("NM mt5", "NM"))
```

- **Split to multiple Excel sheets (group by existing column)**

```python
""" split to multiple sheets and check for Excel limit number of rows (sheets are named per row number)"""

writer = pd.ExcelWriter(r'C:\Users\file.xlsx')
for group, data in ClosedTrades.groupby('Month'):
    # Get number of parts/chunks... by dividing the df total number of rows by expected number of rows plus 1
    expected_rows = 1048574
    chunks = math.floor(len(data['Login'])/expected_rows + 1)
    
    # Slice the dataframe...
    df_list = []
    
    i = 0
    j = expected_rows
    for x in range(chunks):
        df_sliced = data[i:j]
        
        df_sliced.to_excel(writer,group + ' ROWS_'+str(i), index=False)
        
        i += expected_rows
        j += expected_rows
writer.save()
```

- **Split to multiple Excel sheets (group by given list)**

```python
Names = ['Andr','Est','Myr','Pab','Kons','Zer','Sab','Azi','Ann']
GroupLength = math.floor(len(df)/len(Names)+1) # set nr of rows to slice df

writer =  pd.ExcelWriter(r'C:\Users\georgiosc\Downloads\output.xlsx')
for count,i in enumerate(range(0, len(df), GroupLength)):
    df[i : i+GroupLength].to_excel(writer, sheet_name=Names[count], index=False, header=True)
writer.save()
```

- **Import table from excel as Dataframe**

```python
import openpyxl
from openpyxl import load_workbook

wb = load_workbook(filename = r'N:\.....xlsx')
sheet = wb['Sheet1']
lookup_table = sheet.tables['Table1']

# Access the data in the table range
data = sheet[lookup_table.ref]
rows_list = []

# Loop through each row and get the values in the cells
for row in data:
    # Get a list of all columns in each row
    cols = []
    for col in row:
        cols.append(col.value)
    rows_list.append(cols)

# Create a pandas dataframe from the rows_list.
# The first row is the column names
df = pd.DataFrame(data=rows_list[1:], index=None, columns=rows_list[0])
```

- **loop and filter through multiple dfs**

```python
output_list = []
for df in [df1, df2, df3]:
    df = df.loc[df["Country"].str.contains("Greece")]
    output_list.append(df)
  
df1 = output_list[0]
df2 = output_list[1]  
df3 = output_list[2]
```
- **create conditional column**

```python
df.loc[condition, 'new_column_name'] = 'new_column_value'

#OR

df['NewCol'] = np.where((condition),'NewValue',df['OldCol'])

#OR
df['Category']=df['Monthly Income'].apply(lambda x: 'Graded' if x>=5000 else 'UnGraded')
```

- **create multi-conditional column**

```python
conditions = [
    ((Balance['Amount USD'] >= 500) & (Balance['Amount USD'] < 1000)),
    ((Balance['Amount USD'] >= 1000) & (Balance['Amount USD'] < 2500)),
    ((Balance['Amount USD'] >= 2500) & (Balance['Amount USD'] < 5000)),
    ((Balance['Amount USD'] >= 5000) & (Balance['Amount USD'] < 8000)),
    ((Balance['Amount USD'] >= 8000) & (Balance['Amount USD'] < 10000)),
    (Balance['Amount USD'] >= 10000)
    ] 
      
choices = [0, 10, 20, 30, 40, 50, 60]

Balance['CPA'] = np.select(conditions, choices, default=0)
```

- **df with COUNTIFS**

```python
df_list = []
for group,data  in Users.groupby('Year'):
    df = pd.DataFrame({
    'Time Intervals': ['Less than or equal to 1 day', '1 to 7 days', '7 to 30 days', '30 to 90 days', 'more than 90 days'],
    'Count': [
                        data["Time To live Days"][data["Time To live Days"]<=1].count(),
                        data["Time To live Days"][(data["Time To live Days"]>1) & (data["Time To live Days"]<=7)].count(),
                        data["Time To live Days"][(data["Time To live Days"]>7) & (data["Time To live Days"]<=30)].count(),
                        data["Time To live Days"][(data["Time To live Days"]>30) & (data["Time To live Days"]<=90)].count(),
                        data["Time To live Days"][data["Time To live Days"]>90].count()
                        ]},index = [group]*5)
    df_list.append(df)
    
UsersPivot = pd.concat(df_list).rename_axis('Year').reset_index()
UsersPivot = UsersPivot.pivot(index='Time Intervals',columns='Year',values='Count')

OR

#with bins
cut_labels = ['<=1 day', '>1 <=7 days', '>7 <=30 days', '>30 <=90 days','>90 days']
cut_bins = [0, 1, 7, 30, 90,Users["Time To live Days"].max()]
Users['cut_category'] = pd.cut(Users['Time To live Days'], bins=cut_bins, labels=cut_labels)

UsersPivot = pd.pivot_table(Users, values='Time To live Days', index='cut_category', columns='Year', aggfunc='count').reset_index()
```

- **pivot table with multipl aggfunc**

```python
dfPivot = pd.pivot_table(df, index = ['country'], columns = ['Client Type'], values = ['User ID',"Amount EUR"], \
                         aggfunc = {'User ID':pd.Series.nunique, \
                         'Amount EUR':[np.sum,'count']}, \
                         fill_value = 0)
```

- **left join**

```python
MappingCountry = dict(zip(Countries["Country Code"], Countries["Country"]))
df["Country"] = df["Country Code"].map(lambda x: MappingCountry.get(x,x))
```

- **anti left join**

```python
df3 = df1.merge(df2, on='login', how='left', indicator=True)  
df = df3.loc[df3['_merge'] == 'left_only', 'login']  
d = df1[df1['login'].isin(df)]

OR

df3 = df1.merge(df2, on='login')  
df = df1[~df1['login'].isin(df3['login'])]
```

- **outer join multiple dfs**

```python
from functools import reduce

data_frames = [df1,df2,df3,df4]
df = reduce(lambda  left,right: pd.merge(left,right,on=['ID'], how='outer'), data_frames).fillna(0)
```

- **calculations & Join on Index**

```python
""" one may choose to specify axis='index' (instead of axis=0) or axis='columns' (instead of axis=1) """

df = df1.subtract(df2,axis='index',fill_value = 0)

df1.join(df2) #By default, this performs a left join
'OR'
pd.merge(df1, df2, left_index=True, right_index=True) #By default, this performs an inner join
'OR'
pd.concat([df1, df2], axis=1) #By default, this performs an outer join
```

- **Check if value from one df exists in another df**

```python
Df1.assign(InDf2=Df1.Col.isin(Df2.Col).astype(int))
```

- **Textjoin a df column**

```python
IDs = ",".join([str(element) for element in df['User ID'].tolist()])
```

- **Insert column at specific position**

```python
df.insert(3, 'colname', col) # inserts at third column
```

- **rename column or row**

```python
df = df.rename(columns={"A": "a", "B": "c"}) #columns
df.columns.values[0:2] = ['country',"UTM Campaign"] #columns by position
df.rename(columns={df.columns[2]: 'Courses_Duration'},inplace=True) #columns by position
df = df.rename(index={'Period 2': '% Diff'}) #row
```

- **manipulation of multiindex dataframe column names**

```python
df.loc[:, df.columns.get_level_values(1) == 'RHS'] #select columns
df.columns = df.columns.swaplevel(0, 1)
df = df.droplevel(2, axis=1) #remove column level
df = df.reindex(columns=['Registrations', 'FTDs', 'Conversion %'], level=0) #custom sort
df = df.rename(columns={'count': 'Number of Deposits','sum': 'Total Deposits EUR','nunique': 'Number of Users'}, level=1) #rename columns
df.columns.set_levels(['Number of Deposits','Total Deposits EUR','Number of Users'],level=1,inplace=True) #rename levels
df.columns.levels #get column levels
df.columns = df.columns.rename("Country", level=1) #rename index column if None
```

- **Get the name of a df in str format**

```python
name =[x for x in globals() if globals()[x] is df][0]
```

- **remove rows with only 0s**

```python
df = df[np.count_nonzero(df.loc[:, "Col" : "Coln"].values, axis = 1) > 0]
```

- **Suppress scientific notations**

```python
pd.set_option('display.float_format', lambda x: '%.4f' % x)
OR
pd.options.display.float_format = '{:.4f}'.format
OR
df.applymap(lambda x: '%.4f' % x)
```

- **Check for missing values**

```python
null=pd.DataFrame(df.isnull().sum(),columns=["Missing Values"])
null["% Missing Values"]=round((df.isna().sum()/len(df)*100),2)
```

- **Value counts**

```python
dfCount = pd.concat([
    df['Col'].value_counts(dropna=False),
    df['Col'].value_counts(dropna=False,normalize=True),
    df['Col'].value_counts(dropna=False,normalize=True).mul(100).round(2).astype(str) + '%'],
    axis= 1)
dfCount.columns = ['counts', 'Percentage', 'str%']
```

- **check if website is scrapable**

```python
import requests
from bs4 import BeautifulSoup 

# The output to this should be 200. Anything other than 200 means that the website your trying to scrape either does not allow web scraping or allows partially.
r=requests.get("url")
r.status_code
```

- **parse csv dates**

```python
df = pd.read_csv('file.csv', index_col=None,parse_dates=['DateCol'])
```

- **datetime Col to date**

```python
df['date'] = pd.to_datetime(df['event_time']).dt.date
```

- **Assign timezone to string Datetime (Athens time to UTC)**

```python
from pytz import timezone
from datetime import datetime, timedelta

startDate = "2021-01-01" 
endDate = "2021-12-31"

startDateTime = startDate + " 00:00:00"
endDateTime = endDate + " 23:59:59"

UserDatastartDateTime = timezone('Europe/Athens').localize(datetime.strptime(startDateTime, "%Y-%m-%d %H:%M:%S")).astimezone(timezone('UTC'))
UserDatastartDateTime = UserDatastartDateTime.strftime("%Y-%m-%d %H:%M:%S")
UserDataendDateTime = timezone('Europe/Athens').localize(datetime.strptime(endDateTime, "%Y-%m-%d %H:%M:%S")).astimezone(timezone('UTC'))
UserDataendDateTime = UserDataendDateTime.strftime("%Y-%m-%d %H:%M:%S")
```

- **convert timezone of datetime object**

```python
from pytz import timezone
from datetime import datetime, timedelta

df["Registration Time"] = pd.to_datetime(df["Registration Time"], format = "%Y-%m-%d %H:%M:%S")
df['Registration Time'] = df['Registration Time'].dt.tz_convert('Europe/Athens')
df['Registration Time'] = pd.to_datetime(df['Registration Time']).dt.tz_localize(None)
#df["Registration Date"] = df["Registration Time"].dt.date
```
- **String datetime to timestamp**

```python
from datetime import datetime, timedelta
from datetime import timezone as timez

endDate = "2021-12-31"
endDateTime = endDate + " 23:59:59"

MT5endDate = datetime.strptime(endDateTime, "%Y-%m-%d %H:%M:%S").replace(tzinfo = timez.utc).timestamp()
```

- **String to date object**

```python
from datetime import date

endDate = "2022-02-28"
date_object = datetime.strptime(endDate, '%Y-%m-%d').date()
```

- **Timestamp to Datetime or date**

```python
df['DatetimeCol'] = pd.to_datetime(df['timeStampCol'],unit='s')

OR

df["Date"] = df["Date"].apply(lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d"))
```

- **Remove miliseconds from datetime**

```python
df['datetime'] = df['datetime'].apply(lambda x: x.replace(microsecond = 0))
```

- **Year-Month from date**

```python
df['Year_Month'] = df['Date'].apply(lambda x: x.strftime('%Y-%m'))
```

- **First & Last day of previous month**

```python
from datetime import date, timedelta

FirstDay = ((date.today().replace(day=1)- timedelta(days=1)).replace(day=1)).strftime('%Y-%m-%d')
LastDay = (date.today().replace(day=1)- timedelta(days=1)).strftime('%Y-%m-%d')
```

- **Time Diff in text format**

```python
from dateutil.relativedelta import relativedelta

df["diff"] = df.apply(lambda x: relativedelta(x['End'], x['Start']), axis=1)
df["diff text"] = df.apply(lambda x: "%d days %d hours %d minutes %d seconds" % (x['diff'].days, x['diff'].hours, x['diff'].minutes,x['diff'].seconds), axis=1)

OR

df["diff Seconds"] = df.apply(lambda x: (x["End"] - x["Start"]).total_seconds(), axis = 1) # Diff in seconds
df["diff text"] = df["diff Seconds"].apply(lambda x: timedelta(seconds = x))
```

- **Datetime comparison**

```python
import datetime

df['Time'] = np.where((df['Time']==datetime.datetime(1970, 1, 1, 0, 0, 0)),df['MT4 OpenTime'],df['Time'])

OR

df['Period'] = np.where(((df["Registration Date"] >= startDate1 + " 00:00:00") & (df["Registration Date"] <= endDate1 + " 23:59:59")),'Period 1','Period 2')
```
