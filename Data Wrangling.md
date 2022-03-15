# Data-Wrangling

- **view a dataframe in excel**

```python
import pandas as pd
import xlwings as xw

xw.view(df)
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

- **Split to multiple Excel sheets**

```python
""" split to multiple sheets and check for Excel limit number of rows """

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
- **Import pivot table from excel to Dataframe**

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
df.loc[condition, new_column_name] = new_column_value

#OR

df['NewCol'] = np.where((condition),'NewValue',df['OldCol'])
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

- **left join**

```python
MappingCountry = dict(zip(Countries["Country Code"], Countries["Country"]))
df["Country"] = df["Country Code"].map(lambda x: MappingCountry.get(x,x))
```

- **outer join multiple dfs**

```python
data_frames = [df1,df2,df3,df4]
df = reduce(lambda  left,right: pd.merge(left,right,on=['ID'], how='outer'), data_frames).fillna(0)
```

- **Textjoin a df column**

```python
IDs = ",".join([str(element) for element in df['User ID'].tolist()])
```

- **Get the name of a df in str format**

```python
name =[x for x in globals() if globals()[x] is df][0]
```

- **Assign timezone (UTC to Athens time)**

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

- **Stip timezone from datetime object**

```python
from pytz import timezone
from datetime import datetime, timedelta

df['Registration Date'] = df['Registration Date'].dt.tz_convert('Europe/Athens')
df['Registration Date'] = pd.to_datetime(df['Registration Date']).dt.tz_localize(None)
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

- **Year-Month from date**

```python
df['Year_Month'] = df['Date'].apply(lambda x: x.strftime('%Y-%m'))
```
