# Data-Wrangling
- Combine multiple Excel files

```ruby
import glob
import pandas as pd
import os
  
#list with all .xlsx files in the folder
all_files = [i for i in glob.glob(r'C:\Users\*.xlsx')]

#combine all files in the list
combined_excel = pd.concat([pd.read_excel(f, header = 0).assign(FileName=os.path.basename(f)) for f in all_files],ignore_index=True)

#write file
combined_excel.to_excel(r'C:\Users\file.xlsx', index=False)
```

- Combine Multiple Excel Worksheets

```ruby
import pandas as pd

#Combine Multiple Excel Worksheets Into a Single Pandas Dataframe
df = pd.concat(pd.read_excel(r'C:\Users\file.xlsx', sheet_name=None), ignore_index=True)

#Combine Multiple Excel Worksheets Into a Single Pandas Dataframe with a new column for each sheet name
dfs = pd.read_excel(r'C:\Users\file.xlsx', sheet_name=None)

dfs = {k: v.loc[:, ~v.columns.str.contains('Unnamed')] for k, v in dfs.items()}

df = (pd.concat(dfs)
        .reset_index(level=1, drop=True)
        .reset_index())
```

- Read last column from Excel

```ruby
import pandas as pd

xl = pd.ExcelFile(r'C:\Users\file.xlsx')
ncols = xl.book.sheets()[0].ncols
df = xl.parse(0, usecols=[0, 1, ncols-1])
```

- Replace file names in a folder

```ruby
import os

path = os.chdir(r'N:\...') # set current working directory above to the required path

filenames = os.listdir(path)
for filename in filenames:
    os.rename(filename, filename.replace("nm", "NM").replace("mt4 NM", "NM").replace("mt5 NM", "NM").replace("NM mt4", "NM").replace("NM mt5", "NM"))
```

- Split to multiple Excel sheets

```ruby
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
- loop through multiple dfs

```ruby
output_list = []
for df in [df1, df2, df3]:
    df['Month'] =df['Date'].apply(lambda x : x.strftime('%Y-%m'))
    output_list.append(df)
  
df1 = output_list[0]
df2 = output_list[1]  
df3 = output_list[2]
```

- create conditional column

```ruby
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

- outer join multiple dfs

```ruby
data_frames = [df1,df2,df3,df4]
df = reduce(lambda  left,right: pd.merge(left,right,on=['ID'], how='outer'), data_frames).fillna(0)
```

- Textjoin a df column

```ruby
IDs = ",".join([str(element) for element in df['User ID'].tolist()])
```

- Assign timezone (UTC to Athens time)

```ruby
startDate = "2021-01-01" 
endDate = "2021-12-31"

startDateTime = startDate + " 00:00:00"
endDateTime = endDate + " 23:59:59"

UserDatastartDateTime = timezone('Europe/Athens').localize(datetime.strptime(startDateTime, "%Y-%m-%d %H:%M:%S")).astimezone(timezone('UTC'))
UserDatastartDateTime = UserDatastartDateTime.strftime("%Y-%m-%d %H:%M:%S")
UserDataendDateTime = timezone('Europe/Athens').localize(datetime.strptime(endDateTime, "%Y-%m-%d %H:%M:%S")).astimezone(timezone('UTC'))
UserDataendDateTime = UserDataendDateTime.strftime("%Y-%m-%d %H:%M:%S")
```

- Stip timezone from datetime object

```ruby
df['Registration Date'] = df['Registration Date'].dt.tz_convert('Europe/Athens')
df['Registration Date'] = pd.to_datetime(df['Registration Date']).dt.tz_localize(None)
```

- String to date object

```ruby
from datetime import date

endDate = "2022-02-28"
#date_object = datetime.strptime(endDate, '%Y-%m-%d').date()
```
