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
