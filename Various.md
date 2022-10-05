# Various

- **Replace file names in a folder**

```python
import os

path = os.chdir(r'N:\Reporting\Other...') # set current working directory above to the required path

filenames = os.listdir(path)

# #replace text with another text
# for filename in filenames:
#     os.rename(filename, filename.replace("nm", "NM").replace("mt4 NM", "NM").replace("mt5 NM", "NM").replace("NM mt4", "NM").replace("NM mt5", "NM") \
#               .replace("MT4 NM", "NM").replace("Mt4 Live1", "NM").replace("MT5 NM", "NM").replace("Live 1", "NM"))
     
# # replace text between two words   
sub_str1 = "Report"
sub_str2 = "Positions"

for filename in filenames:
    try:
        os.rename(filename, filename[:filename.index(sub_str1) + len(sub_str1)] + ' NM' + filename[filename.index('.'):])
    except:
        os.rename(filename, filename[:filename.index(sub_str2) + len(sub_str2)] + ' NM' + filename[filename.index('.'):])
```

- **Convert all .csv to .xlsx in a folder**

```python
import pandas as pd
import os
from glob import glob

for csv_file in glob(r"N:\Reporting\Other\Open Trades 2022\09. September\*.csv"):
   print(csv_file)
   df = pd.read_csv(csv_file,engine="python",delimiter=";", header = 1)
   xlsx_file = os.path.splitext(csv_file)[0] + '.xlsx'
   df.to_excel(xlsx_file, index=None, header=True)
```

- **Create path for Year & Month**

```python
import os
from datetime import datetime, timedelta
#from pytz import timezone

date = datetime.now() # date = datetime.now(timezone('US/Eastern'))
year = str(date.year)
month = date.strftime("%B")
monthNumber = date.strftime("%m")
#date = date.strftime("%Y-%m-%d")

os.chdir(r'C:\Users\georgiosc\Downloads')

if not os.path.exists(year):
    os.makedirs(year)
os.chdir(year)

if not os.path.exists(monthNumber + '. ' + month):
    os.makedirs(monthNumber + '. ' + month)
os.chdir(monthNumber + '. ' + month)

if not os.path.exists('flexibles'):
    os.makedirs('flexibles')
os.chdir('flexibles')
```

- **Run .py with arguments from cmd**

```python
import sys

startDate = sys.argv[1]
endDate = sys.argv[2]
```

- **Search .py files under all subdirectories of the current directory**

```python
import glob

dir_path = '**depos*.py'

for file in glob.glob(dir_path, recursive=True):
    print(file)
```
