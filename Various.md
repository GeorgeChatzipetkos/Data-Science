# Various

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
