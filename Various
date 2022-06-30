# Various

- **Create path for Year & Month**

```python
import os
from datetime import datetime, timedelta
#from pytz import timezone

date = datetime.now()
year = str(date.year)
month = date.strftime("%B")
#date = date.strftime("%Y-%m-%d")

os.chdir(r'C:\Users\georgiosc\Downloads')

if not os.path.exists(year):
    os.makedirs(year)
os.chdir(year)

if not os.path.exists(month):
    os.makedirs(month)
os.chdir(month)
```
