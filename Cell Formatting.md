#Cell Formatting

- **adjust autowidth of columns**

```python
import win32com.client as win32

excel = win32.gencache.EnsureDispatch('Excel.Application')
wb = excel.Workbooks.Open(r'C:\Users\georgiosc\Downloads\Partners KPIs.xlsx', ReadOnly=False)
for ws in wb.Sheets:
    ws.Columns.AutoFit()
wb.Save()
excel.Application.Quit()
```

- **highlight cells**

```python
import openpyxl
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Border, Alignment

wb = openpyxl.load_workbook(r'N:\Reporting\04. KPIs\KPIs Report\2022\05. May 2022\Payments - Deposits & Withdrawals\Flexibles\Flexibles '+ yesterday +'.xlsx')
ws = wb.active
ws.conditional_formatting.add('B2:C5', CellIsRule(operator='lessThan', formula=['0'],font = Font(color = '00FF0000')))
for row in ws['A1:C1']:
    for cell in row:
        cell.style = 'Accent1'
wb.save(r'N:\Reporting\04. KPIs\KPIs Report\2022\05. May 2022\Payments - Deposits & Withdrawals\Flexibles\Flexibles '+ yesterday +'.xlsx')
wb.close()
```

- **negative numbers with red**

```python
import openpyxl
from openpyxl import load_workbook

for rows in sheet.iter_cols(min_col=2, max_col=None, min_row=2, max_row=sheet.max_row):
    for cell in rows:
        cell.number_format = '"€"#,##0.00_);[Red]"-€"#,##0.00'
```

- **number format**

```python
import openpyxl
from openpyxl import load_workbook

for rows in sheet.iter_cols(min_col=2, max_col=None, min_row=4, max_row=sheet.max_row):
    for cell in rows:
        cell.number_format = '0.00%'
```

- **freeze panes**

```python
import openpyxl
from openpyxl import load_workbook

sheet = wb['Users Count'] #Name of the working sheet
sheet.freeze_panes = 'B3'
```

- **fill cells and create borders**

```python
import openpyxl
from openpyxl import load_workbook
from openpyxl.styles.borders import Border, Side
import win32com.client as win32

fill_cell = PatternFill(patternType='solid', 
                            fgColor='f1f2f4')

thin_border = Border(left=Side(style='thin',color='d8d9db'), 
                     right=Side(style='thin',color='d8d9db'), 
                     top=Side(style='thin',color='d8d9db'), 
                     bottom=Side(style='thin',color='d8d9db'))

black_border = Border(left=None, 
                     right=Side(style='thin',color='000000'), 
                     top=None, 
                     bottom=None)
                     
i=4 
while i < len(UsersCounts.columns)+1: 
    for rows in sheet.iter_cols(min_col=i+1, max_col=i+1, min_row=2, max_row=sheet.max_row):
        for cell in rows:
          cell.border = thin_border
          cell.fill = fill_cell
    i+=4                     
```

- **Conditional formatting**

```python
from openpyxl import *
from openpyxl.formatting import Rule
from openpyxl.styles import Font, PatternFill, Border, Alignment
from openpyxl.formatting.rule import CellIsRule

ws.conditional_formatting.add('B2:C5', CellIsRule(operator='lessThan', formula=['0'],font = Font(color = '00FF0000')))
```
