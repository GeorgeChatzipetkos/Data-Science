# https://stevenkyle2013.medium.com/using-etherscan-python-library-with-etherscan-io-api-to-access-ethereum-data-cd44c3e34190
# https://github.com/pcko1/etherscan-python/tree/master/logs/standard

from etherscan import Etherscan
import pandas as pd
import numpy as np
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as soup

startDate = "2022-01-01" 
endDate = "2022-03-15"

eth = Etherscan('SE3XXRIHR5C2S392BI7C87NQY8PSKCW1N4')

Wallet_Address = '0xd6216fc19db775df9774a6e33526131da7d19a2c' #KuCoin 6
Wallet_Addresses = ['0xbe0eb53f46cd790cd13851d5eff43d12404d33e8',
                   '0x00000000219ab540356cbb839cbe05303d7705fa',
                   '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',
                   '0xda9dfa130df4de4673b89022ee50ff26f6ea73cf']

Contract_Address = '0xdAC17F958D2ee523a2206206994597C13D831ec7' # example for Tether USD

#balance of a wallet address
eth_balance = eth.get_eth_balance(Wallet_Address)
print(float(eth_balance)/1000000000000000000) #  original output unit is in wei and we have to convert it to eth

#balance of multiple wallet addresses
eth_balances = eth.get_eth_balance_multiple(Wallet_Addresses)
for i in range(len(eth_balances)):
    print('Balance for Wallet ',eth_balances[i]['account'], ' is ',float(eth_balances[i]['balance'])/1000000000000000000) #  original output unit is in wei and we have to convert it to eth
eth_balances_df = pd.DataFrame(eth_balances)
eth_balances_df["balance"] = eth_balances_df["balance"].apply(str).apply(float)/1000000000000000000

# latest price of eth
eth.get_eth_last_price()

# balance of a token
eth_balance_token = eth.get_acc_balance_by_token_and_contract_address(
    address = Wallet_Address,
    contract_address = Contract_Address
    )
print(float(eth_balance_token)/1000000) # convert to USDT

# ERC-20 transfers
ERC20Transfers = pd.DataFrame(eth.get_erc20_token_transfer_events_by_address(Wallet_Address,0,999999999,'asc')) #to this address
ERC20Transfers['Datetime'] = pd.to_datetime(ERC20Transfers['timeStamp'],unit='s')
ERC20Transfers["Value Norm."] = ERC20Transfers["value"].apply(str).apply(float)/10**ERC20Transfers['tokenDecimal'].astype(int)

""" Download top accounts """ # on the website there are 10000 accounts but the loop returns less and different number each time. A solution would be to run the code on every 30 pages until page 100
dfs = []
for i in range(1,101,1):
    try:
        url = 'https://etherscan.io/accounts/' + str(i) +'?ps=100' #OR %d %i
        req = Request(url, headers={'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'})   # I got this line from another post since "uClient = uReq(URL)" and "page_html = uClient.read()" would not work (I beleive that etherscan is attemption to block webscraping or something?)
        with urlopen(req, timeout=21) as response:
            response = response.read()
        #response = urlopen(req, timeout=20).read()
        #response_close = urlopen(req, timeout=20).close()
        page_soup = soup(response, "html.parser") # or lxml or html5lib
        Transfers_info_table_1 = page_soup.find("table", {"class": "table table-hover"})
        df=pd.read_html(str(Transfers_info_table_1))[0]
        dfs.append(df)
    except ValueError:
         continue

TopAccounts = pd.concat(dfs)



