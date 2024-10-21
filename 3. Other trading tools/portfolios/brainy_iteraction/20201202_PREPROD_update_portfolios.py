import requests
from glob import glob
import pandas as pd
host = 'http://brainy-core.nonprod.xcapit.net'
endpoint = 'portfolio_configurations/'
target_url = host + '/' + endpoint

all_posible_currencies = ["LINKUP",
                          "LINKDOWN",
                          "BTCUP",
                          "BTCDOWN",
                          "BTC",
                          "LINK",
                          "ETH",
                          "LTC",
                          "BNB",
                          "USDT"]

def get_ordered_currencies(temp_currencies):
    ordered_currencies=""
    for curr in all_posible_currencies:
        index = temp_currencies.find(curr)
        if index<0:
            continue
        temp_currencies = temp_currencies[:index]+temp_currencies[index+len(curr):]
        ordered_currencies += f"{curr} "
    ordered_currencies = ordered_currencies[:-1]
    return ordered_currencies

new_configurations = []
for file in glob("../tests/2020-12-01*"):
    df = pd.read_csv(file,index_col="Unnamed: 0")
    quote = file.split("_")[-3].upper()
    temp_currencies = file.split("_")[-5]
    ordered_currencies = get_ordered_currencies(temp_currencies)
    parameters = df.T.iloc[0].to_dict()
    ordered_currencies.replace(" ","-")
    print(ordered_currencies)
    if ordered_currencies == "BTC LINK ETH BNB USDT" and quote=="USDT":
        portfolio_name = "markowitz_con_link_pro_USDT"    
    elif ordered_currencies == "BTC LINK ETH BNB USDT" and quote=="BTC":
        portfolio_name = "markowitz_con_link_pro_BTC"
    
    else:
        portfolio_name = f'markowitz_{ordered_currencies.replace(" ","-")}_quote_{quote}'
    new_configuration = {
            'portfolio': portfolio_name, # the portfolio is created if it does not exist 
            'currency': quote, 
            'strategies_run_frequency': '2h', 
            'strategy_configurations': [        
                    {
                            'strategy': 'general_markowitz', 
                            'base_currency': None, 
                            'time_frame': 120, 
                            'default_weight': '0.75',
                            'parameters':[
                                    { 'name' : 'commission_factor', 'numeric_value':str(round(float(parameters["factorcom"]),2)), 'string_value': None },
                                    { 'name' : 'n_candles', 'numeric_value': str(int(float(parameters["lecturas"]))), 'string_value': None },
                                    { 'name' : 'slope', 'numeric_value':str(round(float(parameters["pendiente"]),2)), 'string_value': None },
                                    { 'name' : 'risk', 'numeric_value':str(round(float(parameters["riesgo"]),2)), 'string_value': None },
                                    { 'name' : 'ordered_currencies', 'numeric_value':None, 'string_value': ordered_currencies },
                    
                    ]
                    },
                    
                    ]
            }
    new_configurations.append(new_configuration)
    if ordered_currencies == "BTC ETH LTC BNB USDT":
        weight = {'classic':'0.85','pro':'0.75'}
        for profile in ['pro','classic']:
            portfolio_name = f'{profile}_{quote}'
            new_configuration = {
                    'portfolio': portfolio_name, # the portfolio is created if it does not exist 
                    'currency': quote, 
                    'strategies_run_frequency': '2h', 
                    'strategy_configurations': [        
                            {
                                    'strategy': 'general_markowitz', 
                                    'base_currency': None, 
                                    'time_frame': 120, 
                                    'default_weight': weight[profile],
                                    'parameters':[
                                            { 'name' : 'commission_factor', 'numeric_value':str(round(float(parameters["factorcom"]),2)), 'string_value': None },
                                            { 'name' : 'n_candles', 'numeric_value': str(int(float(parameters["lecturas"]))), 'string_value': None },
                                            { 'name' : 'slope', 'numeric_value':str(round(float(parameters["pendiente"]),2)), 'string_value': None },
                                            { 'name' : 'risk', 'numeric_value':str(round(float(parameters["riesgo"]),2)), 'string_value': None },
                                            { 'name' : 'ordered_currencies', 'numeric_value':None, 'string_value': ordered_currencies },
                            ]
                            },
                            
                            ]
                    }
            new_configurations.append(new_configuration)
        
        
        

for new_configuration in new_configurations:
    response = requests.post(target_url, json=new_configuration)
    print(response.json())
    



