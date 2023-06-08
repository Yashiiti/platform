import ast
import math
import pandas as pd

allowed_funcs = {'abs', 'math.sqrt', 'sin', 'cos', 'tan'}  # List of allowed functions

def sanitize_input(expression):
    tree = ast.parse(expression, mode='exec')

    # Check if the expression consists of a single assignment
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        raise ValueError("Invalid expression. Only variable assignment is allowed.")

    assign_node = tree.body[0]

    # Check if the target of assignment is a valid variable name
    if not isinstance(assign_node.targets[0], ast.Name):
        raise ValueError("Invalid target for assignment.")

    target_name = assign_node.targets[0].id

    # Check if the assigned value is a valid expression
    if not isinstance(assign_node.value, ast.Call):
        raise ValueError("Invalid assigned value.")

    function_name = get_function_name(assign_node.value)

    # Check if the function used in the expression is allowed
    if function_name not in allowed_funcs:
        raise ValueError("Invalid function used in the expression.")

    return expression

def get_function_name(node):
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return get_function_name(node.value) + '.' + node.attr
    else:
        raise ValueError("Invalid function call.")

variables = {}

def sqr(p):
    return p*p
lines = ['x=op', 'y=cl', 'alphaexp=y-x']
tickers=['AAPL','GOOG','MSFT']
alphas=[]
for i in range(len(tickers)):
    alphas.append([])
opens=[]
closes=[]
for k in range(len(tickers)):
    data= pd.read_csv("{}.csv".format(tickers[k]))
        # print(data)
    stk_close = data.reset_index()['Close']
    # closes.append(list(stk_close))
    stk_open = data.reset_index()['Open']
    # opens.append(list(stk_open))
    s=0
    lc=[]
    lo=[]

    for i in range(5):
        op=stk_open[i]
        lo.append(stk_open[i+1])
        cl=stk_close[i]
        lc.append(stk_close[i+1])
    
        try:
            # line = sanitize_input(line)
            for line in lines:
                exec(line, globals(), variables)
            z=variables['alphaexp']
            # print(variables['x'],variables['y'])
            # print(z)
            alphas[k].append(z)
            
        except ValueError as e:
            print(f"Error: {str(e)}")
    opens.append(lo)
    closes.append(lc)

print(alphas)
print(closes)
print(opens)
for j in range(len(alphas)):
    x=0
    for i in range(len(tickers)):
       x+=alphas[i][j]
    x=x/len(tickers)
    for i in range(len(tickers)):
        alphas[i][j]-=x
# print(alphas)
val=1000
holdings=[0,0,0]
pnl=0
for i in range(len(alphas[0])):
    for j in range(len(alphas)):
        pnl+=holdings[j]*closes[j][i]
        x=holdings[j]-alphas[j][i]
        val=val-x*opens[j][i]
        print(pnl)
        print(val)
        print(holdings)


        holdings[j]=alphas[j][i]
        

        



    

        
        
# print(alphas)


            # z=variables['alpha']
            # print(z)
        
            
            
        


# Print the final values of variables
# for variable, value in variables.items():
#     print(f"{variable} = {value}")
