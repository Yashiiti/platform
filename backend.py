from flask import Flask, request, render_template,send_file
import pandas as pd
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import matplotlib.pyplot as plt
app = Flask(__name__)


@app.route('/calculate', methods=['POST'])
def calculate():
    expression = request.form['expression']
    lines=expression.split(";")

    # Your calculation logic here
    
    data= pd.read_csv("AAPL.csv")
        # print(data)
    stk_close = data.reset_index()['Close']
    stk_open = data.reset_index()['Open']
    s=0
    alpha=[]
    for i in range(len(stk_close)-1):
        
        # op=stk_open[i]
        # cl=stk_close[i]
        
        variables={'op':stk_open[i],'cl':stk_close[i]}
        # lines=['x=op', 'y=cl', 'alphaexp=x-y']
        
            # line = sanitize_input(line)
        for line in lines:
            exec(line, globals(), variables)
        z=variables['alphaexp']
        # print(z)
        if z>0:
            s+=stk_close[i+1]-stk_close[i]
        else:
            s+=stk_close[i]-stk_close[i+1]
        alpha.append(s)
        print(s)


        # z=variables['alpha']
        # print(z)
    
        
        
    # except ValueError as e:
    #     print(f"Error: {str(e)}")
            # return s
    plt.plot(alpha)
    plt.title('Profit')
    plt.xlabel('Days')
    plt.ylabel('Profit')
    
    # Save the plot as an image file
    image_file = io.BytesIO()
    plt.savefig(image_file, format='png')
    image_file.seek(0)

    # Clear the plot
    plt.clf()
    
    return send_file(image_file, mimetype='image/png')
    return "profit is {}".format(alpha[-1])
    

@app.route('/')
def index():
    return render_template('frontend.html')

if __name__ == '__main__':
    app.run()
