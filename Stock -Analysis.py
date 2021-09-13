from tkinter import * 
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import mysql.connector
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
from tabulate import tabulate

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="root",
  database="stocks"
)

def predictstock():
    tabnam = company.get()
    try:
        st="select * from "+tabnam
        SQL_Query = pd.read_sql_query(st, mydb)
        df = pd.DataFrame(SQL_Query)
        df.columns = SQL_Query.keys()
        df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
        df.index = df['Date']
        data = df.sort_index(ascending=True, axis=0)
        new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
        for i in range(0,len(data)):
            new_data['Date'][i] = data['Date'][i]
            new_data['Close'][i] = data['Close'][i]
        new_data.index = new_data.Date
        new_data.drop('Date', axis=1, inplace=True)
        dataset = new_data.values
        train = dataset[0:987,:]
        valid = dataset[987:,:]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        x_train, y_train = [], []
        for i in range(60,len(train)):
            x_train.append(scaled_data[i-60:i,0])
            y_train.append(scaled_data[i,0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
        inputs = new_data[len(new_data) - len(valid) - 60:].values
        inputs = inputs.reshape(-1,1)
        inputs  = scaler.transform(inputs)

        X_test = []
        for i in range(60,inputs.shape[0]):
            X_test.append(inputs[i-60:i,0])
        X_test = np.array(X_test)

        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        closing_price = model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)

        train = new_data[:987]
        valid = new_data[987:]
        valid['Predictions'] = closing_price
        tab=pd.DataFrame(index=range(0,len(valid)),columns=['Predictions'])
        tab['Predictions']=closing_price
        val=tabulate(tab.tail(29), headers ='keys', tablefmt = 'simple',showindex=False)
        tabl = Label(window, text=val,font=('helvetica', 10),width=10)
        tabl.place(x=30,y=170)
        plot(train,valid)
    except:
        print("Error: unable to convert the data")

def plot(train,valid):
    fig = Figure(figsize = (10,5),dpi = 100)
    plot1 = fig.add_subplot(111)
    plot1.clear()
    plot1.plot(train['Close'])
    plot1.plot(valid[['Close','Predictions']])
    plot1.set_xlabel('Year')
    plot1.set_ylabel('Stocks')
    plot1.legend(['Training value','Pedicted value'])
    plot1.set_title("Closing Price Prediction")
    canvas = FigureCanvasTkAgg(fig,master = window)  
    canvas.draw()
    canvas.get_tk_widget().place(x=145,y=170)

window = Tk()
window.title('Stock price prediction')
window.geometry("1180x730")
window.configure(bg='sky blue')

head=Label(window, text = "Stock Price Prediction", 
          background = 'sky blue', foreground ="green", 
          font = ("Times New Roman", 30))
head.place(x=405,y=2)

company =ttk.Combobox(window, width = 20)
company['values'] = ('tatamotors','hdfcbank','aptech','datamatics','jswsteelns','pnb','sbi','tatastlbslns','tcs')
company['state'] = 'readonly'
company.current(0)
company.place(x=440,y=75)

plot_button = Button(master = window,command = predictstock,height = 1,width = 15,text = "Select company",background = 'white',bd=0,activebackground='#00ff00')
plot_button.place(x=614,y=74)
window.mainloop()