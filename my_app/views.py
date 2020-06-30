from django.shortcuts import render
import joblib
import numpy as np
from numpy import array
from django.contrib import messages

from tensorflow.keras.models import load_model
model = load_model("E:\python projects\Deep learning\power_prediction\modelAI\model.h5")
test_data = joblib.load("E:\python projects\Deep learning\power_prediction\modelAI\data")
scaler = joblib.load("E:\python projects\Deep learning\power_prediction\modelAI\scalers.joblib")

import pandas as pd



def myform(request):
    if request.method == 'POST':
        num=request.POST['number']
        print(num)
        messages.success(request, 'hyy')
        print(test_data)
        x_input = test_data[9961:].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()

        len(temp_input)

        lst_output = []
        n_steps = 144
        i = 0
        j=6*int(num)
        print(j)

        date_rng = pd.date_range(start='01/01/2019 00:00:00', periods=j, freq='10T')

        while (i < j):

            if (len(temp_input) > 144):
                # print(temp_input)
                x_input = np.array(temp_input[1:])
                # print("{} day input {}".format(i,x_input))
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, 144, 1))
                # print(x_input)
                yhat = model.predict(x_input, verbose=0)
                print("{} : {}".format(date_rng[i], scaler.inverse_transform(yhat)))
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                # print(temp_input)
                lst_output.extend(yhat.tolist())
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i = i + 1
        res={'dt':date_rng,'lst':scaler.inverse_transform(lst_output)}
        res['r'] = zip(res['dt'], res['lst'])

        return render(request, 'form.html',res)




    return render(request, 'form.html')




