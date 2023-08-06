from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from PIL import Image
from source import preprocess_input_image, batch_predict, conv_float_int, combine_image, load_trained_model, burn_area
import numpy as np
from keras import backend as K
from tensorflow.python.lib.io import file_io
import boto3
from keras.models import load_model

app = Flask(__name__)

@app.route('/result', methods=['GET', 'POST'])
def home():
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            uploaded_image = Image.open(file)
            
            model, session = load_trained_model("temp_model.h5")
            K.set_session(session)

            input_image_array = np.array(uploaded_image)
            original_width, original_height, pix_num = input_image_array.shape
            new_image_array, row_num, col_num = preprocess_input_image(input_image_array)

            preds = batch_predict(new_image_array, model)
            output_pred = conv_float_int(combine_image(preds, row_num, col_num, original_width, original_height, remove_ghost=True)[:,:,0])
            preds_t = (preds > 0.25).astype(np.uint8)
            output_mask = conv_float_int(combine_image(preds_t, row_num, col_num, original_width, original_height, remove_ghost=False)[:,:,0])

            area, biomass_burnt, equal_days = burn_area(output_mask=output_mask, resolution=float(request.form['resolution']), forest_type=request.form['forest_type'])

            return render_template('result.html', image_path=file.filename, output_pred=output_pred, output_mask=output_mask, area=area, biomass_burnt=biomass_burnt, equal_days=equal_days)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
