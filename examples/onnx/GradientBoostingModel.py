import os
import numpy as np
import pandas as pd
import m2cgen as m2c
import sklearn.datasets as datasets
from xgboost import XGBClassifier, XGBRegressor

import onnxruntime as rt
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost

# prevent scientific notations
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# %watermark -a 'Ethen' -u -d -v -p numpy,pandas,sklearn,m2cgen,xgboost

n_features = 5
X, y = datasets.make_classification(n_samples=10000, n_features=n_features, random_state=42, n_classes=2)
feature_names = [f'f{i}'for i in range(n_features)]
print(f'num rows: {X.shape[0]}, num cols: {X.shape[1]}')
X

tree = XGBClassifier(
    n_estimators=20,
    max_depth=3,
    learning_rate=0.2,
    tree_method='hist',
    verbosity=1
)
tree.fit(X, y, eval_set=[(X, y)])

xgboost_checkpoint = 'model.json'
tree.save_model(xgboost_checkpoint)
tree_loaded = XGBClassifier()
tree_loaded.load_model(xgboost_checkpoint)

assert np.allclose(tree.predict_proba(X[:1]), tree_loaded.predict_proba(X[:1]))

input_payloads = [
    {
        'f0': -2.24456934,
        'f1': -1.36232827,
        'f2': 1.55433334,
        'f3': -2.0869092,
        'f4': -1.27760482
    }
]

rows = []
for input_payload in input_payloads:
    row = [input_payload[feature] for feature in feature_names]
    rows.append(row)

np_rows = np.array(rows, dtype=np.float32)
tree.predict_proba(np_rows)[:, 1]

tree.predict_proba(X[:1])

def convert_xgboost_to_onnx(model, num_features: int, checkpoint: str):

    # boiler plate code for registering the xgboost converter
    update_registered_converter(
        XGBClassifier, 'XGBoostXGBClassifier',
        calculate_linear_classifier_output_shapes, convert_xgboost,
        options={'nocl': [True, False], 'zipmap': [True, False, 'columns']}
    )
    # perform the actual conversion specifying the types of our inputs,
    # at the time of writing this, it doesn't support categorical types
    # that are common in boosted tree libraries such as xgboost or lightgbm
    model_onnx = convert_sklearn(
        model, 'xgboost',
        [('input', FloatTensorType([None, num_features]))],
        target_opset={'': 15, 'ai.onnx.ml': 2},
        options={'zipmap': False}
    )


    with open(checkpoint, "wb") as f:
        f.write(model_onnx.SerializeToString())

onnx_model_checkpoint = 'xgboost.onnx'
convert_xgboost_to_onnx(tree, len(feature_names), onnx_model_checkpoint)

sess = rt.InferenceSession(onnx_model_checkpoint)
input_name = sess.get_inputs()[0].name
output_names = [output.name for output in sess.get_outputs()]

np_rows = np.array(rows, dtype=np.float32)
onnx_predict_label, onnx_predict_score = sess.run(output_names, {input_name: np_rows})
onnx_predict_score
