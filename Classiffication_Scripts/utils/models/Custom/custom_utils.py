import json

CLASS_INDEX = None



def preprocess_input(x):



    # 'RGB'->'BGR'
    x *= (1./255)

    return x


def decode_predictions(preds, top=5, model_json=""):


    global CLASS_INDEX

    if CLASS_INDEX is None:
        CLASS_INDEX = json.load(open(model_json))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        for i in top_indices:
            each_result = []
            each_result.append(CLASS_INDEX[str(i)])
            each_result.append(pred[i])
            results.append(each_result)

    return results