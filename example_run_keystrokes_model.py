import onnxruntime as rt
import pandas as pd
import numpy as np
import re
import json

# ==================================================================
# 1. BAKE METADATA (Run this section once to generate constants)
# ==================================================================
# If variables exist in memory, we convert them to static constants.
# If you are running this for the first time after a restart, ensure 
# you have run the training/preprocessing cells once.
try:
    FEATURE_COLS = X_imp.columns.tolist()
    FEATURE_MEDIANS = X_imp.median().to_dict()
    LABELS = le.classes_.tolist()
    
    # Note to user: If you want to move this code to a production script,
    # you can print() these variables and paste them here as fixed lists/dicts.
except NameError:
    print("CRITICAL: Training variables (X_imp, le) not found.")
    print("Please run the feature building cell (1L5GZTQdfcLo) and the selection cell (YDONKdO8gOZw) first.")
    raise

# ==================================================================
# 2. ESSENTIAL HELPER FUNCTION
# ==================================================================
def parse_ikdd_file(path):
    with open(path, "r", encoding="utf-8") as f: 
        lines = [ln.strip() for ln in f if ln.strip()]
    meta_p = lines[0].split(",")
    meta = {"user_id": meta_p[0], "age_group": meta_p[2]}
    records = []
    dash_pattern = re.compile(r"[\u2013\u2014-]")
    for line in lines[1:]:
        try:
            kp, *vals = line.split(",")
            k1, k2 = map(int, dash_pattern.split(kp))
            if vals: records.append({"key1": k1, "key2": k2, "values": np.mean([float(v) for v in vals if v.strip()])})
        except: continue
    df = pd.DataFrame(records)
    if not df.empty: df["user_id"] = meta["user_id"]
    return meta, df

# ==================================================================
# 3. STANDALONE INFERENCE LOGIC
# ==================================================================
def predict_standalone(file_path, onnx_path):
    # Load session
    sess = rt.InferenceSession(onnx_path)
    
    # Parse input
    meta, df = parse_ikdd_file(file_path)
    if df.empty: return "No data", "N/A"
    
    # Feature Extraction
    dur = df[df['key2'] == 0].pivot_table(index='user_id', columns='key1', values='values', aggfunc='mean')
    dur.columns = [f'dur_{int(c)}' for c in dur.columns]
    lat = df[df['key2'] != 0].copy()
    lat['dig'] = lat['key1'].astype(str) + '_' + lat['key2'].astype(str)
    dig = lat.pivot_table(index='user_id', columns='dig', values='values', aggfunc='mean')
    dig.columns = [f'dig_{c}' for c in dig.columns]
    feats = pd.concat([dur, dig], axis=1)
    
    # Align to the features baked into the ONNX model
    X_input = pd.DataFrame(index=feats.index, columns=FEATURE_COLS)
    for col in FEATURE_COLS:
        if col in feats.columns:
            X_input[col] = feats[col]
        else:
            X_input[col] = FEATURE_MEDIANS[col]
    
    # Convert to numeric and ensure float32 for ONNX
    X_val = X_input.apply(pd.to_numeric).values.astype(np.float32)
    
    # ONNX Run
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_index = sess.run([label_name], {input_name: X_val})[0][0]
    
    return LABELS[pred_index], meta['age_group']

# ==================================================================
# 4. EXECUTION
# ==================================================================
test_file = 'IKDD/IKDD/any_ks&dl_user001_(1).txt'
onnx_file = 'keystrokes_full_pipeline_model.onnx' # Use the integrated model exported earlier

prediction, actual = predict_standalone(test_file, onnx_file)
print(f"File: {test_file}")
print(f"Predicted Age Group: {prediction}")
print(f"Actual Age Group: {actual}")