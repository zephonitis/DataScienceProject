{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a8d8bedf-2364-48f5-93ca-b11fd67200a5",
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask, request, jsonify\n",
    "import joblib\n",
    "import numpy as np\n",
    "import os\n",
    "\n",
    "# Load the trained model\n",
    "model = joblib.load(\"model.pkl\")\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    try:\n",
    "        data = request.get_json(force=True)\n",
    "        \n",
    "        # Expecting input as: { \"features\": [value1, value2, ..., valueN] }\n",
    "        features = data.get(\"features\", None)\n",
    "\n",
    "        if features is None:\n",
    "            return jsonify({\"error\": \"Missing 'features' key in JSON\"}), 400\n",
    "\n",
    "        # Ensure it's in the right format for prediction\n",
    "        features_array = np.array(features).reshape(1, -1)\n",
    "        prediction = model.predict(features_array)\n",
    "\n",
    "        return jsonify({'prediction': int(prediction[0])})\n",
    "\n",
    "    except Exception as e:\n",
    "        return jsonify({\"error\": str(e)}), 500\n",
    "\n",
    "app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
