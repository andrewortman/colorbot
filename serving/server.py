#!/bin/env python

import tensorflow as tf
import flask

session = None
model = None
app = flask.Flask("colorbot-app-server")

@app.route("/v1/predict")
def predict():
	color_query = flask.request.args.get("color")
	if color_query == None:
		return

	if len(color_query) > 64:
		return

	color_query = color_query.lower().strip()

	# validation here

	color_model = model.signature_def["color"]
	input = session.graph.get_tensor_by_name(color_model.inputs["inputs"].name)
	output = session.graph.get_tensor_by_name(color_model.outputs["outputs"].name)

	queries = []
	for i in range(0, len(color_query)):
		queries.append(color_query[0:i+1])
	print queries

	vector = session.run(output, {input: queries})
	colors = vector.tolist()

	return flask.jsonify(colors)

@app.route("/")
def index():
	return flask.send_file("index.html")

if __name__ == "__main__":
	session = tf.Session()
	model = tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], "/home/andrewo/Desktop/colorbot/models/trial9/exports/1497576191")

	app.run(host="0.0.0.0")