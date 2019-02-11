from flask import Flask, request, Response, jsonify, send_from_directory
import json
import config
import util
import os
import traceback
import numpy as np
import tensorflow as tf
import time
import wikipedia

port = 5000

if os.getenv("PORT"):
    port = int(os.getenv("PORT"))


app = Flask(__name__, static_url_path='/static')

@app.route('/', methods=['GET'])
def index():
    import socket
    return "it's working! " + socket.getfqdn()

@app.route('/', methods=['POST'])
def indexPost():
  print(json.loads(request.get_data()))
  return jsonify(
    status=200,
    replies=[{
      'type': 'text',
      'content': 'Roger that',
    }],
    conversation={
      'memory': { 'key': 'value' }
    }
  )


@app.route('/wiki', methods=['POST'])
def wiki_search():

    payload = request.get_json(silent=True,force=True)

    print("payload")
    print(payload)

    if payload == None:
        if request.get_data() !=None:
            payload = json.loads(request.get_data())

    print(payload)
    # doesn't work
    #payload = json.loads(request.get_data())

    # Get user request
    if payload != None:

        if payload.get("nlp").get("source"):
            print("message: {}".format(payload.get("nlp").get("source")))
            message = payload.get("nlp").get("source")
            message = message.replace('can you tell me more about','')
            message = message.replace('can you tell me more information about','')
            message = message.replace('can you tell me some information about','')
            message = message.replace('can you please tell me more about it','')
            message = message.replace('please tell me more information about','')
            message = message.replace('can you give me more information about','')
            message = message.replace('can you please give me more information about','')
            message = message.replace('can you provide more information about','')
            message = message.replace('can you please provide more information about','')
            message = message.replace('can you explain a little bit more what is a','')
            message = message.replace('can you please explain a little bit more what is a','')


            message = message.replace('?','')

            query = message

            if message.replace(" ", "") == 'it' or message.replace(" ", "") == 'this':
                # Check memory
                if payload.get("conversation").get("memory") != None:
                    plankton = payload.get("conversation").get("memory").get("plankton")
                    query = plankton

        elif payload.get("conversation").get("memory") != None:
            plankton = payload.get("conversation").get("memory").get("plankton")
            query = plankton

    response = wikipedia.summary(query, sentences=2)

    return jsonify(
      status=200,
      replies=[{
        'type': 'text',
        'content': response
      }],
      conversation={
        'memory': {}
      }
    )


@app.route('/tensorclassify', methods=['POST'])
def tf_classify():
    # TODO: python -m scripts.label_image     --graph=tf_files/retrained_graph.pb      --image=test/aurelia.jpeg
    import socket

    print("In tf_classify handler from {}".format(socket.getfqdn()))

    file_name = "models/mobilenet/example/3475870145_685a19116d.jpg"
    file_name = "https://www.eopugetsound.org/sites/default/files/styles/magazinewidth_592px/public/topical_article/images/moon_jellyfish.jpg?itok=Esreg6zX"

    # print("req = json.loads(request.get_data())")
    # print(json.loads(request.get_data()))


    payload = request.get_json(silent=True,force=True)

    print("payload")
    print(payload)

    if payload == None:
        if request.get_data() !=None:
            payload = json.loads(request.get_data())

    # doesn't work
    #payload = json.loads(request.get_data())


    if payload != None:
        if payload.get("nlp").get("entities").get("url"):
            file_name = payload.get("nlp").get("entities").get("url")[0].get("raw")


            model_file = "models/mobilenet/retrained_graph.pb"
            label_file = "models/mobilenet/retrained_labels.txt"
            input_height = 224
            input_width = 224
            input_mean = 128
            input_std = 128
            input_layer = "input"
            output_layer = "final_result"

            graph = util.load_graph(model_file)
            t = util.read_tensor_from_image_file(file_name,
                                            input_height=input_height,
                                            input_width=input_width,
                                            input_mean=input_mean,
                                            input_std=input_std)

            input_name = "import/" + input_layer
            output_name = "import/" + output_layer
            input_operation = graph.get_operation_by_name(input_name);
            output_operation = graph.get_operation_by_name(output_name);

            with tf.Session(graph=graph) as sess:
              start = time.time()
              results = sess.run(output_operation.outputs[0],
                                {input_operation.outputs[0]: t})
              end=time.time()

            results = np.squeeze(results)

            top_k = results.argsort()[-5:][::-1]
            labels = util.load_labels(label_file)

            print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
            template = "{} (score={:0.5f})"

            print(top_k)

            for i in top_k:
              print(template.format(labels[i], results[i]))

            # I really don't know, my best guess is []

            if results[0] < 0.1:
                response = "I really don't know, my best guess is that this looks like a " + labels[top_k[0]]
            else:
                response = 'I think this is a ' + labels[top_k[0]]

            response =  'I think this is a ' + labels[top_k[0]]

            return jsonify(
              status=200,
              replies=[{
                'type': 'text',
                'content': response
              }],
              conversation={
                'memory': { 'plankton': labels[top_k[0]] }
              }
            )



@app.route('/errors', methods=['POST'])
def errors():
  print('in /errors !')
  print(json.loads(request.get_data()))
  return jsonify(status=200)

#app.run(port=port)

if __name__ == '__main__':
  print("Launching web application...")
  #app.run(host='0.0.0.0', port=port, ssl_context=('./keys/keys.key', './keys/keys.crt'))
  #app.run(host='0.0.0.0', port=port)
  app.run(debug=True,host='0.0.0.0', port=port)
