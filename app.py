import flask
import openCV_pic as cvtest
from flask import jsonify,render_template
import openCV_pic2 as cvtest2

app = flask.Flask(__name__)
app.config["DEBUG"] = True

ar_type={"id":1,"face":True,"name":"andy"}

@app.route('/', methods=['GET'])
def home():
    return "<h1>Hello Flask!</h1> <h3>This is a face recognition website</h3>"
@app.route('/face', methods=['GET'])
def show_face():
    return render_template('face.html')
@app.route('/getface/<int:Istrain>', methods=['POST','GET'])
def getface(Istrain):
    if(Istrain==1):
        Istrain=True
    else :
        Istrain=False
    # if(flask.request.method=='POST'):
    #     Istrain=True
    print(Istrain)
    list_img_filepath=cvtest.build_keras_model(Istrain)
    image_html="<img src=\"{image_path}\" alt=\"User Image\">"
    final_html=""
    for(i,k) in enumerate(list_img_filepath):
        final_html=final_html+image_html.format(image_path=k)
    return render_template('app.html',image_list=list_img_filepath)
@app.route('/index', methods=['GET'])
def show_page():
    return render_template('index.html',user_image="<h1>GO GO FACE!</h1>")
    #return json
    #return jsonify(ar_type)
#未完成
@app.route('/test', methods=['GET'])
def test_page():
    cvtest2.build_keras_model()
    return "<h1>test</h1>"
# web api run
app.run(port=5000, debug=True)