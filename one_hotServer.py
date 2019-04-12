from flask import Flask,render_template,request
from day0411 import oneHotUtil

app = Flask(__name__)

@app.route('/one', methods=['GET','POST'])
def one_hotTest():
    domain = oneHotUtil.getDomain()
    msg = ''
    if request.method == 'POST':
        age = int(request.form['age'])
        workclass = request.form['workclass']
        education = request.form['education']
        occupation = request.form['occupation']
        race = request.form['race']
        sex = request.form['sex']
        hours_per_week = int(request.form['hours_per_week'])
        msg = oneHotUtil.oneHotTest(age,workclass,education,occupation,race,sex,hours_per_week)

    return render_template('oneHot.html',msg=msg,domain=domain)

if __name__ == '__main__':
    app.run(host='203.236.209.96',debug=True)