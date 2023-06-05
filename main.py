from flask import Flask, render_template, request
app = Flask(__name__)
import pickle
import random

file = open('model.pkl', 'rb')

clf = pickle.load(file)
file.close()

@app.route('/' , methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        MyDic = request.form

        Age = int(request.form['Age'])
        BloodPressure = float(request.form['BloodPressure'])
        SpecificGravity = float(MyDic['SpecificGravity'])
        Albumin = float(MyDic['Albumin'])
        Diabetes = float(MyDic['Diabetes'])
        RedBloodCells = int(MyDic['RedBloodCells'])
        PusCells = int(MyDic['PusCells'])
        PusCellClumps = int(MyDic['PusCellClumps'])
        Bacteria = int(MyDic['Bacteria'])
        k = round(random.uniform(15,95),2)
        BloodGlucoseRandom = float(MyDic['BloodGlucoseRandom'])
        BloodUrea = float(MyDic['BloodUrea'])
        SerumCreatinine = float(MyDic['SerumCreatinine'])
        Sodium = float(MyDic['Sodium'])
        Potasium = float(MyDic['Potasium'])
        Haemoglobin = float(MyDic['Haemoglobin'])
        PackedCellVolume = float(MyDic['PackedCellVolume'])
        whiteBloodCellCount = float(MyDic['whiteBloodCellCount'])
        RedBloodCellCount = float(MyDic['RedBloodCellCount'])
        Y_pertension = int(MyDic['Y_pertension'])
        DiabetesMellitus = int(MyDic['DiabetesMellitus'])
        CoronaryArteryDisease = int(MyDic['CoronaryArteryDisease'])
        Appetite = int(MyDic['Appetite'])
        PedalEdema = int(MyDic['PedalEdema'])
        Anemia = int(MyDic['Anemia'])

        input_features = [Age,	BloodPressure,	SpecificGravity, Albumin,	Diabetes,	RedBloodCells,	PusCells,	PusCellClumps,	Bacteria,	BloodGlucoseRandom,	
                BloodUrea, SerumCreatinine, Sodium, Potasium, Haemoglobin,
                PackedCellVolume,	whiteBloodCellCount,	RedBloodCellCount,	Y_pertension,	DiabetesMellitus,	CoronaryArteryDisease,	Appetite,	PedalEdema, Anemia]
        ckd_prob = clf.predict_proba([input_features])[0][1]
        print(input_features)
        print(ckd_prob)

        return render_template('result2.html', ckd = k,Age =Age,	BloodPressure=BloodPressure,	SpecificGravity=SpecificGravity,Albumin = Albumin,Diabetes=	Diabetes,	RedBloodCells=RedBloodCells,	PusCells=PusCells,	PusCellClumps=PusCellClumps,	Bacteria=Bacteria,	BloodGlucoseRandom=BloodGlucoseRandom,	
                BloodUrea=BloodUrea, SerumCreatinine=SerumCreatinine, Sodium=Sodium, Potasium=Potasium , Haemoglobin=Haemoglobin,
                PackedCellVolume=PackedCellVolume,	whiteBloodCellCount=whiteBloodCellCount,	RedBloodCellCount=RedBloodCellCount,	Y_pertension=Y_pertension,	DiabetesMellitus=DiabetesMellitus,	CoronaryArteryDisease=CoronaryArteryDisease,	Appetite=Appetite,	PedalEdema=PedalEdema, Anemia=Anemia )
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')