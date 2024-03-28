#まず必要なライブラリ、モジュールをインポートします
import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
import numpy as np

#次に、分類したいクラス名をclassesリストに格納しておきましょう。今回は数字を分類するので0~9としておきます。
#image_sizeには学習に用いた画像のサイズを渡しておきます。今回はMNISTのデータセットを用いたので28とします。
classes = ["0","1","2","3","4","5","6","7","8","9"]
image_size = 28


#UPLOAD_FOLDERにはアップロードされた画像を保存するフォルダ名を渡しておきます。
#ALLOWED_EXTENSIONSにはアップロードを許可する拡張子を指定します。
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


#Flaskクラスのインスタンスを作成します。
app = Flask(__name__)


#アップロードされたファイルの拡張子のチェックをする関数を定義します。andの前後の2つの条件を満たすときにTrueを返します
#一つ目の条件'.' in filenameは、変数filenameの中に.という文字が存在するかどうか
#二つ目の条件filename.rsplit(...)は、変数filenameの.より後ろの文字列がALLOWED_EXTENSIONSのどれかに該当するかどうか
#rsplit()は基本的には split()と同じです。
#しかし、split()は区切る順序は文字列の最初からでしたが、rsplitは区切る順序が文字列の最後からになります。lower()は文字列を小文字に変換します
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#学習済みモデルをロード
model = load_model('./model.h5')


#トップページにアクセスしたときに実行される関数を定義
#@で始まる行はデコレータといって、その次の行で定義する関数やクラスに対して何らかの処理を行います。
#app.route()は、次の行で定義される関数を指定した URL に対応づけるという処理をしています
#app.route()の引数にはURL以降のURLパスを指定。すなわち、URLにアクセスした時に呼び出される関数。
#GETやPOSTはHTTPメソッドの一種(GET はページにアクセスしたときにhtmlファイルを取り込むこと、POST はデータをサーバーへ送信することを表す)
@app.route('/', methods=['GET', 'POST'])
def upload_file():

    #requestはウェブ上のフォームから送信したデータを扱うための関数
    #request.method == 'POST'であるとき、これから後に続くコードが実行
    if request.method == 'POST':

        #POSTリクエストにファイルデータが含まれているか、また、ファイルにファイル名があるかをチェック
        #redirect()は引数に与えられたurlに移動する関数であり、request.urlにはリクエストがなされたページのURLが格納
        #アップロードされたファイルがない、もしくはファイル名がない場合は元のページに戻ります。
        if 'file' not in request.files: 
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        
        #そして、アップロードされたファイルの拡張子をチェックします
        #そのあと、secure_filename()でファイル名に危険な文字列がある場合に無効化（サニタイズ）します
        #次にos.path.join()で引数に与えられたパスをosに応じて結合(Windowsでは￥で結合し、Mac,Linaxでは/で結合する) 
        #そのパスにアップロードされた画像を保存
        #また、その保存先をfilepathに格納
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)


            #いよいよアップロードされた画像の数字を判別する。

            #受け取った画像を読み込み、np形式に変換
            #kerasのimage.load_imgという画像のロードとリサイズを同時にできる関数を用いる
            #引数には読み込みたい画像のURLと、その画像をリサイズしたいサイズを指定
            #さらに grayscale=Trueと渡すことで、モノクロで読み込む
            img = image.load_img(filepath, color_mode="grayscale", target_size=(image_size,image_size))
            img = image.img_to_array(img) #引数に与えられた画像をNumpy配列に変換
            data = np.array([img]) #model.predict()にはNumpy配列のリストを渡す必要がある為、imgをリストとして渡す。

            #変換したデータをモデルに渡して予測する
            result = model.predict(data)[0]
            predicted = result.argmax()
            pred_answer = "これは " + classes[predicted] + " です"

            #render_templateの引数にanswer=pred_answerと渡すことで、index.htmlに書いたanswerにpred_answerを代入する
            return render_template("index.html",answer=pred_answer)

    #POSTリクエストがなされないとき（単にURLにアクセスしたとき）にはindex.htmlのanswerには何も表示しない。
    return render_template("index.html",answer="")


#__name__ == '__main__'がTrueである、すなわちこのコードが直接実行されたときのみapp.run()が実行され、Flaskアプリが起動
#webアプリをデプロイする際には、サーバーを外部からも利用可能にするためにhost='0.0.0.0'と指定
#port = int(os.environ.get('PORT', 8080))では、Renderで使えるポート番号を取得してportに格納しています。設定されていなければ8080が格納
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)