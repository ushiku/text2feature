# -*- coding: utf-8 -*-

def text2dep(input_f, kytea_model=None, eda_model=""):

    import subprocess
    import sys
    import time
    
    #kytea,edaのインストールしているかの確認
    #以下の2つのコマンドだと終了ステータスが0ではないのでエラーが返ってくる
    #終了ステータス1のときは成功としたが、環境によって異なるのかは分からん
    try:
        output_test = subprocess.check_output('kytea --help', shell=True)
    except subprocess.CalledProcessError as e:
        returncode =  e.returncode
        if returncode == 1:
            pass
        else:
            print ('"kytea --help"のコマンドが使えません')
            print ('正しくインストールされているか確認してください')
            print ('http://www.phontron.com/kytea/index-ja.html')
            sys.exit(1)

    try:
        output_test = subprocess.check_output('eda', shell=True)
    except subprocess.CalledProcessError as e:
        returncode = e.returncode
        if returncode == 1:
            pass
        else:
            print ('"eda"のコマンドが使えません')
            print ('正しくインストールされているか確認してください')
            print ('http://www.ar.media.kyoto-u.ac.jp/tool/EDA/')
            sys.exit(1)

        

    #kytea,edaの出力ファイル名の定義とコマンド定義
    file_kytea = input_f + '.kytea'
    file_eda = input_f + '.eda'
    if kytea_model is None:
        cmd_kytea = 'kytea < ' + input_f + ' > ' + file_kytea
    else:
        cmd_kytea = 'kytea -model  ' + kytea_model + ' < ' + input_f + ' > ' + file_kytea

    cmd_eda = 'eda -m ' + eda_model + '  -i kytea -o eda < ' + file_kytea

    #kytea実行部分
    start = time.time()
    print (cmd_kytea)
    try:
        output_kytea = subprocess.check_output(cmd_kytea, shell=True)
    except subprocess.CalledProcessError as e:
        returncode = e.returncode
        print ('kyteaがエラーになりました')
        print ('kyteaの終了ステータス=' + str(returncode))
        sys.exit(1)
        
        
    print ('kytea_done')

    #eda実行部分
    print (cmd_eda)
    try:
        output_eda = subprocess.check_output(cmd_eda, shell=True)
    except subprocess.CalledProcessError as e:
        returncode = e.returncode
        print ('edaがエラーになりました')
        print ('edaの終了ステータス=' + str(returncode))
        sys.exit(1)
        
    
    elapsed_time = time.time() - start
    print (("elapsed_time:{0}".format(elapsed_time)) + "[sec]")


    output_eda_str = output_eda.decode('utf-8')
    #edaの出力をファイルに保存
    f = open(file_eda, 'w')
    f.write(output_eda_str)
    f.close
    print(type(output_eda_str))
    eda = output_eda_str.split('\n')
    eda.pop(-1)
    eda.pop(-1)

    return (eda)
