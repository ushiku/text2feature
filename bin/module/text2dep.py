#!/usr/bin/env python
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
            
    start = time.time()


    #コマンドの定義、モデルを何にするかを決定
    if kytea_model is None:
        cmd_kytea = 'kytea'
    else:
        cmd_kytea = 'kytea -model ' + kytea_model
    cmd_eda = 'eda -m ' + eda_model + '  -i kytea'


    #複数ファイルをつなげる。ファイル終端にはEOFを追加
    #ファイルの中身にEOFがあったら困る
    input = ""
    for file_iter in input_f:
        input = input + open(file_iter, 'r').read()
        input = input + 'EOF\n'


    #kytea実行部分
    process_kytea = subprocess.Popen(cmd_kytea.strip().split(" "), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    output_kytea = process_kytea.communicate(input.encode('utf-8'))[0]
    print ('kytea_done')
    print(output_kytea.decode('utf-8'))

    #eda実行部分
    process_eda = subprocess.Popen(cmd_eda.strip().split(" "), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    output_eda = process_eda.communicate(output_kytea)[0]

    #EDAの出力を整形、EOFで区切ってlistに追加するために
    #EDAのEOFの出力を取得したり、それで分割したり、文字コード考慮したり
    eda_line_list = output_eda.decode('utf-8').split("\n")
    EOF_eda = eda_line_list[output_eda.decode('utf-8').count('\n') - 2]
        
    EOF_eda = EOF_eda + '\n'
    output_eda_list = output_eda.split(EOF_eda.encode('utf-8'))
    for i in range(len(output_eda_list) - 1):
        output_eda_list[i] = output_eda_list[i].decode('utf-8')

    #かかった時間を表示、ここは不必要と言われるかも
    elapsed_time = time.time() - start
    print (("elapsed_time:{0}".format(elapsed_time)) + "[sec]")
    
    new_article = []
    new_output = []
    print(output_eda_list)
    for article in output_eda_list:
        if article == b'\n':
            continue
        sentences = article.split('ID=')
        sentences.pop(0)
        for sentence in sentences:
            units = sentence.split('\n')
            units.pop(-1)
            units.pop(-1)
            try:
                units.pop(0)
                new_article.append(units)
            except:
                pass
        new_output.append(new_article)
        new_article = []

    return new_output, output_eda_list
