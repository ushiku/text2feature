#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import re
import shutil
import subprocess

class Text2dep:

    #初期化とKyTea,EDAのコマンドがインストールされているか確認
    def __init__(self):
        #kytea,edaのインストールしているかの確認 
        #以下の2つのコマンドだと終了ステータスが0ではないのでエラーが返ってくる
        #終了ステータス1のときは成功としたが、環境によって異なるのかは分からん
        if shutil.which('kytea'):
            pass
        else:
            print ('"kytea"のコマンドが使えません')
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

    #文字列用
    def kytea_str(self, input_str, kytea_model=None):

        #コマンドの定義、モデルを何にするかを決定
        if kytea_model is None:
            cmd_kytea = 'kytea'
        else:
            cmd_kytea = 'kytea -model ' + kytea_model
        input_str = input_str + '\n'
        process_kytea = subprocess.Popen(cmd_kytea.strip().split(" "), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        output_kytea = process_kytea.communicate(input_str.encode('utf-8'))[0]
        kytea_line_list = output_kytea.decode('utf-8').split("\n")
        return kytea_line_list[0]


    #KyTeaコマンド実行部分
    def kytea(self, input_f, kytea_model=None, pipe_eda=False):        

        #コマンドの定義、モデルを何にするかを決定
        if kytea_model is None:
            cmd_kytea = 'kytea'
        else:
            cmd_kytea = 'kytea -model ' + kytea_model

        #複数ファイルをつなげる。ファイル終端にはEOFを追加
        input = ""
        for file_iter in input_f:
            input = input + open(file_iter, 'r').read()
            input = input + 'EOF\n'

        #kytea実行部分
        process_kytea = subprocess.Popen(cmd_kytea.strip().split(" "), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        output_kytea = process_kytea.communicate(input.encode('utf-8'))[0]
        #False ならEOFごとに区切って、UTF-8にしてリスト型で返す
        #Ture ならEDAに渡すためにそのまま返す
        if pipe_eda == False:
            #kyteaの出力を整形、EOFで区切ってlistに追加するために
            #kyteaのEOFの出力を取得したり、それで分割したり、文字コード考慮したり
            kytea_line_list = output_kytea.decode('utf-8').split("\n")
            EOF_kytea = kytea_line_list[output_kytea.decode('utf-8').count('\n') - 1]

            EOF_kytea = EOF_kytea + '\n'
            output_kytea_list = output_kytea.split(EOF_kytea.encode('utf-8'))
            for i in range(len(output_kytea_list)):
                output_kytea_list[i] = output_kytea_list[i].decode('utf-8')

            output_kytea_list.pop()
            return output_kytea_list
        else:
            return output_kytea

    #EDAコマンド実行部分
    def eda(self, input_kytea, eda_model='', pipe_kytea=False):
        import subprocess
        import sys
        
        #コマンドの定義、モデルを何にするかを決定
        cmd_eda = 'eda -m ' + eda_model + '  -i kytea'

        #eda実行部分
        process_eda = subprocess.Popen(cmd_eda.strip().split(" "), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        if pipe_kytea == False:
            input = ''
            for line in input_kytea:
                input += line
                input += 'EOF/名詞/いーおーえふ\n'
                
            output_eda = process_eda.communicate(input.encode('utf-8'))[0]
        else:
            output_eda = process_eda.communicate(input_kytea)[0]

        #EDAの出力を整形、EOFで区切ってlistに追加するために 
        #EDAのEOFの出力を取得したり、それで分割したり、文字コード考慮したり
        eda_line_list = output_eda.decode('utf-8').split("\n")
        EOF_eda = eda_line_list[output_eda.decode('utf-8').count('\n') - 2]

        EOF_eda = EOF_eda + '\n'
        output_eda_list = output_eda.split(EOF_eda.encode('utf-8'))
        for i in range(len(output_eda_list)):
            output_eda_list[i] = output_eda_list[i].decode('utf-8')

        output_eda_list.pop()
        
        new_article = []
        new_output = []
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
        return new_output

    # KyTeaの出力をEDAに渡しただけ
    def t2f(self, input_f, kytea_model=None, eda_model=''):
        return self.eda(self.kytea(input_f, kytea_model, pipe_eda=True), eda_model, pipe_kytea=True)

    
    @classmethod
    def load_eda(self, eda_file_path_list):
        '''
        edaの出力結果のファイルパスから、eda形式にする。
        '''
        output, article, units = [], [], []
        for eda_file_path in eda_file_path_list:
            for unit in open(eda_file_path, 'r'):
                unit = unit.strip()
                if re.match('ID', unit):
                    continue
                if unit == '':
                    article.append(units)
                    units = []
                    continue
                units.append(unit)
            output.append(article)
            article = []
        return output
