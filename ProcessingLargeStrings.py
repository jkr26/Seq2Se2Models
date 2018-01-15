#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import psycopg2
import psycopg2.extras
"""
Created on Sun Jan 14 16:23:12 2018

@author: jkr
"""


def write_to_db(title_and_summary_list):
    cnxn = psycopg2.connect("host='localhost' dbname='TextSummary' user='PythonExecutor' password='2Cons!stent'")
    cursor = cnxn.cursor()
    try:
        psycopg2.extras.execute_batch(cursor, """
                       INSERT INTO SummaryDescriptionPairs (TitleAndSummary, LongDescription, CorpusName)
                       VALUES (%s, %s, %s);""", 
                       title_and_summary_list)
    except:
        cursor.close()
        cnxn.close()
        return "Failed!"
    cnxn.commit()
    cursor.close()
    cnxn.close()
    return "Passed"

def parse_tab_delimited(file, corpus, batch_size):
    i = 0
    j = 0
    string_list  = ''
    title_and_summary = ''
    long_description = ''
    title_and_summary_list=[]
    with open(file) as f:
        for line in f:
            if "|" in line:
                j+=1
                long_description = string_list
                string_list=''
                title_and_summary_list.append((title_and_summary, long_description, corpus))
                i=0
                if j % batch_size == 0 and j>0 :
                    print(j)
                    try_to_write = write_to_db(title_and_summary_list)
                    title_and_summary_list=[]
                    if try_to_write == "Failed!":
                        print(try_to_write)
                        raise TypeError()
                    
            else:
                if i == 3:
                    title_and_summary = string_list
                    string_list = ''
                i+=1
                string_list = string_list+line
                

if __name__=="__main__":    
    file = "/home/jkr/Documents/MLData/NLPCorpora/WestburyLab.Wikipedia.Corpus.txt"
    cnxn = psycopg2.connect("host='localhost' dbname='TextSummary' user='PythonExecutor' password='2Cons!stent'")
    
    parse_tab_delimited(file, "Wikipedia2010Corpus", 1000000)