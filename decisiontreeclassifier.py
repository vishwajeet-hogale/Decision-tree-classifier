'''
Assume df is a pandas dataframe object of the dataset given
'''
import numpy as np
import pandas as pd
import random
from numpy import log2
'''Calculate the entropy of the enitre dataset'''
	#input:pandas_dataframe
	#output:int/float/double/large

def get_entropy_of_dataset(df):
	entropy = 0
	output_var = df.columns.tolist()[-1]
	unique_categories = df[output_var].unique()
	j = df[output_var].value_counts()
	for value in unique_categories:
		probab = j[value]/len(df[output_var])
		entropy+=-probab*log2(probab)
	return entropy



'''Return entropy of the attribute provided as parameter'''
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large
def get_entropy_of_attribute(df,attribute):
	entropy_of_attribute = 0
	df1 = df.groupby(attribute)

	for j in df1:
		total_of_p_n = len(df[attribute])
		# print(total_of_p_n)
		probab = get_entropy_of_dataset(j[1]) 
		entropy_of_attribute+= -(sum(j[1][j[1].keys()[-1]].value_counts())/total_of_p_n)*probab
	return abs(entropy_of_attribute)



'''Return Information Gain of the attribute provided as parameter'''
	#input:pandas_dataframe,str
	#output:int/float/double/large
def get_information_gain(df,attribute):
	# information_gain = 0
	ent_attr = get_entropy_of_attribute(df,attribute)
	entr = get_entropy_of_dataset(df)
	return entr-ent_attr



''' Returns Attribute with highest info gain'''  
	#input: pandas_dataframe
	#output: ({dict},'str')     
def get_selected_attribute(df):
   
	information_gains={}
	max1=0
	selected_column=''
	for i in list(df.columns[:-1]):
		information_gains[i] = get_information_gain(df,i)
		if(get_information_gain(df,i)>max1):
			max1 = get_information_gain(df,i)
			selected_column = i
	# print(information_gains)
	

	'''
	Return a tuple with the first element as a dictionary which has IG of all columns 
	and the second element as a string with the name of the column selected

	example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
	'''

	return (information_gains,selected_column)


'''
------- TEST CASES --------
How to run sample test cases ?

Simply run the file DT_SampleTestCase.py
Follow convention and do not change any file / function names

'''
