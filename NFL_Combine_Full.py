
# coding: utf-8

# In[1]:


import csv

from bs4 import BeautifulSoup

from urllib.request import urlopen
import glob
import requests
import os
import pandas as pd
from scipy.spatial import distance
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#requires blank folder
mypath = "C:/Users/jerem.DESKTOP-GGM6Q2I/Documents/UNH Data Analytics/python 2/test"
#directory of "100+ touch RBs.csv"
path2 = "C:/Users/jerem.DESKTOP-GGM6Q2I/Documents/UNH Data Analytics/python 2"
os.chdir(mypath)
#this program requires the "100+ touch RBs.csv" file


# In[2]:


#code to scrape combine data
def combinescraper(year):
    response = requests.get("http://nflcombineresults.com/nflcombinedata.php?year=" + str(year) + "&pos=&college=")

# Extracting the source code of the page.
    data = response.text

# Passing the source code to Soup to create a BeautifulSoup object for it.
    soup = BeautifulSoup(data, 'lxml')
    table = soup.findAll("tr", {"class":"tablefont"})
    Year = ""
    Name = ""
    College = ""
    Pos = ""
    Height = ""
    Weight = ""
    Wonderlic = ""
    Forty_Yard = ""
    Bench_Press = ""
    Vert_Leap = ""
    Broad_Jump = ""
    Shuttle = ""
    Three_Cone = ""
    f = open("combine_data" + str(year) + ".csv", 'wb')
    write_to_file = "Year, Name, College, Pos, Height, Weight, Wonderlic, 40_Yard, Bench_Press, Vert_Leap, Broad_Jump, Shuttle, 3Cone\n"
    write_to_unicode = write_to_file.encode('utf-8')
    print(write_to_unicode)
    f.write(write_to_unicode)


    for item in soup.findAll("tr", {"class":"tablefont"}):
        row = item.findAll("div")
        if len(row) == 13:
            Year = row[0].find(text=True)
            Name = row[1].find(text=True)
            College = row[2].find(text=True)
            Pos = row[3].find(text=True)
            Height = row[4].find(text=True)
            Weight = row[5].find(text=True)
            Wonderlic = row[6].find(text=True)
            Forty_Yard = row[7].find(text=True)
            Bench_Press = row[8].find(text=True)
            Vert_Leap = row[9].find(text=True)
            Broad_Jump = row[10].find(text=True)
            Shuttle = row[11].find(text=True)
            Three_Cone = row[12].find(text=True)
            
            
            
            Name = Name.replace(",", "-")
            College = College.replace(",", "-")
            write_to_file = str(Year) + "," + Name + "," + College + "," + Pos + "," + str(Height) + "," + str(Weight) + "," + str(Wonderlic) + "," + str(Forty_Yard) + "," + str(Bench_Press) + "," + str(Vert_Leap) + "," + str(Broad_Jump) + "," + str(Shuttle) + "," + str(Three_Cone)+ "\n"
            write_to_unicode = write_to_file.encode('utf-8')
            print(write_to_unicode)
            f.write(write_to_unicode)

    f.close()


# In[3]:


#scrape combine data
for i in range(1987,2019):
    combinescraper(i)


# In[4]:


#concatenate files
import glob

path = mypath
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df= pd.read_csv(file_, index_col=None, header = 0)
    list_.append(df)
frame = pd.concat(list_)
frame.to_csv("all_combine.csv")


# In[6]:


#select columns
CombineData = pd.read_csv("C:/Users/jerem.DESKTOP-GGM6Q2I/Documents/UNH Data Analytics/python 2/test/all_combine.csv")
cols_of_interest = ['Year', ' Name', ' College', ' Pos', ' Height', ' Weight',
       ' 40_Yard', ' Vert_Leap', ' Broad_Jump',
       ' Shuttle']
CombineData = CombineData[cols_of_interest]
CombineData
CombineData.columns = ['Year', 'Name','College','Pos','Height','Weight','Forty_Yard','Vert_Leap','Broad_Jump','Shuttle']


# In[7]:


QB = ['QB']
WR = ['WR']
TE = ['TE']
RB = ['RB']
FB = ['FB']
OL = ['OT','OG','C']
P_K = ['P','K']
LBs = ['OLB','ILB']
DL = ['DE','DT']
DBs = ['CB','SS','FS']



Orig_QB_CombineData = CombineData.loc[CombineData['Pos'].isin(QB)]
Orig_WR_CombineData = CombineData.loc[CombineData['Pos'].isin(WR)]
Orig_TE_CombineData = CombineData.loc[CombineData['Pos'].isin(TE)]
Orig_RB_CombineData = CombineData.loc[CombineData['Pos'].isin(RB)]
Orig_FB_CombineData = CombineData.loc[CombineData['Pos'].isin(FB)]
Orig_OL_CombineData = CombineData.loc[CombineData['Pos'].isin(OL)]
Orig_P_K_CombineData = CombineData.loc[CombineData['Pos'].isin(P_K)]
Orig_LBs_CombineData = CombineData.loc[CombineData['Pos'].isin(LBs)]
Orig_DL_CombineData = CombineData.loc[CombineData['Pos'].isin(DL)]
Orig_DBs_CombineData = CombineData.loc[CombineData['Pos'].isin(DBs)]


# In[8]:


#clean qbs

cols_of_interest = ['Year', 'Name', 'College', 'Pos', 'Height', 'Weight',
       'Forty_Yard','Vert_Leap', 'Broad_Jump',
       'Shuttle']
QB_CombineData = Orig_QB_CombineData[cols_of_interest]

QB_CombineData = QB_CombineData[QB_CombineData.Shuttle != 9.99]
QB_CombineData = QB_CombineData[QB_CombineData.Forty_Yard != 9.99]
QB_CombineData = QB_CombineData[QB_CombineData.Vert_Leap != 'None']
QB_CombineData = QB_CombineData[QB_CombineData.Broad_Jump != 'None']

QB_CombineData
QB_CombineData.to_csv("QB_Combine.csv")
#pandas_profiling.ProfileReport(QB_CombineData)


# In[9]:


#clean rbs
RB_CombineData = Orig_RB_CombineData

RB_CombineData = RB_CombineData[RB_CombineData.Shuttle != 9.99]
RB_CombineData = RB_CombineData[RB_CombineData.Forty_Yard != 9.99]
# RB_CombineData = RB_CombineData[RB_CombineData.Bench_Press != 'None']
RB_CombineData = RB_CombineData[RB_CombineData.Vert_Leap != 'None']
RB_CombineData = RB_CombineData[RB_CombineData.Broad_Jump != 'None']


RB_CombineData2 = RB_CombineData
RB_CombineData = RB_CombineData[cols_of_interest]



RB_CombineData
RB_CombineData.to_csv("RB_Combine.csv")
#pandas_profiling.ProfileReport(RB_CombineData)


# In[10]:


#clean wrs

WR_CombineData = Orig_WR_CombineData[cols_of_interest]

WR_CombineData = WR_CombineData[WR_CombineData.Shuttle != 9.99]
WR_CombineData = WR_CombineData[WR_CombineData.Forty_Yard != 9.99]
WR_CombineData = WR_CombineData[WR_CombineData.Vert_Leap != 'None']
WR_CombineData = WR_CombineData[WR_CombineData.Broad_Jump != 'None']

WR_CombineData
WR_CombineData.to_csv("WR_Combine.csv")
#pandas_profiling.ProfileReport(WR_CombineData)


# In[11]:


#clean tes

TE_CombineData = Orig_TE_CombineData[cols_of_interest]

TE_CombineData = TE_CombineData[TE_CombineData.Shuttle != 9.99]
TE_CombineData = TE_CombineData[TE_CombineData.Forty_Yard != 9.99]
TE_CombineData = TE_CombineData[TE_CombineData.Vert_Leap != 'None']
TE_CombineData = TE_CombineData[TE_CombineData.Broad_Jump != 'None']
TE_CombineData
TE_CombineData.to_csv("TE_Combine.csv")
#pandas_profiling.ProfileReport(TE_CombineData)


# In[12]:


#clean ols

OL_CombineData = Orig_OL_CombineData[cols_of_interest]

OL_CombineData = OL_CombineData[OL_CombineData.Shuttle != 9.99]
OL_CombineData = OL_CombineData[OL_CombineData.Forty_Yard != 9.99]
OL_CombineData = OL_CombineData[OL_CombineData.Vert_Leap != 'None']
OL_CombineData = OL_CombineData[OL_CombineData.Broad_Jump != 'None']
OL_CombineData
OL_CombineData.to_csv("OL_Combine.csv")
#pandas_profiling.ProfileReport(OL_CombineData)


# In[13]:


#clean dls

DL_CombineData = Orig_DL_CombineData[cols_of_interest]

DL_CombineData = DL_CombineData[DL_CombineData.Shuttle != 9.99]
DL_CombineData = DL_CombineData[DL_CombineData.Forty_Yard != 9.99]
DL_CombineData = DL_CombineData[DL_CombineData.Vert_Leap != 'None']
DL_CombineData = DL_CombineData[DL_CombineData.Broad_Jump != 'None']
DL_CombineData
DL_CombineData.to_csv("DL_Combine.csv")
#pandas_profiling.ProfileReport(DL_CombineData)


# In[14]:


#clean lbs

LBs_CombineData = Orig_LBs_CombineData[cols_of_interest]

LBs_CombineData = LBs_CombineData[LBs_CombineData.Shuttle != 9.99]
LBs_CombineData = LBs_CombineData[LBs_CombineData.Forty_Yard != 9.99]
LBs_CombineData = LBs_CombineData[LBs_CombineData.Vert_Leap != 'None']
LBs_CombineData = LBs_CombineData[LBs_CombineData.Broad_Jump != 'None']
LBs_CombineData
LBs_CombineData.to_csv("LB_Combine.csv")
#pandas_profiling.ProfileReport(LBs_CombineData)


# In[15]:


#clean dbs

DBs_CombineData = Orig_DBs_CombineData[cols_of_interest]

DBs_CombineData = DBs_CombineData[DBs_CombineData.Shuttle != 9.99]
DBs_CombineData = DBs_CombineData[DBs_CombineData.Forty_Yard != 9.99]
DBs_CombineData = DBs_CombineData[DBs_CombineData.Vert_Leap != 'None']
DBs_CombineData = DBs_CombineData[DBs_CombineData.Broad_Jump != 'None']
DBs_CombineData
DBs_CombineData.to_csv("DB_Combine.csv")
#pandas_profiling.ProfileReport(DBs_CombineData)


# In[16]:


#print changes in size from cleaning
print('# of records for QBs:',Orig_QB_CombineData.shape,'-->',QB_CombineData.shape)
print('# of records for WRs:',Orig_WR_CombineData.shape,'-->',WR_CombineData.shape)
print('# of records for RBs:',Orig_RB_CombineData.shape,'-->',RB_CombineData.shape)
print('# of records for TEs:',Orig_TE_CombineData.shape,'-->',TE_CombineData.shape)
print('# of records for OLs:',Orig_OL_CombineData.shape,'-->',OL_CombineData.shape)
print('# of records for DLs:',Orig_DL_CombineData.shape,'-->',DL_CombineData.shape)
print('# of records for LBs:',Orig_LBs_CombineData.shape,'-->',LBs_CombineData.shape)
print('# of records for DBs:',Orig_DBs_CombineData.shape,'-->',DBs_CombineData.shape)


# In[108]:


#Now add the "100+ touch RBs.csv" file to the directory


# In[17]:


os.chdir(path2)
hundred_plus = pd.read_csv('100+ touch RBs.csv')
os.chdir(mypath)
hundred_plus = hundred_plus[['Name']]

# make a new DF of players that were rookies in 2016-2018
RB_Combine_Last_3  = RB_CombineData2[(RB_CombineData2['Year']==2016) | (RB_CombineData2['Year']==2017) | (RB_CombineData2['Year']==2018)]
RB_Combine_Not_Last_3  = RB_CombineData2[(RB_CombineData2['Year']!=2016) & (RB_CombineData2['Year']!=2017) & (RB_CombineData2['Year']!=2018)]


# # Inner join on dataset that contains only players with 100+ touches
RB_Combine_Not_Last_3 = pd.merge(RB_Combine_Not_Last_3, hundred_plus, on='Name', how='inner')

# concatinate back in last three years
frames = (RB_Combine_Not_Last_3, RB_Combine_Last_3)
Over_100_Touches = pd.concat(frames)

# # Add the recent RB's back in if they were filtered out
# RB_Combine = pd.merge(RB_Combine, RB_Combine_Not_Last_3, on='Name', how='inner')



################################################################################
# Rename Over_100_Touches -> RB_Combine so we can use the rest of the code again
################################################################################
RB_Combine2 = Over_100_Touches
RB_Combine2 = RB_Combine2.reset_index(drop=True)
RB_Combine2

# All names of RBs are unique
RB_Combine2 = RB_Combine2.drop_duplicates()

RB_Combine2 = RB_Combine2[np.isfinite(RB_Combine2['Year'])]
RB_Combine2 = RB_Combine2.reset_index(drop=True)
RB_Combine2

# Columns in DF
RB_Combine2.columns

X = pd.DataFrame(RB_Combine2['Name'])
Y = RB_Combine2[['Height', 'Weight', 'Forty_Yard', 'Vert_Leap', 'Broad_Jump', 'Shuttle']]

# Scale Data

Scaler = MinMaxScaler(feature_range=(0,1))
Y_Scaled = Scaler.fit_transform(Y)

# Y.to_csv('D:/Python/NFL Combine/Y_RB.csv')
# Xa = X.values
Ya = Y_Scaled
Xa = X.values


# In[110]:


len(Ya)


# # Run Nearest Neighbors on RBs

# In[111]:


RB_Combine2


# In[18]:


AthleticDF = pd.DataFrame(Y_Scaled)

AthleticDF = AthleticDF.drop(0,1)
AthleticDF = AthleticDF.drop(1,1)
AthleticDF.columns = ['Forty_Yard','Vert_Leap','Broad_Jump','Shuttle']
AthleticDF

RB_Names = pd.DataFrame(RB_Combine2[['Year','Name']])
AthleticDF = pd.merge(RB_Names,AthleticDF, how='left', left_index=True, right_index=True)
# AthleticDF = AthleticDF[(AthleticDF.Year == 2018)|(AthleticDF.Year == 2017)|(AthleticDF.Year == 2016)]
# AthleticDF.sort_values('Shuttle', ascending=False)


# In[19]:


#look at mean and standard deviation
print(AthleticDF['Forty_Yard'].describe())
# print(AthleticDF['Bench_Press'].describe())
print(AthleticDF['Vert_Leap'].describe())
print(AthleticDF['Broad_Jump'].describe())
print(AthleticDF['Shuttle'].describe())


# In[20]:


# Weighting Modifications

AthleticDF['Forty_Yard'] = 1-AthleticDF['Forty_Yard']
AthleticDF['Shuttle'] = 1-AthleticDF['Shuttle']


# In[21]:


#Scaling and Standardizing
AthleticDF['Forty_Yard'] = (((AthleticDF['Forty_Yard'] + (.5 - AthleticDF['Forty_Yard'].mean()))*(.5 / (3 * AthleticDF['Forty_Yard'].std()))))
AthleticDF['Forty_Yard'] = (AthleticDF['Forty_Yard'] + (.5 - AthleticDF['Forty_Yard'].mean()))
AthleticDF['Shuttle'] = (((AthleticDF['Shuttle'] + (.5 - AthleticDF['Shuttle'].mean()))*(.5 / (3 * AthleticDF['Shuttle'].std()))))
AthleticDF['Shuttle'] = (AthleticDF['Shuttle'] + (.5 - AthleticDF['Shuttle'].mean()))
# AthleticDF['Bench_Press'] = (((AthleticDF['Bench_Press'] + (.5 - AthleticDF['Bench_Press'].mean()))*(.5 / (3 * AthleticDF['Bench_Press'].std()))))
# AthleticDF['Bench_Press'] = (AthleticDF['Bench_Press'] + (.5 - AthleticDF['Bench_Press'].mean()))
AthleticDF['Vert_Leap'] = (((AthleticDF['Vert_Leap'] + (.5 - AthleticDF['Vert_Leap'].mean()))*(.5 / (3 * AthleticDF['Vert_Leap'].std()))))
AthleticDF['Vert_Leap'] = (AthleticDF['Vert_Leap'] + (.5 - AthleticDF['Vert_Leap'].mean()))
AthleticDF['Broad_Jump'] = (((AthleticDF['Broad_Jump'] + (.5 - AthleticDF['Broad_Jump'].mean()))*(.5 / (3 * AthleticDF['Broad_Jump'].std()))))
AthleticDF['Broad_Jump'] = (AthleticDF['Broad_Jump'] + (.5 - AthleticDF['Broad_Jump'].mean()))
print(AthleticDF['Forty_Yard'].describe())
# print(AthleticDF['Bench_Press'].describe())
print(AthleticDF['Vert_Leap'].describe())
print(AthleticDF['Broad_Jump'].describe())
print(AthleticDF['Shuttle'].describe())


# In[22]:


#Creating Athletic Score Weights
Speed_Weight = 0.3
Explosive_Weight = 0.2
Bench_Weight = 0.05


# AthleticDF['Athletic Score'] = Speed_Weight*AthleticDF['Forty_Yard'] + Bench_Weight*AthleticDF['Bench_Press'] + Explosive_Weight*AthleticDF['Vert_Leap'] + Explosive_Weight*AthleticDF['Broad_Jump'] + Speed_Weight*AthleticDF['Shuttle']
# AthleticDF['Athletic Score'] = (AthleticDF['Athletic Score'] - (AthleticDF['Athletic Score'].min())) * (100/(AthleticDF['Athletic Score'].max() - AthleticDF['Athletic Score'].min()))
# AthleticDF[['Year', 'Name', 'Forty_Yard', 'Bench_Press', 'Vert_Leap', 'Broad_Jump', 'Shuttle', 'Athletic Score']].sort_values('Athletic Score', ascending=False)


# In[23]:


#Creating Athletic Score for Standardized
AthleticDF['Athletic Score'] = Speed_Weight*AthleticDF['Forty_Yard'] + Explosive_Weight*AthleticDF['Vert_Leap'] + Explosive_Weight*AthleticDF['Broad_Jump'] + Speed_Weight*AthleticDF['Shuttle']
AthleticDF['Athletic Score'] = (AthleticDF['Athletic Score'] - (AthleticDF['Athletic Score'].min())) * (100/(AthleticDF['Athletic Score'].max() - AthleticDF['Athletic Score'].min()))
AthleticDF[['Year', 'Name', 'Forty_Yard', 'Vert_Leap', 'Broad_Jump', 'Shuttle', 'Athletic Score']].sort_values('Athletic Score', ascending=False)


# In[25]:


AthleticDF = pd.merge(AthleticDF,RB_Combine2, how='left', left_index=True, right_index=True, on=("Year", "Name", "Forty_Yard", "Vert_Leap", "Broad_Jump", "Shuttle"))
AthleticDF


# In[26]:


def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))

dist(Ya[1],Ya[2])

def takeSecond(elem):
    return elem[1]


# In[27]:


def Grade(num):
    if num == 0:
        return "-"
    elif num < 0.2:
        return "A"
    elif num < 0.4:
        return "B"
    elif num < 0.6:
        return "C"
    elif num < 0.8:
        return "D"
    else: 
        return "F"

def NN(person):

    Temp = []

    for itm2 in range(0,len(Ya)):
        distance = dist(Ya[[X[X.Name == person].index[0]]], Ya[[itm2]])
        Temp.append([itm2,distance])
        Temp.sort(key=takeSecond)

    Temp = Temp[0:6]

    df0 = pd.DataFrame(AthleticDF.iloc[Temp[0][0]])
    df1 = pd.DataFrame(AthleticDF.iloc[Temp[1][0]])
    df2 = pd.DataFrame(AthleticDF.iloc[Temp[2][0]])
    df3 = pd.DataFrame(AthleticDF.iloc[Temp[3][0]])
    df4 = pd.DataFrame(AthleticDF.iloc[Temp[4][0]])
    df5 = pd.DataFrame(AthleticDF.iloc[Temp[5][0]])
    
    df0 = df0.T
    df1 = df1.T
    df2 = df2.T
    df3 = df3.T
    df4 = df4.T
    df5 = df5.T
    
    frames = [df0, df1, df2, df3, df4, df5]
    
    result = pd.concat(frames)
    
    Distances_List = [Temp[0][1],Temp[1][1],Temp[2][1],Temp[3][1],Temp[4][1],Temp[5][1]]
    
    result['Eucl. Distance'] = Distances_List
    
    result['Eucl. Distance'] = result['Eucl. Distance'].round(3)
    
    result['Distance_Grade'] = result.apply(lambda x: Grade(float(x['Eucl. Distance'])), axis=1)
    
    result['Rank'] = ['-',1,2,3,4,5]
    
    result = result[['Rank','Year', 'Name', 'College', 'Pos', 'Height', 'Weight', 'Forty_Yard',
       'Vert_Leap', 'Broad_Jump', 'Shuttle', 'Eucl. Distance', 'Distance_Grade', 'Athletic Score']]
    
    return(result)


# In[28]:


NN('''Dalvin Cook''')


# In[122]:


#Creating Radar Plot


# In[29]:


Height = list()
Weight = list()
Forty_Yard = list()
Vert_Leap = list()
Broad_Jump = list()
Shuttle = list()


# In[30]:


#make sure file names are correct
#get median values
RB_Combine = pd.read_csv('RB_Combine.csv')
Z = pd.DataFrame(RB_Combine[['Height', 'Weight', 'Forty_Yard', 'Vert_Leap', 'Broad_Jump', 'Shuttle']])
Height.append(Z['Height'].median())
Weight.append(Z['Weight'].median())
Forty_Yard.append(Z['Forty_Yard'].median())
Vert_Leap.append(Z['Vert_Leap'].median())
Broad_Jump.append(Z['Broad_Jump'].median())
Shuttle.append(Z['Shuttle'].median())

WR_Combine = pd.read_csv('WR_Combine.csv')
Z = pd.DataFrame(WR_Combine[['Height', 'Weight', 'Forty_Yard', 'Vert_Leap', 'Broad_Jump', 'Shuttle']])
Height.append(Z['Height'].median())
Weight.append(Z['Weight'].median())
Forty_Yard.append(Z['Forty_Yard'].median())
Vert_Leap.append(Z['Vert_Leap'].median())
Broad_Jump.append(Z['Broad_Jump'].median())
Shuttle.append(Z['Shuttle'].median())

TE_Combine = pd.read_csv('TE_Combine.csv')
Z = pd.DataFrame(TE_Combine[['Height', 'Weight', 'Forty_Yard', 'Vert_Leap', 'Broad_Jump', 'Shuttle']])
Height.append(Z['Height'].median())
Weight.append(Z['Weight'].median())
Forty_Yard.append(Z['Forty_Yard'].median())
Vert_Leap.append(Z['Vert_Leap'].median())
Broad_Jump.append(Z['Broad_Jump'].median())
Shuttle.append(Z['Shuttle'].median())

QB_Combine = pd.read_csv('QB_Combine.csv')
Z = pd.DataFrame(QB_Combine[['Height', 'Weight', 'Forty_Yard', 'Vert_Leap', 'Broad_Jump', 'Shuttle']])
Height.append(Z['Height'].median())
Weight.append(Z['Weight'].median())
Forty_Yard.append(Z['Forty_Yard'].median())
Vert_Leap.append(Z['Vert_Leap'].median())
Broad_Jump.append(Z['Broad_Jump'].median())
Shuttle.append(Z['Shuttle'].median())

DB_Combine = pd.read_csv('DB_Combine.csv')
Z = pd.DataFrame(DB_Combine[['Height', 'Weight', 'Forty_Yard', 'Vert_Leap', 'Broad_Jump', 'Shuttle']])
Height.append(Z['Height'].median())
Weight.append(Z['Weight'].median())
Forty_Yard.append(Z['Forty_Yard'].median())
Vert_Leap.append(Z['Vert_Leap'].median())
Broad_Jump.append(Z['Broad_Jump'].median())
Shuttle.append(Z['Shuttle'].median())

RB_Combine = RB_Combine.reset_index()
QB_Combine = QB_Combine.reset_index()
DB_Combine = DB_Combine.reset_index()
WR_Combine = WR_Combine.reset_index()
TE_Combine = TE_Combine.reset_index()


# In[31]:


#scale data
Height = np.array(Height)
Weight = np.array(Weight)
Forty_Yard = np.array(Forty_Yard)
Vert_Leap = np.array(Vert_Leap)
Broad_Jump = np.array(Broad_Jump)
Shuttle = np.array(Shuttle)
Height = (Height - min(Height)) / (max(Height) - min(Height))
Weight = (Weight - min(Weight)) / (max(Weight) - min(Weight))
Forty_Yard = (Forty_Yard - min(Forty_Yard)) / (max(Forty_Yard) - min(Forty_Yard))
Vert_Leap = (Vert_Leap - min(Vert_Leap)) / (max(Vert_Leap) - min(Vert_Leap))
Broad_Jump = (Broad_Jump - min(Broad_Jump)) / (max(Broad_Jump) - min(Broad_Jump))
Shuttle = (Shuttle - min(Shuttle)) / (max(Shuttle) - min(Shuttle))
Shuttle = 1-Shuttle
Forty_Yard = 1-Forty_Yard


# In[32]:


#Creating Radar Plot
import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, LabelSet, Legend

num_vars = 6

centre = 0.5

theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
# rotate theta such that the first axis is at the top
theta += np.pi/2

def unit_poly_verts(theta, centre ):
    """Return vertices of polygon for subplot axes.
    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [centre ] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts

def radar_patch(r, theta, centre ):
    """ Returns the x and y coordinates corresponding to the magnitudes of 
    each variable displayed in the radar plot
    """
    # offset from centre of circle
    offset = 0.01
    yt = (r*centre + offset) * np.sin(theta) + centre 
    xt = (r*centre + offset) * np.cos(theta) + centre 
    return xt, yt

verts = unit_poly_verts(theta, centre)
x = [v[0] for v in verts] 
y = [v[1] for v in verts] 

p = figure(title="RB Combine Stats")
o = figure(title="WR Combine Stats")
n = figure(title="TE Combine Stats")
m = figure(title="QB Combine Stats")
l = figure(title="DB Combine Stats")
k = figure(title="Positional Combine Stats")
text = ['Height', 'Weight', 'Forty_Yard', 'Vert_Leap', 'Broad_Jump', 'Shuttle','']
source = ColumnDataSource({'x':x + [centre ],'y':y + [1],'text':text})

p.line(x="x", y="y", source=source)
o.line(x="x", y="y", source=source)
n.line(x="x", y="y", source=source)
m.line(x="x", y="y", source=source)
l.line(x="x", y="y", source=source)
k.line(x="x", y="y", source=source)

labels = LabelSet(x="x",y="y",text="text",source=source)
p.add_layout(labels)
labels = LabelSet(x="x",y="y",text="text",source=source)
o.add_layout(labels)
labels = LabelSet(x="x",y="y",text="text",source=source)
n.add_layout(labels)
labels = LabelSet(x="x",y="y",text="text",source=source)
m.add_layout(labels)
labels = LabelSet(x="x",y="y",text="text",source=source)
l.add_layout(labels)
labels = LabelSet(x="x",y="y",text="text",source=source)
k.add_layout(labels)

# example factor:
# f1 = np.array([0.88, 0.01, 0.03, 0.03, 0.00, 0.06, 0.01, 0.00, 0.00])
# f2 = np.array([0.07, 0.95, 0.04, 0.05, 0.00, 0.02, 0.01, 0.00, 0.00])
# f3 = np.array([0.01, 0.02, 0.85, 0.19, 0.05, 0.10, 0.00, 0.00, 0.00])
# f4 = np.array([0.02, 0.01, 0.07, 0.01, 0.21, 0.12, 0.98, 0.00, 0.00])
# f5 = np.array([0.01, 0.01, 0.02, 0.71, 0.74, 0.70, 0.00, 0.00, 0.00])
RB = np.array([Height[0], Weight[0], Forty_Yard[0], Vert_Leap[0], Broad_Jump[0], Shuttle[0]])
WR = np.array([Height[1], Weight[1], Forty_Yard[1], Vert_Leap[1], Broad_Jump[1], Shuttle[1]])
TE = np.array([Height[2], Weight[2], Forty_Yard[2], Vert_Leap[2], Broad_Jump[2], Shuttle[2]])
QB = np.array([Height[3], Weight[3], Forty_Yard[3], Vert_Leap[3], Broad_Jump[3], Shuttle[3]])
DB = np.array([Height[4], Weight[4], Forty_Yard[4], Vert_Leap[4], Broad_Jump[4], Shuttle[4]])
##xt = np.array(x)
# flist = [f1,f2,f3,f4,f5]
poslist = [RB, WR, TE, QB, DB]
# colors = ['blue','green','red', 'orange','purple']
colors = ['red', 'gold', 'green', 'black', 'purple']
mylegend=["RB", "WR", "TE", "QB", "DB"]
# legend = Legend(items=[(RB,["RB", width=0.9, color='red', source=data),("WR",WR),("TE",TE),("QB",QB),("DB",DB)], location=(0, -30))

# p.add_layout(legend, 'right')

xt, yt = radar_patch(RB, theta, centre)
p.patch(x=xt, y=yt, fill_alpha=0.15, fill_color='red', legend="RB")
show(p)
xt, yt = radar_patch(WR, theta, centre)
o.patch(x=xt, y=yt, fill_alpha=0.15, fill_color='gold', legend="WR")
show(o)
xt, yt = radar_patch(TE, theta, centre)
n.patch(x=xt, y=yt, fill_alpha=0.15, fill_color='green', legend="TE")
show(n)
xt, yt = radar_patch(QB, theta, centre)
m.patch(x=xt, y=yt, fill_alpha=0.15, fill_color='black', legend="QB")
show(m)
xt, yt = radar_patch(DB, theta, centre)
l.patch(x=xt, y=yt, fill_alpha=0.15, fill_color='purple', legend="DB")
show(l)
for i in range(len(poslist)):
    xt, yt = radar_patch(poslist[i], theta, centre)
    k.patch(x=xt, y=yt, fill_alpha=0.15, fill_color=colors[i], legend=mylegend[i])
show(k)


# In[33]:


#function to display radar chart for person within position
def radar(player, position):
    num_vars = 6

    centre = 0.5

    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2
    verts = unit_poly_verts(theta, centre)
    x = [v[0] for v in verts] 
    y = [v[1] for v in verts] 

    p = figure(title="Player Stats for " + player+ " compared to other " + position.upper()+"s")
    text = ['Height', 'Weight', 'Forty_Yard', 'Vert_Leap', 'Broad_Jump', 'Shuttle','']
    source = ColumnDataSource({'x':x + [centre ],'y':y + [1],'text':text})
    p.line(x="x", y="y", source=source)
    labels = LabelSet(x="x",y="y",text="text",source=source)
    p.add_layout(labels)
    #retrieves stat for player and scales between 0 and 1 based off group stats of that position
    locals()[position.upper()] = np.array([(globals()[position.upper() + "_Combine"]["Height"][globals()[position.upper() + "_Combine"].loc[globals()[position.upper() + "_Combine"]["Name"] == player].iloc[0][0]] - globals()[position.upper() + "_Combine"]['Height'].min())/(globals()[position.upper() + "_Combine"]["Height"].max() - globals()[position.upper() + "_Combine"]["Height"].min()),         (globals()[position.upper() + "_Combine"]["Weight"][globals()[position.upper() + "_Combine"].loc[globals()[position.upper() + "_Combine"]["Name"] == player].iloc[0][0]] - globals()[position.upper() + "_Combine"]['Weight'].min())/(globals()[position.upper() + "_Combine"]["Weight"].max() - globals()[position.upper() + "_Combine"]["Weight"].min()),         1 - (globals()[position.upper() + "_Combine"]["Forty_Yard"][globals()[position.upper() + "_Combine"].loc[globals()[position.upper() + "_Combine"]["Name"] == player].iloc[0][0]] - globals()[position.upper() + "_Combine"]['Forty_Yard'].min())/(globals()[position.upper() + "_Combine"]["Forty_Yard"].max() - globals()[position.upper() + "_Combine"]["Forty_Yard"].min()),         (globals()[position.upper() + "_Combine"]["Vert_Leap"][globals()[position.upper() + "_Combine"].loc[globals()[position.upper() + "_Combine"]["Name"] == player].iloc[0][0]] - globals()[position.upper() + "_Combine"]['Vert_Leap'].min())/(globals()[position.upper() + "_Combine"]["Vert_Leap"].max() - globals()[position.upper() + "_Combine"]["Vert_Leap"].min()),         (globals()[position.upper() + "_Combine"]["Broad_Jump"][globals()[position.upper() + "_Combine"].loc[globals()[position.upper() + "_Combine"]["Name"] == player].iloc[0][0]] - globals()[position.upper() + "_Combine"]['Broad_Jump'].min())/(globals()[position.upper() + "_Combine"]["Broad_Jump"].max() - globals()[position.upper() + "_Combine"]["Broad_Jump"].min()),         1 - (globals()[position.upper() + "_Combine"]["Shuttle"][globals()[position.upper() + "_Combine"].loc[globals()[position.upper() + "_Combine"]["Name"] == player].iloc[0][0]] - globals()[position.upper() + "_Combine"]['Shuttle'].min())/(globals()[position.upper() + "_Combine"]["Shuttle"].max() - globals()[position.upper() + "_Combine"]["Shuttle"].min())])
    xt, yt = radar_patch(locals()[position.upper()], theta, centre)
    p.patch(x=xt, y=yt, fill_alpha=0.15, fill_color='red', legend=position.upper())
    show(p)


# # Radar Plot for Individual

# In[34]:


radar("Zac Stacy", 'rb')

