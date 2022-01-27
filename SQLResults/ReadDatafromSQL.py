import pandas as pd
import pyodbc

from Configurations import Configurations

def getTableQueryTecniques(Table, skinGroup, participant_number, differenceHR, techids):
    ##Parameters
    param = "declare @skinGroup as varchar(150), @AlgoType as varchar(100), @PIID  as varchar(150), @DifferenceHR as int, @NegDifferenceHR as int," \
            "@PreProcess as int, @FFT as varchar(20), @Filter as int , @Result as int, @Smoothen as bit " \
            "set @skinGroup = '"+skinGroup+"' " \
            "set @PIID =  '"+participant_number+"' " \
            "set @DifferenceHR = "+differenceHR+" " \
            "set @NegDifferenceHR = -1 * @DifferenceHR"
    listId = str(techids)
    listId = listId.replace('[','')
    listId = listId.replace(']','')

    QuerySelect = "select"
    QueryColumns = "TechniqueId,DifferenceHR, Techniques.AlgorithmType, Techniques.PreProcess, Techniques.FFT, Techniques.Filter, Techniques.Result,Techniques.Smoothen"
    QueryTable = "from " + Table + " join Techniques on Techniques.Id = " + Table + ".TechniqueId join Participants on Participants.ParticipantId = " + Table + ".ParticipantId"
    QueryCondition = "where  DifferenceHR <@DifferenceHR and DifferenceHR>@NegDifferenceHR and SkinGroup =@skinGroup and Participants.ParticipantId  = @PIID and TechniqueId in (" +\
                      listId + ")"
    FullQuery = param + " " + QuerySelect + " " + QueryColumns + " " + QueryTable + " " + QueryCondition
    return FullQuery


def getTableQuery(Table, skinGroup,AlgoType, participant_number, differenceHR):
    ##Parameters
    param = "declare @skinGroup as varchar(150), @AlgoType as varchar(100), @PIID  as varchar(150), @DifferenceHR as int, @NegDifferenceHR as int," \
            "@PreProcess as int, @FFT as varchar(20), @Filter as int , @Result as int, @Smoothen as bit " \
            "set @skinGroup = '"+skinGroup+"' " \
            "set @PIID =  '"+participant_number+"' " \
            "set @AlgoType = '"+AlgoType+"' " \
            "set @DifferenceHR = "+differenceHR+" " \
            "set @NegDifferenceHR = -1 * @DifferenceHR"

    QuerySelect = "select"
    QueryColumns = "TechniqueId,DifferenceHR, Techniques.AlgorithmType, Techniques.PreProcess, Techniques.FFT, Techniques.Filter, Techniques.Result,Techniques.Smoothen"
    QueryTable = "from " + Table + " join Techniques on Techniques.Id = " + Table + ".TechniqueId join Participants on Participants.ParticipantId = " + Table + ".ParticipantId"
    QueryCondition = "where  DifferenceHR <@DifferenceHR and DifferenceHR>@NegDifferenceHR and SkinGroup =@skinGroup and Participants.ParticipantId  = @PIID and AlgorithmType = @AlgoType"
    FullQuery = param + " " + QuerySelect + " " + QueryColumns + " " + QueryTable + " " + QueryCondition
    return FullQuery

# Defining the connection string
conn = pyodbc.connect('''DRIVER={SQL Server}; Server=ELCIELO; 
                        UID=sa; PWD=A@1234567; DataBase=ARPOS''')

# Fetching the data from the selected table using SQL query
# RawData = pd.read_sql_query('''select * from [Participants]''', conn)
differenceHR = str(10)

skinGroup="Europe_WhiteSkin_Group"
objConfig = Configurations(skinGroup=skinGroup)

TechniqueIdsResting1ResultsInitial = []
TechniqueIdsResting1RevisedResults = []
TechniqueIdsResting1UpSampledResults = []

for participant_number in objConfig.ParticipantNumbers:
    ###Original Tbale
    AlgoType="FastICA"

    Table="Resting1Results"
    FullQuery = getTableQuery(Table, skinGroup,AlgoType, participant_number, differenceHR)
    Resting1ResultsInitial = pd.read_sql_query(FullQuery,conn)
    Resting1ResultsInitial.head()

    count=1
    for column in Resting1ResultsInitial.iterrows():
        TechniqueIdsResting1ResultsInitial.append(column[count].TechniqueId)

    Table="Resting1RevisedResults"
    FullQuery = getTableQuery(Table, skinGroup,AlgoType, participant_number, differenceHR)
    Resting1RevisedResults = pd.read_sql_query(FullQuery,conn)
    Resting1RevisedResults.head()

    count=1
    for column in Resting1RevisedResults.iterrows():
        TechniqueIdsResting1RevisedResults.append(column[count].TechniqueId)

    Table="Resting1UpSampledResults"
    FullQuery = getTableQuery(Table, skinGroup,AlgoType, participant_number, differenceHR)
    Resting1UpSampledResults= pd.read_sql_query(FullQuery,conn)
    Resting1UpSampledResults.head()

    count=1
    for column in Resting1UpSampledResults.iterrows():
        TechniqueIdsResting1UpSampledResults.append(column[count].TechniqueId)

    # Participants_TechniqueIdsResting1ResultsInitial[participant_number] = TechniqueIdsResting1ResultsInitial
    # Participants_TechniqueIdsResting1RevisedResults[participant_number] = TechniqueIdsResting1RevisedResults
    # Participants_TechniqueIdsResting1UpSampledResults[participant_number] =TechniqueIdsResting1UpSampledResults
    # CommonFiles = list(set(FileName1) & set(FileName2))

unique_TechniqueIdsResting1ResultsInitial = list(set(TechniqueIdsResting1ResultsInitial))

unique_TechniqueIdsResting1RevisedResults = list(set(TechniqueIdsResting1RevisedResults))

unique_TechniqueIdsResting1UpSampledResults = list(set(TechniqueIdsResting1UpSampledResults))

CommonTechniqueAmongAll = list(set(unique_TechniqueIdsResting1ResultsInitial)
                               & set(unique_TechniqueIdsResting1RevisedResults)
                               & set(unique_TechniqueIdsResting1UpSampledResults))
# print(CommonTechniqueAmongAll)
# for item in CommonTechniqueAmongAll:
#     print(str(item) +',' )

Participants_TechniqueIdsResting1ResultsInitial = {}
Participants_TechniqueIdsResting1RevisedResults = {}
Participants_TechniqueIdsResting1UpSampledResults = {}

for participant_number in objConfig.ParticipantNumbers:
    # differenceHR = str(5)
    # if(participant_number == "PIS-6327" or participant_number == "PIS-9219" or participant_number == "PIS-7728" or participant_number == "PIS-8308" or  participant_number == "PIS-4014"
    # or participant_number == "PIS-8343"):
    #     differenceHR = str(20)
    TechniqueResultId = []
    Table= "Resting1RevisedResults"#Resting1UpSampledResults"Resting1RevisedResults" #"Resting1Results"
    FullQuery = getTableQueryTecniques(Table, skinGroup, participant_number, differenceHR,CommonTechniqueAmongAll)
    rs1 = pd.read_sql_query(FullQuery,conn)
    rs1.head()

    count=1
    for column in rs1.iterrows():
        TechniqueResultId.append(column[count].TechniqueId)

    Participants_TechniqueIdsResting1RevisedResults[participant_number] = TechniqueResultId

    TechniqueResultId2 = []
    Table = "Resting1UpSampledResults"  # Resting1UpSampledResults"Resting1RevisedResults" #"Resting1Results"
    FullQuery2 = getTableQueryTecniques(Table, skinGroup, participant_number, differenceHR, CommonTechniqueAmongAll)
    rs2 = pd.read_sql_query(FullQuery2, conn)
    rs2.head()

    count=1
    for column in rs2.iterrows():
        TechniqueResultId2.append(column[count].TechniqueId)

    Participants_TechniqueIdsResting1UpSampledResults[participant_number] = TechniqueResultId2
    # commonRESULTS= (list(set(TechniqueResultId) & set(TechniqueResultId2)))

    # istrue = TechniqueResultId.__contains__(7642)
    # istrue4 = TechniqueResultId2.__contains__(7642)

    # print(participant_number)
    #
    # if(len(TechniqueResultId) == len(TechniqueResultId2) ):
    #     Participants_TechniqueIdsResting1ResultsInitial[participant_number] = TechniqueResultId
    # else:
    #     if(len(TechniqueResultId)> len(TechniqueResultId2)):
    #         Participants_TechniqueIdsResting1ResultsInitial[participant_number] = TechniqueResultId
    #     else:
    #         Participants_TechniqueIdsResting1ResultsInitial[participant_number] = TechniqueResultId2

print('---------------------------RevisedResults-----------------------------------')
PIS8073 = Participants_TechniqueIdsResting1RevisedResults.get("PIS-8073")
PIS2047 = Participants_TechniqueIdsResting1RevisedResults.get("PIS-2047")
PIS4014 = Participants_TechniqueIdsResting1RevisedResults.get("PIS-4014")
PIS1949 = Participants_TechniqueIdsResting1RevisedResults.get("PIS-1949")
PIS3186 = Participants_TechniqueIdsResting1RevisedResults.get("PIS-3186")
PIS7381 = Participants_TechniqueIdsResting1RevisedResults.get("PIS-7381")
PIS4709 = Participants_TechniqueIdsResting1RevisedResults.get("PIS-4709")
PIS6729 = Participants_TechniqueIdsResting1RevisedResults.get("PIS-6729")
PIS6327 = Participants_TechniqueIdsResting1RevisedResults.get("PIS-6327")
PIS6888 = Participants_TechniqueIdsResting1RevisedResults.get("PIS-6888")
PIS9219 = Participants_TechniqueIdsResting1RevisedResults.get("PIS-9219")
PIS2212 = Participants_TechniqueIdsResting1RevisedResults.get("PIS-2212")
PIS5868 = Participants_TechniqueIdsResting1RevisedResults.get("PIS-5868")
PIS3252 = Participants_TechniqueIdsResting1RevisedResults.get("PIS-3252")
PIS4497 = Participants_TechniqueIdsResting1RevisedResults.get("PIS-4497")
PIS8343 = Participants_TechniqueIdsResting1RevisedResults.get("PIS-8343")
PIS7728 = Participants_TechniqueIdsResting1RevisedResults.get("PIS-7728")
PIS8308 = Participants_TechniqueIdsResting1RevisedResults.get("PIS-8308")
PIS8308P2 = Participants_TechniqueIdsResting1RevisedResults.get("PIS-8308P2")
PIS5868P2 = Participants_TechniqueIdsResting1RevisedResults.get("PIS-5868P2")
PIS3252P2 = Participants_TechniqueIdsResting1RevisedResults.get("PIS-3252P2")

# FPS15 = list( set(PIS6729)&set(PIS6888) &set(PIS5868)&set(PIS3186))
# FPS15_2 = list(set(PIS3252))
# print('FPS15')
# print(FPS15)
# print('FPS15_2')
# print(FPS15_2)
#
# FPS30_1 = list( set(PIS1949) &set(PIS7381)&set(PIS2212)&set(PIS4497)&set(PIS8308P2)&set(PIS5868P2)&set(PIS3252P2))
# FPS30_2 = list( set(PIS2047) &set(PIS8308) &set(PIS8343))
# FPS30_3 =PIS9219
# print('FPS30_1')
# print(FPS30_1)
# print('FPS30_2')
# print(FPS30_2)
# print('PIS9219')
# print(PIS9219)
#
# FPS15andFPS30Common =list(set(FPS15_2)&set(FPS30_2))
# print('FPS15andFPS30Common')
# print(FPS15andFPS30Common)
#
# Remaining1 =  list( set(PIS4014)   & set(PIS4709) & set(PIS8073))
# Remaining2 =PIS6327
# Remaining3 =  PIS7728
# print('Remaining1')
# print(Remaining1)
# print('Remaining2')
# print(Remaining2)
# print('Remaining3')
# print(Remaining3)
#
# a=0
# Remaining3 =  list( set(PIS8073) &set(PIS4014)  & set(PIS4709) & set(PIS6327) &set(PIS7728))
#
# FPS30 = list( set(PIS2047)&set(PIS1949) &set(PIS7381)&set(PIS2212)&set(PIS9219)&set(PIS4497)
#               &set(PIS8343)&set(PIS8308)&set(PIS8308P2)&set(PIS5868P2)&set(PIS3252P2))
CommonTechniqueAmongAllOriginal = list(   set(PIS1949 ) &
                                         set(PIS7381 ) & set(PIS2212 )& set(PIS5868 )&set(PIS8308P2 ) &
                                         set(PIS3252P2 )&set(PIS5868P2 ) & set(PIS4497 ) & set(PIS8073)& set(PIS6729 )& set(PIS6888 )& set(PIS4709 ) )
print('CommonTechniqueAmongAllOriginal')
print(CommonTechniqueAmongAllOriginal)

allodd1 = list(  set(PIS3186 ) & set(PIS7728 ))
allodd2 = list(set(PIS4014 )& set(PIS2047) & set(PIS8308 ))
allodd3 = list(   set(PIS3252 )& set(PIS8343 ) )

print('allodd1')
print(allodd1)
print('allodd2')
print(allodd2)
print('allodd3')
print(allodd3)
print('PIS6327')
print(PIS6327 )
print('PIS9219')
print(PIS9219 )


print('---------------------------UpSampledResults-----------------------------------')
PIS8073 = Participants_TechniqueIdsResting1UpSampledResults.get("PIS-8073")
PIS2047 = Participants_TechniqueIdsResting1UpSampledResults.get("PIS-2047")
PIS4014 = Participants_TechniqueIdsResting1UpSampledResults.get("PIS-4014")
PIS1949 = Participants_TechniqueIdsResting1UpSampledResults.get("PIS-1949")
PIS3186 = Participants_TechniqueIdsResting1UpSampledResults.get("PIS-3186")
PIS7381 = Participants_TechniqueIdsResting1UpSampledResults.get("PIS-7381")
PIS4709 = Participants_TechniqueIdsResting1UpSampledResults.get("PIS-4709")
PIS6729 = Participants_TechniqueIdsResting1UpSampledResults.get("PIS-6729")
PIS6327 = Participants_TechniqueIdsResting1UpSampledResults.get("PIS-6327")
PIS6888 = Participants_TechniqueIdsResting1UpSampledResults.get("PIS-6888")
PIS9219 = Participants_TechniqueIdsResting1UpSampledResults.get("PIS-9219")
PIS2212 = Participants_TechniqueIdsResting1UpSampledResults.get("PIS-2212")
PIS5868 = Participants_TechniqueIdsResting1UpSampledResults.get("PIS-5868")
PIS3252 = Participants_TechniqueIdsResting1UpSampledResults.get("PIS-3252")
PIS4497 = Participants_TechniqueIdsResting1UpSampledResults.get("PIS-4497")
PIS8343 = Participants_TechniqueIdsResting1UpSampledResults.get("PIS-8343")
PIS7728 = Participants_TechniqueIdsResting1UpSampledResults.get("PIS-7728")
PIS8308 = Participants_TechniqueIdsResting1UpSampledResults.get("PIS-8308")
PIS8308P2 = Participants_TechniqueIdsResting1UpSampledResults.get("PIS-8308P2")
PIS5868P2 = Participants_TechniqueIdsResting1UpSampledResults.get("PIS-5868P2")
PIS3252P2 = Participants_TechniqueIdsResting1UpSampledResults.get("PIS-3252P2")

# UP_FPS15 = list( set(PIS6729)&set(PIS6888) &set(PIS5868)&set(PIS3186))
# UP_FPS15_2 = list(set(PIS3252))
# print('UP_FPS15')
# print(UP_FPS15)
# print('UP_FPS15_2')
# print(UP_FPS15_2)
#
# UP_FPS30_1 = list( set(PIS1949) &set(PIS7381)&set(PIS2212)&set(PIS4497)&set(PIS8308P2)&set(PIS5868P2)&set(PIS3252P2))
# UP_FPS30_2 = list( set(PIS2047) &set(PIS8308) &set(PIS8343))
# # UP_FPS30_3 =PIS9219
# print('UP_FPS30_1')
# print(UP_FPS30_1)
# print('UP_FPS30_2')
# print(UP_FPS30_2)
#
#
# FPS15andFPS30Common =list(set(UP_FPS15_2)&set(UP_FPS30_2))
# print('FPS15andFPS30Common')
# print(FPS15andFPS30Common)
#
# UP_Remaining0=(PIS4014)
# UP_Remaining1 =  list(  set(PIS4709) & set(PIS8073)&set(PIS9219) )
# UP_Remaining2 =PIS6327
# UP_Remaining3 =  PIS7728
# print('UP_Remaining0')
# print(UP_Remaining0)
# print('Remaining1')
# print(Remaining1)
# print('Remaining2')
# print(Remaining2)
# print('Remaining3')
# print(Remaining3)
CommonTechniqueAmongAllOriginal = list(   set(PIS1949 ) &
                                         set(PIS7381 ) & set(PIS2212 )& set(PIS5868 )
                                          &set(PIS8308P2 ) &
                                         set(PIS3252P2 )&set(PIS5868P2 ) & set(PIS4497 ) & set(PIS8073)& set(PIS6729 )& set(PIS6888 )& set(PIS4709 ) )
print('CommonTechniqueAmongAllOriginal')
print(CommonTechniqueAmongAllOriginal)

allodd1 = list(  set(PIS3186 ) & set(PIS7728 ))
allodd2 = list(set(PIS4014 )& set(PIS2047) & set(PIS8308 ))
allodd3 = list(   set(PIS3252 )& set(PIS8343 ) )
a=0
# CommonTechniqueAmongAllOriginal = list(   set(p11) & set(p12))
# #
# #
# #                                          set(p6) & set(p12)& set(p13)&set(p19) &
# #                                          set(p21)&set(p20) & set(p15) & set(p1)& set(p8)& set(p10)& set(p7) )
#
#
# CommonTechniqueAmongAllOriginal = list(   set(p4) &
#                                          set(p6) & set(p12)& set(p13)&set(p19) &
#                                          set(p21)&set(p20) & set(p15) & set(p1)& set(p8)& set(p10)& set(p7) )
#
# print('CommonTechniqueAmongAllOriginal')
# print(CommonTechniqueAmongAllOriginal)
#
# allodd1 = list(  set(p5) & set(p17))
# allodd2 = list(set(p3)& set(p2) & set(p18))
# allodd3 = list(  set(p2) &  set(p16) )
# print('allodd1')
# print(allodd1)
# print('allodd2')
# print(allodd2)
# print('allodd3')
# print(allodd3)
# print('p9')
# print(p9)

# CommonTechniqueAmongAllOriginal4 = list(   set(p17) & set(CommonTechniqueAmongAllOriginal) )

# CommonTechniqueAmongAllOriginal = list(   set(CommonTechniqueAmongAllOriginal) & set(CommonTechniqueAmongAllOriginal3) )


#
# CommonTechniqueAmongAllOriginal = list(  set(p2) &
#                                          set(p4) &
#                                          set(p6) &
#                                          set(p12) &
#                                          set(p15) &
#                                          set(p16) &
#                                          set(p17)  &
#                                          set(p18) &
#                                          set(p19)&
#                                          set(p20)&
#                                          set(p21))
# a=0
# allp=p1+p2+p3+p4+p5+p6+p7+p8+p9+p10+p11+p12+p13+p14+p15+p16+p17+p18+p19+p20+p21
# my_list_allp = list(set(allp))

# for item in my_list_allp:
#     print(str(item) + ',')
#
# p = []
# kp = []
# for k, v in Participants_TechniqueIdsResting1ResultsInitial.items():
#     p.append(v)
#     kp.append(k)
#
# result = set(p[0])
# lenR = len(result)
# count = 0
# for s in p[1:]:
#     pino = kp[count]
#     if(len(s)>=lenR):
#         print(pino)
#         if(len(s)>0):
#             result.intersection_update(s)
#     count = count +1

# # print(result)
# CommonTechniqueAmongAllOriginal = list(  set(p3) &
#                                          set(p4) &
#                                          set(p5) &
#                                          set(p6) &
#                                          set(p8) &
#                                          set(p10) &
#                                          set(p13) &
#                                          set(p17)  &
#                                          set(p20)
#                                         )
#
# CommonTechniqueAmongAllOriginal2 = list(
#                                          set(p2) &
#                                          set(p7)& ## has 7642 matches PIS-6327
#                                           set(p16) &
#                                          set(p18)
#                                         )
#
# CommonTechniqueAmongAllOriginal3 = list(
#                                          set(p1) &
#                                          # set(p9) &
#                                          # set(p11) &
#                                          # set(p12) &
#                                          # set(p14)&
#                                          set(p15) &
#                                          set(p19) &
#                                          set(p21)
#                                         )
#
# CommonTechniqueAmongAllOriginal4 = list(
#                                          set(p11) &## has 7642 matches PIS-6327
#                                          set(p12) #&## has 7642 matches PIS-6327
#                                       #  set(p9)
#                                         )
#
# CommonTechniqueAmongAllOriginal5 = list(  set(p14) )
#
#
# print(CommonTechniqueAmongAllOriginal)
# print(CommonTechniqueAmongAllOriginal2)
# print(CommonTechniqueAmongAllOriginal3)
# print(CommonTechniqueAmongAllOriginal4)
# a=0