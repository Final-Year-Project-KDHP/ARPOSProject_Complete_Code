import pandas as pd
import pyodbc


class SQLConfig:
    #dbconnection
    conn = None
    cursor = None
    # Constructor
    def __init__(self):
        # Defining the connection string
        self.conn = pyodbc.connect('''DRIVER={SQL Server}; Server=ELCIELO; 
                                UID=sa; PWD=A@1234567; DataBase=ARPOS''')
        self.cursor = self.conn.cursor()

    def getProcessedCases(self,SkinGroup,AttemptType):
        Query = "select  CONCAT(Techniques.AlgorithmType , '_PR-',Techniques.PreProcess,'_FFT-',Techniques.FFT , '_FL-' , Techniques.Filter  , '_RS-' , Techniques.Result , '_SM-' + Techniques.Smoothen) as CaseProcessed, ParticipantsResultsEntireSignal.ParticipantId,HeartRateStatus,UpSampled from ParticipantsResultsEntireSignal join Techniques on Techniques.Id = ParticipantsResultsEntireSignal.TechniqueId"
        Query = Query + " join Participants on Participants.ParticipantId = ParticipantsResultsEntireSignal.ParticipantId where SkinGroup = '" + SkinGroup + "' and AttemptType=" + AttemptType
        dataTable = pd.read_sql_query(Query, self.conn)
        dataTable.head()
        return dataTable

    def getTableQueryDifferences(self, skinGroup,PreProcess,FFT,Filter,Result,Smoothen):
        ##Parameters
        param = "declare @skinGroup as varchar(150), @PreProcess int,@FFT varchar(50), @Filter int, @Result int, @Technique int, @Smoothen varchar(50) " +\
                " set @PreProcess = " + PreProcess +\
                " set @FFT = '" + FFT + "' " +\
                " set @Filter = " + Filter +\
                " set @Result = " + Result +\
                " set @Smoothen = '" + Smoothen + "' " +\
                " set @skinGroup = '" + skinGroup + "' "

        QuerySelect = "select"
        QueryColumn1 = "Resting1Results.ParticipantId, Resting1Results.DifferenceHR as 'OriginalDifference', "
        QueryColumn2 = "(select DifferenceHR from Resting1UpSampledResults where Resting1UpSampledResults.TechniqueId = Resting1Results.TechniqueId and Resting1UpSampledResults.ParticipantId = Resting1Results.ParticipantId ) as 'UpSampledDifference', "
        QueryColumn3 = "(select DifferenceHR from Resting1RevisedResults where Resting1RevisedResults.TechniqueId = Resting1Results.TechniqueId and Resting1RevisedResults.ParticipantId = Resting1Results.ParticipantId ) as 'RevisedOriginalDifference', "
        QueryColumn4 = "Techniques.*"
        QueryTable = "from Resting1Results"
        QueryJoins = "join Techniques on Techniques.Id = TechniqueId join Participants on Participants.ParticipantId = Resting1Results.ParticipantId"
        QueryCondition = "where TechniqueId in (select Id from Techniques where  PreProcess = @PreProcess and FFT = @FFT and Filter = @Filter and Result =@Result and Smoothen = @Smoothen) AND SkinGroup=@skinGroup "

        FullQuery = param + " " + QuerySelect + " " + QueryColumn1 + QueryColumn2 + QueryColumn3 + QueryColumn4 + " " + QueryTable  + " " + QueryJoins + " " + QueryCondition
        return FullQuery

    def getTableQueryDifferencesByTechniqueAndHeartRateStatus(self, skinGroup,PreProcess,FFT,Filter,Result,Smoothen,HeartRateStatus,UpSampled):
        ##Parameters
        param = "declare @skinGroup as varchar(150), @PreProcess int,@FFT varchar(50), @Filter int, @Result int, @Technique int, @Smoothen varchar(50),@heartRateStatus varchar(150)  " + \
                " set @PreProcess = " + PreProcess + \
                " set @heartRateStatus = '" + HeartRateStatus + "' " +\
                " set @FFT = '" + FFT + "' " + \
                " set @Filter = " + Filter + \
                " set @Result = " + Result + \
                " set @Smoothen = '" + Smoothen + "' " + \
                " set @skinGroup = '" + skinGroup + "' "

        QuerySelect = "select"
        QueryColumn1 = "ParticipantsResultsEntireSignal.ParticipantId, ParticipantsResultsEntireSignal.HeartRateStatus, ParticipantsResultsEntireSignal.differenceHR as 'HeartRateDifference', ParticipantsResultsEntireSignal.differenceSPO as 'SPODifference', ParticipantsResultsEntireSignal.ColorFPS,ParticipantsResultsEntireSignal.IRFPS, "
        # QueryColumn2 = "(select t2.differenceHR from ParticipantsResultsEntireSignal t2 where TechniqueId=ParticipantsResultsEntireSignal.TechniqueId and ParticipantId = ParticipantsResultsEntireSignal.ParticipantId and HeartRateStatus =@heartRateStatus and UpSampled =1) as 'UpSampledHeartRateDifference', "
        # QueryColumn3 = "(select t2.differenceSPO from ParticipantsResultsEntireSignal t2 where TechniqueId=ParticipantsResultsEntireSignal.TechniqueId and ParticipantId = ParticipantsResultsEntireSignal.ParticipantId  and HeartRateStatus =@heartRateStatus and UpSampled =1)  as 'UpSampledSPODifference', "
        QueryColumn4 = "Techniques.*"
        QueryTable = "from ParticipantsResultsEntireSignal"
        QueryJoins = "join Participants on Participants.ParticipantId = ParticipantsResultsEntireSignal.ParticipantId join Techniques on Techniques.Id = ParticipantsResultsEntireSignal.TechniqueId "
        QueryCondition = "where TechniqueId in (select Id from Techniques where  PreProcess = @PreProcess and FFT = @FFT and Filter = @Filter and Result =@Result and Smoothen = @Smoothen) and SkinGroup =@SkinGroup and HeartRateStatus =@heartRateStatus and UpSampled =" + UpSampled

        FullQuery = param + " " + QuerySelect + " " + QueryColumn1 +  QueryColumn4 + " " + QueryTable  + " " + QueryJoins + " " + QueryCondition
        return FullQuery

    def getTableQueryTimetoRun(self, skinGroup,PreProcess,FFT,Filter,Result,Smoothen,Position):
        ##Parameters
        FullQuery = "exec AverageTimeAgainstTechnique '" + skinGroup + "', " + PreProcess + ",'" + FFT + "'," + Filter + "," + Result + ",'" + Smoothen + "', '" + Position + "'"
        dataTable = pd.read_sql_query(FullQuery, self.conn)
        dataTable.head()
        return dataTable

    def getTableQueryTimetoRunForAll(self, HeartRateStatus,UpSampled,AttemptType, TechniqueId):
        ##Parameters
        FullQuery = "exec AverageTimeAgainstTechniqueIdForAll " + UpSampled + ",'" + HeartRateStatus + "'," + AttemptType + "," + TechniqueId
        dataTable = pd.read_sql_query(FullQuery, self.conn)
        dataTable.head()
        return dataTable

    def GetAllTimeForTechniqueId(self, HeartRateStatus,UpSampled,AttemptType, TechniqueId):
        ##Parameters
        FullQuery = "exec GetAllTimeForTechniqueId " + UpSampled + ",'" + HeartRateStatus + "'," + AttemptType + "," + TechniqueId
        dataTable = pd.read_sql_query(FullQuery, self.conn)
        dataTable.head()
        return dataTable

    def getTableQueryGroupWiseDifferences(self, skinGroup,PreProcess,FFT,Filter,Result,Smoothen):
        ##Parameters
        FullQuery = "exec GroupWiseDataOriginalandUpSampled '" + skinGroup + "', " + PreProcess + ",'" + FFT + "'," + Filter + "," + Result + ",'" + Smoothen + "'"
        dataTable = pd.read_sql_query(FullQuery, self.conn)
        dataTable.head()
        return dataTable

    def ReadDataTableByTechniqueParameters(self,skinGroup,PreProcess,FFT,Filter,Result,Smoothen):
        FullQuery = self.getTableQueryDifferences(skinGroup,PreProcess,FFT,Filter,Result,Smoothen)
        dataTable = pd.read_sql_query(FullQuery, self.conn)
        dataTable.head()
        return dataTable

    def GetBestSouthAsianCases(self,skinGroup,HeartRateStatus,UpSampled,AttemptType, TechniqueId):
        ##Parameters
        FullQuery = "exec GetBestSouthAsianCases '" + skinGroup + "', " + UpSampled + ",'" + HeartRateStatus + "'," + AttemptType + "," + TechniqueId
        dataTable = pd.read_sql_query(FullQuery, self.conn)
        dataTable.head()
        return dataTable

    def GetBestAmongAll(self,HeartRateStatus,UpSampled,AttemptType, TechniqueId):
        ##Parameters
        FullQuery = "exec GetBestAmongAll " + UpSampled + ",'" + HeartRateStatus + "'," + AttemptType + "," + TechniqueId
        dataTable = pd.read_sql_query(FullQuery, self.conn)
        dataTable.head()
        return dataTable

    def GetUpSampledVSNotSampledDataSpecific(self,HeartRateStatus,UpSampled,AttemptType, TechniqueId):
        ##Parameters
        FullQuery = "exec GetUpSampledVSNotSampledDataSpecific " + UpSampled + ",'" + HeartRateStatus + "'," + AttemptType + "," + TechniqueId
        dataTable = pd.read_sql_query(FullQuery, self.conn)
        dataTable.head()
        return dataTable

    def ReadDataTableByTechniqueAndHeartRateStatus(self,skinGroup,PreProcess,FFT,Filter,Result,Smoothen,HeartRateStatus,UpSampled):
        FullQuery = self.getTableQueryDifferencesByTechniqueAndHeartRateStatus(skinGroup,PreProcess,FFT,Filter,Result,Smoothen,HeartRateStatus,UpSampled)
        dataTable = pd.read_sql_query(FullQuery, self.conn)
        dataTable.head()
        return dataTable

    def getTechniqueDetailFromId(self,TechniqueId):
        TechniqueIdQuery = "select *  from Techniques where Id = "+ TechniqueId
        dataTable = pd.read_sql_query(TechniqueIdQuery, self.conn)
        dataTable.head()
        return dataTable

    def getTechniqueId(self,AlgorithmType,PreProcess,FFT,Filter,Result,Smoothen):
        TechniqueIdQuery = "select Id as TechniqueId from Techniques where AlgorithmType = '"+ AlgorithmType + "' and PreProcess="+\
                      PreProcess + " and " + "FFT='"+FFT+"' and Filter="+Filter+" and Result="+Result+" and Smoothen='"+Smoothen+"'"

        self.cursor.execute(TechniqueIdQuery)
        # result_set = self.cursor.fetchall()
        TechniqueId = self.cursor.fetchone()[0]
        # for row in result_set[0]:
        #     TechniqueId = str(row["TechniqueId"])

        return TechniqueId

    def ExistsInDb(self,AlgorithmType,PreProcess,FFT,Filter,Result,Smoothen,HeartRateStatus, ParticipantId):
        TechniqueId = self.getTechniqueId(AlgorithmType,str(PreProcess),FFT,str(Filter),str(Result),str(Smoothen))
        queryExists = "select count(ParticipantResultId) as CountRows from ParticipantsResultsEntireSignal " \
                      "where ParticipantId= '"  + ParticipantId + "' and HeartRateStatus = '"  + HeartRateStatus + "' and TechniqueId="+str(TechniqueId)

        self.cursor.execute(queryExists)
        CountValue = self.cursor.fetchone()[0]
        # for row in result_set:
        #     CountValue = row["CountRows"]
        if(CountValue>0):
            return True
        else:
            return False

    def SaveRowParticipantsResultsEntireSignal(self, objParticipantsResultEntireSignalDataRow):
        try:

            self.cursor.execute("exec AddParticipantsResultEntireSignal '"+ objParticipantsResultEntireSignalDataRow.ParticipantId+ "', '" +
                                objParticipantsResultEntireSignalDataRow.HeartRateStatus + "', " +
                                objParticipantsResultEntireSignalDataRow.bestHeartRateSnr+ ", " +
                                objParticipantsResultEntireSignalDataRow.bestBPM + ",'"+
                                objParticipantsResultEntireSignalDataRow.channelType + "','"+
                                objParticipantsResultEntireSignalDataRow.regionType + "'," +
                                objParticipantsResultEntireSignalDataRow.FrequencySamplingError + ","+
                                objParticipantsResultEntireSignalDataRow.oxygenSaturationValueError + ","+
                                objParticipantsResultEntireSignalDataRow.heartRateError + "," +
                                objParticipantsResultEntireSignalDataRow.bestSPO + "," +
                                objParticipantsResultEntireSignalDataRow.HeartRateValue + "," +
                                objParticipantsResultEntireSignalDataRow.SPOValue + "," +
                                objParticipantsResultEntireSignalDataRow.differenceHR + "," +
                                objParticipantsResultEntireSignalDataRow.differenceSPO + ",'" +
                                objParticipantsResultEntireSignalDataRow.TotalWindowCalculationTimeTaken + "','" +
                                objParticipantsResultEntireSignalDataRow.PreProcessTimeTaken + "','"+
                                objParticipantsResultEntireSignalDataRow.AlgorithmTimeTaken + "','"+
                                objParticipantsResultEntireSignalDataRow.FFTTimeTaken + "','"+
                                objParticipantsResultEntireSignalDataRow.SmoothTimeTaken + "','"+
                                objParticipantsResultEntireSignalDataRow.FilterTimeTaken + "','"+
                                objParticipantsResultEntireSignalDataRow.ComputingHRSNRTimeTaken + "','"+
                                objParticipantsResultEntireSignalDataRow.ComputingSPOTimeTaken + "','"+
                                # objParticipantsResultEntireSignalDataRow.TechniqueId + ","+
                                objParticipantsResultEntireSignalDataRow.Algorithm_type  + "',"+
                                objParticipantsResultEntireSignalDataRow.Preprocess_type   + ",'"+
                                objParticipantsResultEntireSignalDataRow.FFT_type   + "',"+
                                objParticipantsResultEntireSignalDataRow.Filter_type   + ","+
                                objParticipantsResultEntireSignalDataRow.Result_type   + ",'"+
                                objParticipantsResultEntireSignalDataRow.isSmoothen    + "',"+
                                objParticipantsResultEntireSignalDataRow.UpSampled + ","+
                                objParticipantsResultEntireSignalDataRow.ColorFPS + ","+
                                objParticipantsResultEntireSignalDataRow.IRFPS + ",'"+
                                objParticipantsResultEntireSignalDataRow.SelectedColorFPSMethod + "','"+
                                objParticipantsResultEntireSignalDataRow.SelectedIRFPSMethod + "', " +
                                objParticipantsResultEntireSignalDataRow.GroundTruthHeartRate + ","+
                                objParticipantsResultEntireSignalDataRow.GroundTruthSPO + ","+
                                objParticipantsResultEntireSignalDataRow.AttemptType + ",'"+
                                objParticipantsResultEntireSignalDataRow.FPSNotes + "' "
                                )

            self.conn.commit()
        except Exception:
            print('ERROR adding ' + objParticipantsResultEntireSignalDataRow.Algorithm_type + ', preprocess: ' +objParticipantsResultEntireSignalDataRow.Preprocess_type
                  + ', FFT: ' + objParticipantsResultEntireSignalDataRow.FFT_type
                  + ', Filter: ' + objParticipantsResultEntireSignalDataRow.Filter_type  + ', Result: ' + objParticipantsResultEntireSignalDataRow.Result_type
                  + ', Smoothen' + objParticipantsResultEntireSignalDataRow.SmoothTimeTaken
                  )
