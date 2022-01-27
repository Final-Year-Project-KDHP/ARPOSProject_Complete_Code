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
                            objParticipantsResultEntireSignalDataRow.GroundTruthSPO + " ")

        self.conn.commit()

