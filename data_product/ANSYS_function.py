from win32com import client
import pandas  as pd
import os
import sys
import random
import json
import datetime
import time
import shutil


class Property_setting():
    
    def __init__(self,project_name):
        self.project_name=project_name
        self.property_range_setting={}
        self.dataset={"input":{},"output":{}}
        self.dataset["output"]["intersect"]="NO"
        oAnsoftApp = client.Dispatch("Ansoft.ElectronicsDesktop")
        oDesktop = oAnsoftApp.GetAppDesktop()
        oProject = oDesktop.SetActiveProject(project_name)
        self.oDesign = oProject.SetActiveDesign("Maxwell2DDesign1")

        self.OK_number=0
        self.Err_number=0
    def _add_property(self,key,maximum,minimum,unit,dtype="float"):
        
        self.property_range_setting[key]={"max":maximum,"min":minimum,"unit":unit,"dtype":dtype}

    def property_print(self):
        print(self.property_range_setting)
        print(self.property_range_setting.keys())
        #print(self.property_range_setting["output"].keys())

    def _add_output(self,key,file_name):
        if not self.dataset["output"]["intersect"]=="YES":
            with open(file_name,'r') as fp:
                str_data = fp.read()

            self.dataset["output"][key]=str_data

    def _change_output_to_csv_DataFrame(self,key):
    
        if sys.version_info[0] < 3: 
            from StringIO import StringIO
        else:
            from io import StringIO

        csv_data=pd.read_csv(StringIO(self.property_range_setting[key]))

    def _set_property_in_random(self):
        for key in self.property_range_setting.keys():
            
            random_value=self.property_range_setting[key]["min"]+(self.property_range_setting[key]["max"]-self.property_range_setting[key]["min"])*random.random()
            
            if self.property_range_setting[key]['dtype']=='int':
                random_value=int(random_value)
     
            self.dataset["input"][key]={"value":random_value,"unit":self.property_range_setting[key]["unit"]}
            self.oDesign.ChangeProperty(self.change_property_struct(key,str(random_value)+self.property_range_setting[key]["unit"]))
            
    def change_property_struct(self,type_name,value):
    
        struct=[
                "NAME:AllTabs",
                [
                    "NAME:LocalVariableTab",
                    [
                        "NAME:PropServers", 
                        "LocalVariables"
                    ],
                    [
                        "NAME:ChangedProps",
                        [
                            "NAME:"+type_name,
                            "Value:="		, value
                        ]
                    ]
                ]
                   ]
        return(struct)

    def _run_analysis(self):


        try:
            self.OK_number+=1
            self.oDesign.AnalyzeAllNominal()
            #print("OK:",self.OK_number)
        except:
            self.Err_number+=1
            self.dataset["output"]["intersect"]="YES"
            print("Err:",self.Err_number)
            
    def _get_output_data(self):
    
        
        if not self.dataset["output"]["intersect"]=="YES":
            oModule = self.oDesign.GetModule("ReportSetup")
            oModule.ExportToFile("Moving1.Torque", os.getcwd()+"/output_temp/Moving1.csv")
            oModule.ExportToFile("InducedVoltage(Winding_A)-InducedVoltage(Winding_B)", os.getcwd()+ "/output_temp/InducedVoltage(Winding_A)-InducedVoltage(Winding_B).csv")
            oModule.ExportToFile("CoreLoss", os.getcwd()+ "/output_temp/CoreLoss.csv")
        
       
    
    def output_property_setting_as_jsonfile(self):
        ret = json.dumps(self.property_range_setting)
        with open('json_data/property_setting.json', 'w') as fp:
            fp.write(ret)

    def output_data_as_jsonfile(self):
        nowtime=datetime.datetime.today()
        time_delta = datetime.timedelta(hours=3) #時差
        nowtime = nowtime + time_delta #本地時間加3小時
        datetime_format = nowtime.strftime("%Y%m%d%H%M%S")
        
        serial_number=self.project_name+datetime_format
        
        ret = json.dumps(self.dataset)
        with open('json_data/rawdata/'+serial_number+'.json', 'w') as fp:
            fp.write(ret)
            
            
    def initialize(self):
        self.dataset={"input":{},"output":{}}
        self.dataset["output"]["intersect"]="NO"
        for file in os.listdir("output_temp"):
            os.remove("output_temp/"+file)