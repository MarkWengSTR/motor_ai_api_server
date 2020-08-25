from ANSYS_function import Property_setting
import json
import time





data=Property_setting("spm_10p12s_sample")


data._add_property("Dso",                1.1*105.2,              0.9*105.2,                 "mm")
data._add_property("Dsi",                1.1*57,                 0.9*57,                    "mm")
data._add_property("length",             1.1*96,                 0.9*96,                    "mm")
data._add_property("airgap",             1.1*0.5,                0.9*0.5,                   "mm")
data._add_property("mag_thick",          1.1*3,                  0.9*3,                     "mm")
data._add_property("mag_emb",            1.1*0.8,                0.9*0.8,                   "mm")
data._add_property("Wy",                 1.1*6.4,                0.9*6.4,                   "mm")
data._add_property("Wt",                 1.1*9.2,                0.9*9.2,                   "mm")
data._add_property("Hs0",                1.1*1,                  0.9*1,                     "mm")
data._add_property("Hs1",                1.1*1,                  0.9*1,                     "mm")
data._add_property("Bs0",                1.1*4.5,                0.9*4.5,                   "mm")
#data._add_property("slot_area"         ,1.1*12,10)              0.9*           
data._add_property("slot",               1.1*12,                 0.9*12,               "",dtype="int")
data._add_property("Im",                 1.1*297,                0.9*297,                   "A")
data._add_property("speed_rpm",          1.1*1769,               0.9*1769,                  "rpm")

data.output_property_setting_as_jsonfile()


for i in range(10000):
    data._set_property_in_random()
    data._run_analysis()
    data._get_output_data()

    data._add_output("Torque","output_temp/Moving1.csv")
    data._add_output("Voltage","output_temp/InducedVoltage(Winding_A)-InducedVoltage(Winding_B).csv")
    data._add_output("CoreLoss","output_temp/CoreLoss.csv")
    print("Sum:",i+1)


    data.output_data_as_jsonfile()
    data.initialize()
#data.property_print()

#change_property("slot","12*2")



"""
for i in range(5):
    #跑分析
    try:
        oDesign.AnalyzeAllNominal()
    except:
        print(i)
"""

#輸出csv
#oModule = oDesign.GetModule("ReportSetup")
#oModule.ExportToFile("Moving1.Torque", "D:/SPM_10p12s/Moving1.csv")