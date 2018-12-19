def getMacroVar(macroString,macroVarName,geoLocation,macroData):

    macroVarNameList = macroVarName[0].tolist()
    matchers = macroString.split()

    macroVarNameList = [item.lower() for item in macroVarNameList]
    matchers = [item.lower() for item in matchers]

    columnPos = [position for position, fullNames in enumerate(macroVarNameList) if all(strings in fullNames for strings in matchers)][0]

    macroDataSubset = pd.DataFrame(pd.to_datetime(macroData[0])).join(pd.DataFrame(macroData[columnPos]).astype(float))
    macroDataSubset.columns = ["Date", macroString]

    return macroDataSubset
    

    
