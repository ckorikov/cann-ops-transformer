#!/bin/bash

declare -A NumCoresMap
NumCoresMap["Ascend910A"]="32"
NumCoresMap["Ascend910B1"]="25"
NumCoresMap["Ascend910B2"]="24"
NumCoresMap["Ascend910B3"]="20"
NumCoresMap["Ascend910B4"]="20"

export NumCoresMap