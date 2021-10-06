library(foreign)
library(readxl)
library(haven)

# load ornge month data
ornge_month <- readxl::read_xlsx('data/Ornge Data_Month.xlsx')

# load ornge stata file 
orgne <- read_dta('data/Ornge Data FY19-20.dta')

