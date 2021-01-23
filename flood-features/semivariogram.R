# load libraries

require(geoR)
require(MASS)
library(e1071)
library(stats)
library(geosphere)

# flood-features folder location
local = '/home/thiago-costa/projects/multimodal-fusion-floods-tweets-climatics/flood-features/data'

local_df = paste(local, "ALAGAMENTOS-2015_2018-GEO_CERTO_INSIDE_ARRANGE_HOURS_PRECIPITACAO_FEATURES_umidadeOK.csv", sep='/')

#------load csv------

df_floods  = read.csv(local_df)

#------create dataframe------
coordinates_of_floods <- data.frame("Latitude" = df['latitude'], "Longitude" = df['longitude'],"umidade" = df['umidade'])

#------spliting data------
df_floods = as.geodata(coordinates_of_floods,coords.col = 1:2, data.col = 3)

print(df$data)

#------distance------
distance = round(max(distm(df_floods$coords, fun=distHaversine))/2000)

print('----DISTANCE----')
print(distance)

#------variog------
df_floods.var <- variog(df_floods,coords = df_floods$coords, data= df_floods$data, max.dist= distance,estimator.type="classical")

#-------------------------VARIOFIT-------------------------
#print('*********************SUMMARY*********************')

#eyefit_values=eyefit(df_floods.var)

#print("sillp estimated")
#sillp_estimated=eyefit_values[[1]]$cov.pars[1]   ##C1
#print(sillp_estimated)

#print("alcance estimated")
#alcance_estimated=eyefit_values[[1]]$cov.pars[2] ##phi
#print(alcance_estimated)

#print("pepita estimated")
#pepita_estimated=eyefit_values[[1]]$nugget       ##C0
#print(pepita_estimated)

## ########################################################################################### #
##------------------ADJUSTING THE BEST MODEL BASED ON VARIOFIT RESULTS------------------------
## ########################################################################################### #

# empirical results through variofit
sillp = 10000
alcance = 0.002
pepita = 0

sillp_gau = sillp
alcance_gau = alcance
pepita_gau = pepita

local_semivariance = paste(local, 'semivariance.png', sep='/')

png(local_semivariance)

plot(df_floods.var,
	xlab='Distance of floods',
	ylab='Semivariance',
	weights="equals",
	xlim=c(0, 0.42),
	main='Result of the Semivariogram')
	dgau.wls <- variofit(df_floods.var,
		ini=c(sillp_gau, alcance_gau), 
		cov.model="gaus")
		
lines(dgau.wls,col="red")
legend("topright",c("Gauss"),fill=c("blue"),cex=1.0)

dev.off()

print('----------------GUASSIAN----------------')
print(summary(dgau.wls))