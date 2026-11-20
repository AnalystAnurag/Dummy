## 1. Packages and libraries

Install once, if not already installed:

```r
install.packages(c(
  "readxl", "TTR", "dygraphs", "zoo", "forecast", "tseries",
  "ggplot2", "dplyr", "tidyr", "moments", "vars", "mFilter",
  "tidyverse", "urca", "stargazer", "tsDyn", "pdfetch", "quantmod",
  "xts", "FinTS", "rugarch", "rmgarch", "BEKKs", "MTS"
))
```
```r
library(readxl)
library(TTR)
library(dygraphs)
library(zoo)
library(forecast)
library(tseries)
library(ggplot2)
library(dplyr)
library(tidyr)
library(moments)
library(vars)
library(mFilter)
library(tidyverse)
library(urca)
library(stargazer)
library(tsDyn)
library(pdfetch)
library(quantmod)
library(xts)
library(FinTS)
library(rugarch)
library(rmgarch)
library(BEKKs)
library(MTS)
```



## Part I – ARIMA concepts and forecasting


## Downloading data from Yahoo Finance

```r
#remove.packages("pdfetch")
```

```r
#nifty_50 <-  pdfetch_YAHOO("^NSEI")
#nifty_50
```

```r
#plot(nifty_50$`^NSEI.close`)
```

## Another method to download the data

```r
#library(quantmod)
```

```r
#getSymbols("^NSEI", open= "2021-01-01", close = "2026-09-19")
#View(NSEI)
```

```r
#plot.ts(NSEI$NSEI.Close)
```

```r
## Downloading multiple dataset
#library(pdfetch)
#x=pdfetch_YAHOO(c("^NSEI","ITC.NS","SBIN.NS","RELIANCE.NS"),
               # "close",from="2015-01-01",to="2026-08-05")
#View(x)
#data=read.delim("clipboard")
#data
```

```r
data("AirPassengers")
air<- AirPassengers
str(air)

```

```r
head(air)
```

```r
ts(air, frequency = 12, start=c(1949,1))
```

```r
attributes(air)
```

```r
# Log Transformation

lair<- log(air)
lair
```

```r
decomp <- decompose(air)
decomp$figure
plot(decomp$figure,
     type = 'b',
     xlab= 'Month',
     ylab= "Seasonality Index",
     col = "blue",
     las = 2)

plot(decomp)
```

#### Interpretation:

#### Seasonality Plot

- Under an additive decomposition, July tends to be approximately 64 units higher than the underlying trend, purely because of seasonal effects.
- November tends to be approximately 54 units below the underlying trend because of seasonality.
- July to September, there is increasing effect and October to December its showing declining effect.

#### Decomposition Plot

- The additive decomposition indicates that the time series contains three major systematic features: a strong upward trend, a pronounced and recurring 12-month seasonal pattern, and an irregular component fluctuating around zero. The trend component shows sustained growth over the study period, while the seasonal component exhibits a highly consistent pattern, with positive effects during the middle months of the year and negative effects toward the beginning and end of the year. The residual component does not display a clear systematic pattern, although its variability appears to increase in the later period.


### Data Set


```r
nft_50 <- read_excel("D:/Msc BA Classnotes/Sem 3/Time series Analysis/Nifty 50.xlsx")
nft_50
```

```r
nft_50$Date <- as.Date(nft_50$Date)
```

```r
names(nft_50)<- c('Date','Price')
```

```r
nft_50 <- nft_50[order(nft_50$Date),]
```

```r
nft_50<- na.omit(nft_50)
```

```r
#plotting original daily price

plot(nft_50$Date, nft_50$Price, type = "l",
     main = "Nifty 50", xlab ="Date", ylab="Price")
```

```r
## Moving Average
```

```r
nft_50$ma_5 <- SMA(nft_50$Price, 5)
nft_50$ma_30 <- SMA(nft_50$Price, 30)
nft_50$ma_90 <- SMA(nft_50$Price, 90)

```

```r
# Convert to xts
data_xts <- xts::xts(
  nft_50[, c("Price", "ma_5", "ma_30", "ma_90")],
  order.by = nft_50$Date
)

# Interactive plot
dygraph(data_xts, main = "Nifty 50 Moving Averages") %>%
  dyRangeSelector()
```

#### Least-squares Method

```r
nft_50$t <- 1:nrow(nft_50)

model <- lm(Price ~ t, data = nft_50)

nft_50$LS_Trend <- fitted(model)

ls_xts <- xts(
  nft_50[, c("Price", "LS_Trend")],
  order.by = nft_50$Date
)

dygraph(ls_xts, main = "Nifty 50: Least Squares Trend") %>%
  dyRangeSelector()

```

#### Semi averages

```r
n <- nrow(nft_50)
h <- floor(n / 2)

mean1 <- mean(nft_50$Price[1:h])
mean2 <- mean(nft_50$Price[(h + 1):n])

nft_50$SemiAvg <- c(
  rep(mean1, h),
  rep(mean2, n - h)
)

semi_xts <- xts(
  nft_50[, c("Price", "SemiAvg")],
  order.by = nft_50$Date
)

dygraph(semi_xts, main = "Nifty 50: Semi-Average Trend") %>%
  dyRangeSelector()
```


### Arima  #################################################### 

## Data import

```r
nift <- read_excel("D:/Msc BA Classnotes/Sem 3/Time series Analysis/Nifty 50.xlsx")
nift
```

### Data Cleaning

```r
nift$Date <- as.Date(nift$Date)
nift <- na.omit(nift)
```

```r
nift$Price <- as.numeric(gsub(",","", nift$Price))
```

## Converting data to zoo series

```r
nft_y <- zoo(nift$Price, order.by = nift$Date)
```

```r
nft_y <- ts(coredata(nft_y))
```

### Performing Stationary test

$$
\begin{aligned}
  H_0 &: \text{The time series is non-stationary (contains a unit root)} \\
  H_1 &: \text{The time series is stationary}
\end{aligned}
$$

```r
adf.test(nft_y)
```

#### Interpretation:

- As the p-value is greater than 0.05, we fail to reject the null hypothesis and conclude that the data is non stationary.

```r
kpss.test(nft_y)
```

#### Interpretation:

- Given the p-value is less than 0.05, we reject the null hypothesis and conclude that data is non stationary.

## Converting data to stationary series

```r
nft_yy <- diff(nft_y)
```

## Further checking stationarity

```r
adf.test(nft_yy)
```

## Removing Na values from the difference series

```r
nft_yy_clean <- na.omit(nft_yy)
```

#### Interpretation:

- As the p-value is less than 0.05, we reject the null hypothesis concluding that data is stationary at first order difference.

## Performing ACF & PACF plot

```r
#ACF for MA term
#PACF for AR term
acf(nft_yy_clean, main="ACF of first differenced series", na.action = na.pass)
pacf(nft_yy_clean, main="PACF of first differenced series", na.action = na.pass)
```

### Manually estimating candidate ARIMA models

```r
m1 <-  Arima(nft_y, order = c(0,1,1))
m2 <-  Arima(nft_y, order = c(1,1,1))
m3 <-  Arima(nft_y, order = c(1,1,1))
m4 <-  Arima(nft_y, order = c(2,1,1))
```

### Comparing AIC & BIC values

```r
AIC(m1,m2,m3,m4)
BIC(m1,m2,m3,m4)
```

#### Interpretation:

- Based on AIC and BIC values, model1 , i.e m1 appears to be the efficient one.

### Checking Model Accuracy

```r
cat("Accuracy ARIMA model m1:\n")
print(accuracy(m1))
cat("Accuracy ARIMA model m2:\n")
print(accuracy(m2))
cat("Accuracy ARIMA model m3:\n")
print(accuracy(m3))
cat("Accuracy ARIMA model m4:\n")
print(accuracy(m4))
```
#### Interpretation

* Among the four models, m4 produces the lowest training RMSE, MAE and MAPE. However, the differences in accuracy across the models are very small. Therefore, model selection should not be based solely on these training-set accuracy measures; residual diagnostics and out-of-sample forecasting performance should also be considered.

### Model Diagnostics

```r
checkresiduals(m1)
```

### Model forecasting

```r
fc_m1 <- forecast(m1, h = 500)
```

### Plotting the forecasted values

```r
plot(fc_m1, main = "ARIMA forecast", xlab = "Time", ylab = "Price" )
```

### Creating future dates with the SAME length as forecast horizon


```r
last_date <- max(nift$Date)
future_dates <- seq.Date(from = last_date + 1, 
                         by = "day", length.out = length(fc_m1$mean))
```

### Forecast dataframe

```r
fc_nift <- data.frame(
  Date = future_dates,
  Forecast = as.numeric(fc_m1$mean)
)
```

### Actual dataframe

```r
actual_nift <- data.frame(
  Date = nift$Date,
  Price = nift$Price
)
```

### Plot actual and forecast
```r
ggplot() +
  geom_line(data = actual_nift, aes(x = Date, y = Price), color = "blue", 
            linewidth = 1) +
  geom_line(data = fc_nift, aes(x = Date, y = Forecast), color = "red", 
            linewidth = 1) +
  labs(
    title = "Actual Series with ARIMA Forecast",
    x = "Date",
    y = "Price"
  ) +
  theme_minimal()+
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
    plot.title   = element_text(hjust = 0.5)
  )
```


### Auto Arima

##Out-sample

```r
# Load libraries
```

## Data import

```r
data_os <-  read_xlsx("D:/Msc BA Classnotes/Sem 3/Time series Analysis/Nifty 50.xlsx")
data_os
```

### Data Cleaning

```r
# Clean data
data_os$Date  <- as.Date(data_os$Date)
data_os$Price <- as.numeric(gsub(",", "", data_os$Price))
data_os <- data_os %>%
  arrange(Date) %>%
  na.omit()
```

# Creating zoo series

```r
zoo_price <- zoo(data_os$Price, order.by = data_os$Date)
```

# Plot zoo series

```r
plot(zoo_price,
     main = "Nifty 50 Daily Price (Zoo Series)",
     xlab = "Date",
     ylab = "Price",
     col  = "blue",
     lwd  = 2)
```
### Convert to ts for ARIMA fitting

```r
ts_price <- ts(coredata(zoo_price))
```

### Fit ARIMA

```r

fit_full <- auto.arima(ts_price)
summary(fit_full)
```
### Forecast 200 periods

```r
h <- 200
fc_full <- forecast(fit_full, h = h)
```

### Create future dates 

```r
last_date    <- max(index(zoo_price))
future_dates <- seq.Date(from = last_date + 1,
                         by   = "day",
                         length.out = h)
```

### Combine forecast data with dates

```r
fc_df <- data.frame(
  Date     = future_dates,
  Forecast = as.numeric(fc_full$mean)
)
```

### Plot using ggplot so x-axis shows dates

```r
ggplot() +
  geom_line(data = data_os,
            aes(x = Date, y = Price),
            color     = "black",
            linewidth = 1) +
  geom_line(data = fc_df,
            aes(x = Date, y = Forecast),
            color     = "red",
            linewidth = 1)+
  labs(
    title = "Out-of-sample ARIMA Forecast",
    x     = "Date",
    y     = "Price"
  ) +
  theme_minimal() +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
    plot.title   = element_text(hjust = 0.5)
  )
```
### In Sample ARIMA

### Loaded libraries

```r
```

### Data Import

```r
data_is <- read_xlsx("D:/Msc BA Classnotes/Sem 3/Time series Analysis/Nifty 50.xlsx")
data_is
```
### Clean data

```r
data_is$Date <- as.Date(data_is$Date)
data_is$Price <- as.numeric(gsub(",", "", data_is$Price))
data_is <- data_is %>%
  arrange(Date) %>%
  na.omit()
```

### Create zoo series

```r
zoo_price_is <- zoo(data_is$Price, order.by = data_is$Date)
```

### Split into 80% estimation and 20% forecast

```r
n <- length(zoo_price_is)
train_size <- floor(0.8 * n)
train_zoo <- zoo_price_is[1:train_size]
```

```r
test_zoo <- zoo_price_is[(train_size + 1):n]
```

### Converting zoo to ts for ARIMA

```r
train_ts <- ts(coredata(train_zoo))
```

### Fit ARIMA on training sample

```r
fit_train <- auto.arima(train_ts)
summary(fit_train)
```
### Forecast 20% holdout period

```r
fc_20 <- forecast(fit_train, h = length(test_zoo))
```

### Creating holdout results

```r
result_20 <- data.frame(
  Date = index(test_zoo),
  Actual = as.numeric(coredata(test_zoo)),
  Forecast = as.numeric(fc_20$mean)
)
```

### Create full actual series data frame

```r
full_actual <- data.frame(
  Date = index(zoo_price),
  Price = as.numeric(coredata(zoo_price))
)
```

### Plot full actual series + 20% forecast

```r
ggplot() +
  geom_line(data = full_actual, aes(x = Date, y = Price), color = "black", linewidth = 1) +
  geom_line(data = result_20, aes(x = Date, y = Forecast), color = "red", linewidth = 1)+
  labs(
    title = "Nifty 50: Full Actual Series with 20% ARIMA Forecast",
    x = "Date",
    y = "Price"
  ) +
  theme_minimal() +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1)
  )
```
### Accuracy measures for holdout

```r
mae_20 <- mean(abs(result_20$Actual - result_20$Forecast), na.rm = TRUE)
rmse_20 <- sqrt(mean((result_20$Actual - result_20$Forecast)^2, na.rm = TRUE))

cat("MAE:", mae_20, "\n")
cat("RMSE:", rmse_20, "\n")
```

---

## Part II – VAR and VECM

### Var Vecm


## Sector wise time series analysis

```r
sector <- read_xlsx("D:/Msc BA Classnotes/Sem 3/Time series Analysis/SECTOR WISE.xlsx")
sector
```

## Data cleaning

```r
# Convert Date
sector$Date <- as.Date(sector$Date)

# Remove missing observations
sector <- na.omit(sector)
```

##Converting data into zoo series

```r
sec_z <- zoo(sector[,-1], order.by = sector$Date)
plot(sec_z)
```

### Checking summary statistics

```r
summary(sec_z)
```

### Checking peakedness and spread of data

```r
skewness(sec_z)
```

#### Interpretation:

- All the sectors appear to be rightly skewed however Auto sector has more skewness towards right.

```r
kurtosis(sec_z)
```

#### Interpretation:

- Except autosector, all other sectors have kurtosis less than 3 which means they are platykurtic with lesser tails or fewers extreme observations. Autosector is showing leptokurtic nature.

```r
par(mfrow = c(3, 2))

for(i in 1:ncol(sec_z)) {
  
  plot(sec_z[, i],
       main = colnames(sec_z)[i],
       xlab = "Date",
       ylab = "Value",
       col = "blue")
}
```

### Performing stationary test using ADF test

$$
\begin{aligned}
H_0&:\text{Data is non stationary} \\
H_1&:\text{Data is stationary}
\end{aligned}
$$

```r
adf.test(sec_z$Healthcare)
adf.test(sec_z$FMCG)
adf.test(sec_z$FS)
adf.test(sec_z$Auto)
adf.test(sec_z$Bank)
```

- The p-value is greater than 0.05, we fail to reject null hypothesis, there by concluding the data is non stationary.

- To validate further let's perform kpss test

### KPSS Test

$$ 
\begin{aligned}
H_0&: \text{Data is stationary} \\ 
H_1&: \text{Data is non stationary}
\end{aligned}
$$

```r
kpss.test(sec_z$Healthcare)
kpss.test(sec_z$FMCG)
kpss.test(sec_z$FS)
kpss.test(sec_z$Auto)
kpss.test(sec_z$Bank)
```

- The p-value is less than 0.01, we reject the null hypothesis that data is stationary and hereby conclude that the series is non-stationary.

#### Generating return series

```r
ret_health <- diff(log(sec_z$Healthcare))
ret_fmcg <- diff(log(sec_z$FMCG))
ret_fs <- diff(log(sec_z$FS))
ret_auto <- diff(log(sec_z$Auto))
ret_bank <- diff(log(sec_z$Bank))
```

## Performing stationarity test on return series

```r
adf.test(ret_health)
adf.test(ret_fmcg)
adf.test(ret_fs)
adf.test(ret_auto)
adf.test(ret_bank)
```

- The p-value is less than 0.05, so we reject the null hypothesis, thereby concluding that series is stationary.

### Combining the return series

```r
sec_y <- cbind(ret_health , ret_fmcg, ret_fs, ret_auto, ret_bank)
sec_y
```

```r
plot(sec_y)
```

#### Interpretation:

The return series of the Healthcare, FMCG, Financial Services, Auto and Banking sectors fluctuate around zero without exhibiting a clear long-term trend. The series display occasional extreme positive and negative returns and periods of volatility clustering, with particularly pronounced fluctuations around 2020. The simultaneous large movements across several sectors suggest the presence of cross-sector dependence and common market shocks. Overall, the return series appear substantially more stable than the corresponding index-level series, supporting their use in subsequent time-series and volatility analysis.

### Lag selection

```r
```

```r
lagselect <- VARselect(sec_y, lag.max = 10, type= "const")
lagselect
```

#### Interpretation:

\*\* The optimal lag-order selection criteria provide mixed results. The AIC and FPE criteria suggest seven lags, whereas the HQ and Schwarz (SC/BIC) criteria suggest one lag. Since the HQ and SC criteria both select one lag and favour a more parsimonious model, one lag may be considered as the preferred specification, subject to diagnostic testing.

### Performing Johansen's cointegration test

$$
\begin{aligned}
H_0&: \text{ There are no cointegration relationships}\\
H_1&: \text{ There are cointegration relationships}
\end{aligned}
$$

```r
## Assuming lag as 2


jo_test <- ca.jo(sec_z,
                 type = "trace",
                 ecdet ="trend",
                 K = 2)
```

### Checking the summary

```r
summary(jo_test)
```

#### Interpretation:

* We can observe that when r = 0, the test statistic(**97.99**) \> (**87.31**) so we reject the null hypothesis and hereby conclude that there is at-least one cointegrating relationship. Checking further values of r, we can conclude that there is atmost one cointegrating relationship. Even if the sectors show short term relationship between them, there can be mild long run relationship between the sectors.

### Performing VECM test

```r
vecm_model <- cajorls(jo_test, r =1)
summary(vecm_model$rlm)
```
#### Interpretation: 

** The VECM results indicate that the long-run equilibrium relationship is statistically significant for four of the five sectors. The error-correction coefficients for Healthcare (−0.000108, p < 0.001), Financial Services (−0.000207, p < 0.001), Auto (−0.000297, p < 0.001), and Bank (−0.000559, p < 0.001) are negative and statistically significant, indicating that these sectors adjust toward the long-run equilibrium following a deviation. Among these sectors, Banking exhibits the largest absolute adjustment coefficient, suggesting the strongest adjustment response, followed by Auto, Financial Services and Healthcare. In contrast, the error-correction coefficient for FMCG is statistically insignificant (p = 0.832), indicating that FMCG does not significantly adjust to restore the long-run equilibrium. The short-run dynamics are comparatively limited, with significant lagged effects primarily involving Healthcare, Financial Services and Banking.

### Performing VAR test

```r
VAR1 <- VAR(sec_y, lag.max =1, type = "const",
            season = NULL , exog = NULL)
summary(VAR1)
```
```r
stargazer(VAR1[["varresult"]], type = "text")
```
#### Interpretation: 

** The estimated VAR(1) model is stable, meaning that shocks to the system are not expected to persist indefinitely and the model is suitable for further VAR-based analysis such as impulse response and forecast error variance decomposition.

** In the Healthcare equation, only the lagged Healthcare return is statistically significant (β = 0.0677, p < 0.01), indicating limited short-run persistence in Healthcare returns. Lagged returns of FMCG, Financial Services, Auto and Bank do not have statistically significant effects on current Healthcare returns.

** There is no statistically significant evidence that the previous day's returns of any of the five sectors explain current FMCG returns.

** The Financial Services equation shows significant short-run dependence on lagged Healthcare and Banking returns. Lagged Healthcare returns have a negative association with current Financial Services returns (β = −0.0659, p = 0.0119), while lagged Banking returns have a positive association (β = 0.1920, p = 0.0039). The lagged FS return itself is marginally significant at the 10% level.

** Current Auto returns show weak evidence of dependence on lagged Financial Services returns, with a positive coefficient of 0.1155 (p = 0.0904). However, this relationship is only marginally significant at the 10% level.

** The Banking equation indicates significant dependence on lagged Healthcare and Banking returns. Lagged Healthcare returns have a negative association with current Banking returns (β = −0.0711, p = 0.0100), while lagged Banking returns have a positive association (β = 0.1893, p = 0.0069), indicating short-run persistence in Banking returns.

** Overall, the results suggest that cross-sector interdependence is stronger within the same trading period than through one-day lagged return transmission.

### Model Diagnostics

```r
## Diagnostics for stable model
roots(VAR1, modulus = TRUE) ## values to be less than one 

## Diagnostic test for VAR serial correlation
serial1 = serial.test(VAR1, lags.pt = 10, type= "PT.asymptotic")
serial1

##Diagnostic test for VAR heteroscedasticity 

ARCH1 = arch.test(VAR1, lags.multi = 10, multivariate.only = TRUE)
ARCH1

## Diagnostic for VAR Normality
NORM1 = normality.test(VAR1, multivariate.only = TRUE)
NORM1
## Diagnostic for stability 
stability1 = stability(VAR1, type = "OLS-CUSUM")
stability1

plot(stability1)
```
#### Interpretation:

** The OLS-CUSUM stability plots indicate that the empirical fluctuation processes for all five equations remain within the critical boundaries throughout the sample period. This suggests that there is no significant evidence of parameter instability or structural breaks in the estimated relationships. Although some fluctuations are visible, particularly for the Auto equation during the later part of the sample, none of the processes crosses the critical boundaries. Hence, the estimated model demonstrates satisfactory parameter stability over the study period.

### Granger-Causality test

```r
Granger1 = causality(VAR1, cause = "ret_auto")
Granger1
```
```r
Granger2 = causality(VAR1, cause = "ret_health")
Granger2
```
```r
Granger3 = causality(VAR1, cause = "ret_fs")
Granger3
```
```r
Granger4 = causality(VAR1, cause = "ret_bank")
Granger4
```
```r
Granger5 = causality(VAR1, cause = "ret_fmcg")
Granger5
```

### Impulse Response Function

```r
AUTOIRF_1 = irf(VAR1, impulse = "ret_bank", response  = "ret_auto",
              n.ahead = 10, boot =TRUE) 

plot(AUTOIRF_1, ylab = "ret_auto", mail = "shock from ret_bank")
```

```r
AUTOIRF_2 = irf(VAR1, impulse = "ret_auto", response  = "ret_bank",
              n.ahead = 10, boot =TRUE) 

plot(AUTOIRF_2, ylab = "ret_bank", mail = "shock from ret_auto")
```

#### Interpretation:

** 1 std.dev shock given to bank auto responded negatively for first 2 period and becomes positive for 3rd day.
** 1 std.dev shock given to auto, bank responded positively for first 3 period and then market corrected.

```r
IRF_ALL = irf(VAR1, impulse = colnames(sec_y), 
              response = colnames(sec_y),
              n.ahead = 10,
              boot = TRUE, 
              ci = 0.95,
              runs = 100)
IRF_ALL
```
### Variance Decomposition

```r
FEVDI = fevd(VAR1, n.ahead = 10)
```

```r
plot(FEVDI)
```

```r
FEVDI[["ret_health"]]
FEVDI[["ret_fmcg"]]
FEVDI[["ret_fs"]]
FEVDI[["ret_auto"]]
FEVDI["ret_bank"]
```
#### Interpretation:

* The FEVD results indicate that the forecast error variance of Healthcare returns is predominantly explained by its own innovations. At the first forecast horizon, 100% of the variance is attributable to shocks in Healthcare itself. By the second horizon, approximately 99.95% remains attributable to Healthcare shocks, while FMCG and Bank contribute only around 0.023% and 0.022%, respectively. The contributions of FS and Auto are negligible. From the third horizon onward, the decomposition remains virtually unchanged, indicating that cross-sector shocks have very limited explanatory power for the forecast error variance of Healthcare returns within the model.


```r
##VAR Forecasting
forecast=predict(VAR1,n.ahead = 10,cl=0.95)
fanchart(forecast,names = "ret_auto")
fanchart(forecast,names = "ret_bank")
```

---

## Part III – GARCH models

### Data Import

> **How to interpret:** Check the workbook path and confirm that the expected Nifty, Gold and Crude columns are present.

```r
data <- read_excel("D:/Msc BA Classnotes/Sem 3/Time series Analysis/GARCH.xlsx")
head(data)
```

### Converting Data Into Zoo Series

> **How to interpret:** Confirm that dates are parsed correctly and all price columns are numeric.

```r
data_g <- read.zoo(data.frame(data), FUN = as.Date, format ='%d%m%Y')
plot(data_g)
```

### Data Cleaning

> **How to interpret:** Check the remaining sample size after removing missing observations; the series should remain date-aligned.

```r
str(data_g)


data_g <- na.omit(data_g)
```

### Checking Stationarity On Raw Series

> **How to interpret:** For ADF, p > 0.05 means you fail to reject the unit-root null; p < 0.05 provides evidence against a unit root.

```r
adf.test(data_g$Nifty)


adf.test(data_g$Gold)


adf.test(data_g$Crude)
```

### Creating Return Series

> **How to interpret:** Log differences calculate continuously compounded returns. Prices must be positive and numeric.

```r
data_g$Nifty <- as.numeric(as.character(data_g$Nifty))


nifty_ret <- diff(log(data_g$Nifty))
gold_ret <- diff(log(data_g$Gold))
crude_ret <- diff(log(data_g$Crude))
```

### Checking Stationarity Of Return Series

> **How to interpret:** ADF p < 0.05 supports stationarity of the return series.

```r
adf.test(nifty_ret)


adf.test(gold_ret)


adf.test(crude_ret)
```

### Creating Dataframe Of Return Series

> **How to interpret:** Confirm that all return columns are aligned on the same dates before multivariate modelling.

```r
yy <- cbind.zoo(nifty_ret, gold_ret, crude_ret)
```

### Visualizing The Return Series

> **How to interpret:** Look for returns around zero, outliers and clusters of large movements; clusters may indicate volatility clustering.

```r
## Volatility clustering

plot(yy)
```

### Arch Effect

> **How to interpret:** The ARCH-LM null is no ARCH effects. p < 0.05 supports time-varying conditional variance and motivates GARCH modelling.

```r
ArchTest(nifty_ret)


ArchTest(gold_ret)


ArchTest(crude_ret)
```

### Serial Correlation Test

> **How to interpret:** The Ljung–Box null is no autocorrelation up to the selected lag. p < 0.05 indicates remaining serial correlation.

```r
Box.test(nifty_ret, lag = 10, type = "Ljung-Box")


Box.test(gold_ret, lag = 10, type = "Ljung-Box")


Box.test(crude_ret, lag = 10, type = "Ljung-Box")
```

### Normality Test

> **How to interpret:** The Jarque–Bera null is normality. p < 0.05 indicates departures from a normal distribution.

```r
jarque.bera.test(nifty_ret)


jarque.bera.test(gold_ret)


jarque.bera.test(crude_ret)
```

### Performing Garch Model

> **How to interpret:** GARCH models conditional variance. In GARCH(1,1), alpha captures new shocks and beta captures persistence from past conditional variance.

```r
NOTE:
```

### Estimating Using Variance Model

> **How to interpret:** The sGARCH(1,1) specification models volatility; Student-t errors allow heavier tails than normal errors.

```r
spec = ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
                  mean.model = list(armaOrder = c(0,0)), distribution.model = "std")
fit_1 = ugarchfit(spec= spec, data=nifty_ret)
fit_1


fit_2 = ugarchfit(spec= spec, data=gold_ret)
fit_2


fit_3 = ugarchfit(spec= spec, data=crude_ret)
fit_3
```

### Estimating Using Mean Model

> **How to interpret:** ARMA(1,1) terms model serial dependence in the conditional mean while sGARCH(1,1) models conditional variance.

```r
spec_1 = ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
                  mean.model = list(armaOrder = c(1,1)), distribution.model = "std")
fit_4 = ugarchfit(spec= spec_1, data=nifty_ret)
fit_4


fit_5 = ugarchfit(spec= spec_1, data=gold_ret)
fit_5


fit_6 = ugarchfit(spec= spec_1, data=crude_ret)
fit_6
```

### Diagnostic

> **How to interpret:** Inspect standardized residuals and squared residuals. Ideally, there is no meaningful residual autocorrelation or remaining volatility pattern.

```r
plot(fit_1, which= "all")
```

### Diagnostics

> **How to interpret:** For standardized residuals, Ljung–Box p > 0.05 suggests no remaining serial correlation; ARCH-LM p > 0.05 suggests no remaining ARCH effect.

```r
residuals_fit1 = residuals(fit_1, standardize =  TRUE)


Box.test(residuals_fit1, lag = 10, type = "Ljung-Box")


ArchTest(residuals_fit1, lags = 10)


##Forecast volatility 
forecast = ugarchforecast(fit_1, n.ahead = 10)
plot(forecast, which = 3)


plot(fit_2, which= "all")


residuals_fit2 = residuals(fit_2, standardize =  TRUE)


Box.test(residuals_fit2, lag = 10, type = "Ljung-Box")


ArchTest(residuals_fit2, lags = 10)


##Forecast volatility 
forecast = ugarchforecast(fit_2, n.ahead = 10)
plot(forecast, which = 3)


plot(fit_3, which= "all")


residuals_fit3 = residuals(fit_3, standardize =  TRUE)


Box.test(residuals_fit3, lag = 10, type = "Ljung-Box")


ArchTest(residuals_fit3, lags = 10)


##Forecast volatility 
forecast = ugarchforecast(fit_3, n.ahead = 10)
plot(forecast, which = 3)
```

### Egarch Model

> **How to interpret:** EGARCH can capture asymmetric responses to positive and negative shocks. Examine the asymmetry coefficient and its significance.

```r
spec_e = ugarchspec(variance.model = list(model = "eGARCH", garchOrder = c(1,1)),
                  mean.model = list(armaOrder = c(0,0)))
fit_7 = ugarchfit(spec= spec_e, data=nifty_ret)
fit_7
```

### Diagnostic Test

> **How to interpret:** Use residual ACF and Ljung–Box for serial correlation, and the ARCH test for remaining conditional heteroskedasticity.

```r
plot(fit_7, which = "all")


residuals_fit7 <- residuals(fit_7, standardize = TRUE)


acf(residuals_fit7, main = "ACF of Standardized Residuals")


Box.test(residuals_fit7, lag = 20, type = "Ljung-Box")



# ARCH test on residuals
ArchTest(residuals_fit7, lags = 12)


forecast <- ugarchforecast(fit_7, n.ahead=10)
plot(forecast, which = 3)


fit_8 = ugarchfit(spec= spec_e, data=gold_ret)
fit_8
```

### Diagnostic

> **How to interpret:** Inspect standardized residuals and squared residuals. Ideally, there is no meaningful residual autocorrelation or remaining volatility pattern.

```r
plot(fit_8, which = "all")


residuals_fit8 <- residuals(fit_8, standardize = TRUE)


acf(residuals_fit8, main = "ACF of Standardized Residuals")


Box.test(residuals_fit8, lag = 20, type = "Ljung-Box")


# ARCH test on residuals
ArchTest(residuals_fit8, lags = 12)


forecast <- ugarchforecast(fit_8, n.ahead=10)
plot(forecast, which = 3)


fit_9 = ugarchfit(spec= spec_e, data=crude_ret)
fit_9


plot(fit_9, which = "all")


residuals_fit9 <- residuals(fit_9, standardize = TRUE)


acf(residuals_fit9, main = "ACF of Standardized Residuals")


Box.test(residuals_fit9, lag = 20, type = "Ljung-Box")


# ARCH test on residuals
ArchTest(residuals_fit9, lags = 12)


forecast <- ugarchforecast(fit_9, n.ahead=10)
plot(forecast, which = 3)



nse_tgarch =ugarchspec(variance.model = list(model="gjrGARCH", garchOrder = c(1,1)), 
                       mean.model = list(armaOrder = c(0,0)))

nse_tgarch.fit_10 = ugarchfit(nse_tgarch, data = nifty_ret)

nse_tgarch.fit_10
```

### Model Diagnostics

> **How to interpret:** Apply diagnostics to the corresponding series. The source appears to reuse residual objects in two tests; verify those references before interpreting Gold and Crude results.

```r
plot(nse_tgarch.fit_10, which = "all")


residuals_fit10 <- residuals(nse_tgarch.fit_10, standardize = TRUE)


acf(residuals_fit10, main = "ACF of Standardized Residuals")


Box.test(residuals_fit10, lag = 20, type = "Ljung-Box")


# ARCH test on residuals
ArchTest(residuals_fit10, lags = 12)


forecast <- ugarchforecast(nse_tgarch.fit_10, n.ahead=10)
plot(forecast, which = 3)



nse_tgarch.fit_11 = ugarchfit(nse_tgarch, data = gold_ret)

nse_tgarch.fit_11


PERFORMING MODEL DIAGNOSTICS

plot(nse_tgarch.fit_11, which = "all")


residuals_fit11 <- residuals(nse_tgarch.fit_11, standardize = TRUE)


acf(residuals_fit11, main = "ACF of Standardized Residuals")


Box.test(residuals_fit11, lag = 20, type = "Ljung-Box")


# ARCH test on residuals
ArchTest(residuals_fit10, lags = 12)


forecast <- ugarchforecast(nse_tgarch.fit_11, n.ahead=10)
plot(forecast, which = 3)



nse_tgarch.fit_12 = ugarchfit(nse_tgarch, data = crude_ret)

nse_tgarch.fit_12


PERFORMING MODEL DIAGNOSTICS

plot(nse_tgarch.fit_12, which = "all")


residuals_fit12 <- residuals(nse_tgarch.fit_12, standardize = TRUE)


acf(residuals_fit12, main = "ACF of Standardized Residuals")


Box.test(residuals_fit11, lag = 20, type = "Ljung-Box")


# ARCH test on residuals
ArchTest(residuals_fit12, lags = 12)


forecast <- ugarchforecast(nse_tgarch.fit_12, n.ahead=10)
plot(forecast, which = 3)
```

### Multivariate Garch

> **How to interpret:** DCC-GARCH estimates time-varying conditional variances and correlations. `rcov()` returns conditional covariance matrices and `rcor()` returns dynamic correlations.

```r
uspec = ugarchspec(
  mean.model = list(armaOrder = c(0,0)),
  variance.model = list(garchOrder = c(1,1), model = "sGARCH"),
  distribution.model = "norm"
)


specm = dccspec(uspec = multispec(replicate(3,uspec)), dccOrder = c(1,1), 
                distribution = "mvnorm")


modelfit = dccfit(specm,data = data.frame(yy))
modelfit


H = rcov(modelfit)
R = rcor(modelfit)


plot(rcov(modelfit))


plot(rcor(modelfit))


num_assets <- ncol(yy)


# Plotting individual correlation pairs

for (i in 1:(num_assets - 1)) {
  for (j in (i + 1):num_assets) {
    plot(R[i, j, ],
         type = "l",
         main = paste("Dynamic Conditional Correlation:", colnames(yy)[i], "vs", colnames(yy)[j]),
         xlab = "Time",
         ylab = "Correlation")
    grid()
  }
}


#BEKK model




# Convert returns to matrix
Y <- as.matrix(yy)


# BEKK(1,1) specification
spec <- bekk_spec(
  model = list(
    type = "bekk",
    asymmetric = FALSE
  )
)


# Estimate BEKK-GARCH with t-ratios
fit <- bekk_fit(
  spec,
  Y,
  QML_t_ratios = TRUE
)


# Model results
summary(fit)


#P value estiamtion Shock spillover
A_pvalue <- 2 * (1 - pnorm(abs(fit$A_t)))
cat("\nA Matrix - Estimates\n")


print(fit$A)


cat("\nA Matrix - t-values\n")


print(fit$A_t)


cat("\nA Matrix - p-values\n")


print(A_pvalue)


#P value estiamtion volatility spillover
G_pvalue <- 2 * (1 - pnorm(abs(fit$G_t)))
cat("\nG Matrix - Estimates\n")


print(fit$G)


cat("\nG Matrix - t-values\n")


print(fit$G_t)


cat("\nG Matrix - p-values\n")

print(G_pvalue)
