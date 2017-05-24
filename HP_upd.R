# Competition
# House Prices: Advanced Regression Techniques
# Goal: Predict sales prices and practice feature engineering, RFs, and gradient boosting
# Link: https://www.kaggle.com/c/house-prices-advanced-regression-techniques

#### Data preparation ####

### Load and preprocess the data----

library(dplyr)
library(tidyr)
library(Amelia)

getwd()
setwd('C:\\Users\\user\\Documents\\R projects\\Kaggle\\House Prices')

train <- read.csv("train.csv", stringsAsFactors = F)
missmap(train) # ploting missingness map
train[is.na(train)] <- 0
train <- train %>% 
            mutate_if(is.character, as.factor)

## Check whether there are any NA values

missmap(train)

### Split the training data into strain and stest----

test.n <- sample(1:nrow(train), nrow(train)/3, replace = F)
stest <- train[test.n,]
strain <- train[-test.n,]
rm(test.n)

### Evaluation metric function----

RMSE <- function(x,y){
  a <- sqrt(sum((log(x)-log(y))^2)/length(y))
  return(a)
}


#### Plotting SalesPrice distrution by The general zoning classification ####

train %>% ggplot(aes(x = MSZoning, y = SalePrice)) + 
            theme(legend.position="top", 
                  axis.text=element_text(size = 6)) +
            geom_point(aes(color = YearBuilt), alpha = 0.5, size = 1.5, 
                       position = position_jitter(width = 0.25, height = 0)) + 
            stat_summary(fun.y = median, fun.ymin = median, fun.ymax = median,
                         geom = "crossbar", width = 0.5) + # adding madian
            scale_y_continuous(labels = comma)


#### Regression tree ####

library(rpart)
library(rpart.plot)

### Fit a regression tree on strain dataset----

set.seed(415)

tree <- rpart(SalePrice ~., data = strain, method = "anova")
rpart.plot(tree) # ploting the tree

printcp(tree) # display the results
plotcp(tree) # visualize cross-validation results
summary(tree) # detailed summary of splits

## Check the model using stest dataset

predict <- predict(tree, stest)
RMSE_tree <- round(RMSE(predict, stest$SalePrice), digits = 3) ## RMSE = 0.236
plot1 <- predict-stest$SalePrice


#### Random Forest ####

library(randomForest)

### Fit a Random Forest on strain dataset----

set.seed(415)

RF1 <- randomForest(strain %>% select(-SalePrice), strain$SalePrice,
                      ntree = 200,
                      nodesize = 7,
                      importance = T)

## Check the model using stest dataset

predict <- predict(RF1, stest)
RMSE_RF1 <- round(RMSE(predict, stest$SalePrice), digits = 3) ## RMSE = 0.134

### Feature Importance----

varImp <- importance(RF1)
featureImportance <- data.frame(Feature=row.names(varImp), Importance=varImp[,1])

## Plotting Feature Importance

featureImportance %>% ggplot(aes(x=reorder(Feature, Importance), y=Importance)) +
                        geom_bar(stat="identity", fill="#53cfff") +
                        coord_flip() + 
                        theme_light(base_size=20) +
                        xlab("") +
                        ylab("Importance") + 
                        ggtitle("Random Forest Feature Importance") +
                        theme(plot.title=element_text(size=18))

### Improve the model with feature importance----

set.seed(415)

RF2 <- randomForest(SalePrice ~ GrLivArea + BsmtFinSF1 + Neighborhood + OverallQual + 
                         + TotalBsmtSF + GarageCars + GarageArea + X2ndFlrSF + X1stFlrSF, train,
                       ntree = 200,
                       nodesize = 7,
                       importance = T)

## Check the model using stest dataset

predict <- predict(RF2, stest)
RMSE_RF2 <- round(RMSE(predict, stest$SalePrice), digits = 3) ## RMSE = 0.08 (Better result)
plot2 <- predict-stest$SalePrice


#### GBM ####

library(gbm)

### Fit GBM on strain dataset----

set.seed(415)

gbm <- gbm(SalePrice ~., data = strain, distribution = "laplace",
              shrinkage = 0.05,
              interaction.depth = 5,
              bag.fraction = 0.66,
              n.minobsinnode = 1,
              cv.folds = 100,
              keep.data = F,
              verbose = F,
              n.trees = 500)

## Check the model using stest dataset

predict <- predict(gbm, stest, n.trees = 500)
RMSE_gbm <- round(RMSE(predict, stest$SalePrice), digits = 3) ## RMSE = 0.134
plot3 <- predict-stest$SalePrice


#### Plotting The difference between predict and real values ####

data_plot <- data.frame("Regression tree" = plot1,
                        "Random forest" = plot2,
                        "GBM" = plot3)
data_plot$Id <- row.names(data_plot)
data_plot <- gather(data_plot, method, value, - Id)
data_plot$method <- as.factor(data_plot$method)
levels(data_plot$method) <- c(paste0("GBM (", RMSE_gbm, ")"), 
                              paste0("Random Forest (", RMSE_RF2, ")"),
                              paste0("Regression Tree (", RMSE_tree, ")"))

ggplot(data_plot, aes(x = Id, y = value, colour = method))+
  geom_point(alpha = 0.7, size = 2)+
  ggtitle("The difference between predict and real prices")+
  labs(x = "Buyer Id", y = "The difference between prices", colour = " ")+
  scale_x_discrete(breaks = c(0))+
  theme(legend.position = "top",
        legend.text = element_text(size = 12),
        axis.text.x = element_blank(), 
        axis.title.x = element_text(size = 14),
        axis.text.y = element_text(size = 14), 
        axis.title.y = element_text(size = 14),
        title = element_text(size = 16))

## Conclusion
## Random Forest gave the most accurate results:
#  - Regression Tree RMSE = 0.236;
#  - Random Forest RMSE = 0.08;
#  - GBM RMSE = 0.134.
## But using the Kaggle's test dataset GBM gave the most accurate results